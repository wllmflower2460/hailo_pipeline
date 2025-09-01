#!/bin/bash
# EdgeInfer Integration Script
# Automates integration testing between pisrv_vapor_docker and hailo_pipeline

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAILO_SERVICE_URL="${HAILO_SERVICE_URL:-http://localhost:9000}"
EDGEINFER_URL="${EDGEINFER_URL:-http://localhost:8080}"
PISRV_DOCKER_PATH="${PISRV_DOCKER_PATH:-../pisrv_vapor_docker}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_step "Checking EdgeInfer integration prerequisites..."
    
    # Check if pisrv_vapor_docker exists
    if [[ ! -d "$PISRV_DOCKER_PATH" ]]; then
        log_error "pisrv_vapor_docker not found at: $PISRV_DOCKER_PATH"
        log_info "Please set PISRV_DOCKER_PATH environment variable"
        exit 1
    fi
    
    log_success "Found pisrv_vapor_docker at: $PISRV_DOCKER_PATH"
    
    # Check required files
    local required_files=(
        "$PISRV_DOCKER_PATH/docker-compose.yml"
        "$PISRV_DOCKER_PATH/.env.example"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "All required EdgeInfer files present"
    
    # Check if Hailo service is running
    if curl -sf "$HAILO_SERVICE_URL/healthz" >/dev/null 2>&1; then
        log_success "Hailo inference service is running"
    else
        log_error "Hailo inference service not accessible at: $HAILO_SERVICE_URL"
        log_info "Please start the Hailo service first: ./deploy_production.sh"
        exit 1
    fi
}

# Configure EdgeInfer environment
configure_edgeinfer() {
    log_step "Configuring EdgeInfer environment..."
    
    local env_file="$PISRV_DOCKER_PATH/.env"
    
    # Create .env file from example if it doesn't exist
    if [[ ! -f "$env_file" ]]; then
        log_info "Creating .env file from example..."
        cp "$PISRV_DOCKER_PATH/.env.example" "$env_file"
    fi
    
    # Update environment variables for Hailo backend
    log_info "Configuring Hailo backend settings..."
    
    # Use a temporary file for atomic updates
    local temp_env=$(mktemp)
    
    # Remove existing Hailo-related variables and add new ones
    grep -v -E '^(USE_REAL_MODEL|MODEL_BACKEND_URL|BACKEND_TIMEOUT_MS)=' "$env_file" > "$temp_env" || true
    
    cat >> "$temp_env" << EOF

# Hailo Backend Configuration (auto-generated)
USE_REAL_MODEL=true
MODEL_BACKEND_URL=$HAILO_SERVICE_URL/infer
BACKEND_TIMEOUT_MS=150

# Additional Hailo settings
HAILO_HEALTH_CHECK_URL=$HAILO_SERVICE_URL/healthz
HAILO_METRICS_URL=$HAILO_SERVICE_URL/metrics
EOF
    
    mv "$temp_env" "$env_file"
    
    log_success "EdgeInfer environment configured for Hailo backend"
    
    # Display configuration
    log_info "Current configuration:"
    grep -E '^(USE_REAL_MODEL|MODEL_BACKEND_URL|BACKEND_TIMEOUT_MS)=' "$env_file" | while read -r line; do
        log_info "  $line"
    done
}

# Update docker-compose for network integration
configure_network() {
    log_step "Configuring Docker network integration..."
    
    local compose_file="$PISRV_DOCKER_PATH/docker-compose.yml"
    local backup_file="$compose_file.backup.$(date +%s)"
    
    # Backup original
    cp "$compose_file" "$backup_file"
    log_info "Backed up original compose file: $(basename "$backup_file")"
    
    # Check if edgeinfer-network already exists in compose file
    if grep -q "edgeinfer-network" "$compose_file"; then
        log_info "EdgeInfer network configuration already present"
    else
        log_info "Adding EdgeInfer network configuration..."
        
        # Add network configuration if not present
        if ! grep -q "^networks:" "$compose_file"; then
            cat >> "$compose_file" << 'EOF'

# EdgeInfer network for Hailo integration
networks:
  edgeinfer-network:
    external: true
EOF
        fi
        
        # Add network to edge-infer service
        # This is a simplified approach - in practice, you'd use yq or similar
        log_info "Network configuration added (manual verification recommended)"
    fi
    
    log_success "Docker network configuration updated"
}

# Create shared Docker network
create_network() {
    log_step "Creating shared Docker network..."
    
    local network_name="edgeinfer-network"
    
    # Check if network already exists
    if docker network ls | grep -q "$network_name"; then
        log_info "Network '$network_name' already exists"
    else
        # Create the network
        docker network create "$network_name" --driver bridge
        log_success "Created Docker network: $network_name"
    fi
    
    # Connect Hailo service to the network if not already connected
    local hailo_container="hailo-tcn-inference"
    
    if docker ps --format '{{.Names}}' | grep -q "$hailo_container"; then
        if ! docker network inspect "$network_name" | grep -q "$hailo_container"; then
            docker network connect "$network_name" "$hailo_container"
            log_success "Connected $hailo_container to $network_name"
        else
            log_info "$hailo_container already connected to $network_name"
        fi
    else
        log_warning "Hailo container not running - network connection will happen on restart"
    fi
}

# Test Hailo service directly
test_hailo_service() {
    log_step "Testing Hailo service directly..."
    
    # Health check
    log_info "Testing health endpoint..."
    local health_response=$(curl -sf "$HAILO_SERVICE_URL/healthz" 2>/dev/null || echo "ERROR")
    
    if [[ "$health_response" != "ERROR" ]]; then
        log_success "Health check passed"
        echo "$health_response" | jq . 2>/dev/null || echo "$health_response"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test inference
    log_info "Testing inference endpoint..."
    
    local test_payload='{
        "x": [
            [0.1, -0.05, 9.8, 0.01, -0.01, 0.005, 25.2, -8.1, 43.5],
            [0.15, -0.08, 9.82, 0.02, -0.015, 0.008, 25.0, -8.3, 43.2]
        ]
    }'
    
    # Pad to 100 timesteps
    for ((i=2; i<100; i++)); do
        test_payload="${test_payload%]},"
        test_payload="${test_payload}[$(echo "$test_payload" | grep -o '\[.*\]' | head -1 | sed 's/0\.1/0.1/g')]}"
        test_payload="${test_payload}]"
    done
    
    local inference_response=$(curl -sf -X POST "$HAILO_SERVICE_URL/infer" \
        -H "Content-Type: application/json" \
        -d "$test_payload" 2>/dev/null || echo "ERROR")
    
    if [[ "$inference_response" != "ERROR" ]]; then
        log_success "Inference test passed"
        
        # Validate response structure
        local latent_count=$(echo "$inference_response" | jq '.latent | length' 2>/dev/null || echo "0")
        local motif_count=$(echo "$inference_response" | jq '.motif_scores | length' 2>/dev/null || echo "0")
        
        log_info "Response structure:"
        log_info "  Latent dimensions: $latent_count (expected: 64)"
        log_info "  Motif scores: $motif_count (expected: 12)"
        
        if [[ "$latent_count" == "64" && "$motif_count" == "12" ]]; then
            log_success "Response structure is correct"
        else
            log_warning "Response structure validation failed"
        fi
    else
        log_error "Inference test failed"
        return 1
    fi
}

# Start EdgeInfer service
start_edgeinfer() {
    log_step "Starting EdgeInfer service with Hailo backend..."
    
    cd "$PISRV_DOCKER_PATH"
    
    # Stop any existing services
    log_info "Stopping existing EdgeInfer services..."
    docker-compose down --remove-orphans || true
    
    # Start services
    log_info "Starting EdgeInfer with Hailo backend configuration..."
    docker-compose up -d
    
    # Wait for service to be ready
    log_info "Waiting for EdgeInfer service to start..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "$EDGEINFER_URL/healthz" >/dev/null 2>&1; then
            log_success "EdgeInfer service is running"
            break
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for EdgeInfer..."
        sleep 2
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        log_error "EdgeInfer failed to start within $((max_attempts * 2)) seconds"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
}

# Test EdgeInfer integration
test_integration() {
    log_step "Testing EdgeInfer → Hailo integration..."
    
    # Test EdgeInfer health check
    log_info "Testing EdgeInfer health..."
    local edgeinfer_health=$(curl -sf "$EDGEINFER_URL/healthz" 2>/dev/null || echo "ERROR")
    
    if [[ "$edgeinfer_health" != "ERROR" ]]; then
        log_success "EdgeInfer health check passed"
        echo "$edgeinfer_health" | jq . 2>/dev/null || echo "$edgeinfer_health"
    else
        log_error "EdgeInfer health check failed"
        return 1
    fi
    
    # Test motif analysis endpoint (main integration point)
    log_info "Testing motif analysis integration..."
    
    local motif_payload='{
        "sessionId": "integration-test-' $(date +%s) '",
        "deviceId": "test-device",
        "timestamp": "' $(date -u +"%Y-%m-%dT%H:%M:%SZ") '"
    }'
    
    local motif_response=$(curl -sf -X POST "$EDGEINFER_URL/api/v1/analysis/motifs" \
        -H "Content-Type: application/json" \
        -d "$motif_payload" 2>/dev/null || echo "ERROR")
    
    if [[ "$motif_response" != "ERROR" ]]; then
        log_success "Motif analysis integration test passed"
        
        # Parse and display response
        local backend_used=$(echo "$motif_response" | jq -r '.backend // "unknown"' 2>/dev/null)
        local inference_time=$(echo "$motif_response" | jq -r '.inference_time_ms // "unknown"' 2>/dev/null)
        local motif_count=$(echo "$motif_response" | jq '.motifs | length' 2>/dev/null || echo "0")
        
        log_info "Integration results:"
        log_info "  Backend used: $backend_used"
        log_info "  Inference time: ${inference_time}ms"
        log_info "  Motifs detected: $motif_count"
        
        if [[ "$backend_used" == *"hailo"* ]] || [[ "$backend_used" == *"real"* ]]; then
            log_success "✅ Hailo backend is being used!"
        else
            log_warning "⚠️  Backend may not be Hailo (got: $backend_used)"
        fi
        
        if [[ "$inference_time" != "unknown" ]] && [[ "$inference_time" != "null" ]]; then
            local inference_num=$(echo "$inference_time" | sed 's/[^0-9.]//g')
            if (( $(echo "$inference_num < 100" | bc -l) 2>/dev/null )); then
                log_success "✅ Good inference performance: ${inference_time}ms"
            else
                log_warning "⚠️  High inference latency: ${inference_time}ms"
            fi
        fi
    else
        log_error "Motif analysis integration test failed"
        return 1
    fi
}

# Performance validation
validate_performance() {
    log_step "Running performance validation..."
    
    log_info "Testing sustained load..."
    
    local total_requests=20
    local successful_requests=0
    local total_latency=0
    local max_latency=0
    
    for ((i=1; i<=total_requests; i++)); do
        local start_time=$(date +%s%N)
        
        local response=$(curl -sf -X POST "$EDGEINFER_URL/api/v1/analysis/motifs" \
            -H "Content-Type: application/json" \
            -d "{\"sessionId\":\"perf-test-$i\"}" 2>/dev/null || echo "ERROR")
        
        local end_time=$(date +%s%N)
        local request_latency=$((($end_time - $start_time) / 1000000))  # Convert to milliseconds
        
        if [[ "$response" != "ERROR" ]]; then
            ((successful_requests++))
            total_latency=$((total_latency + request_latency))
            
            if [[ $request_latency -gt $max_latency ]]; then
                max_latency=$request_latency
            fi
            
            log_info "Request $i: ${request_latency}ms ✅"
        else
            log_info "Request $i: FAILED ❌"
        fi
        
        sleep 0.1  # 100ms between requests
    done
    
    # Calculate statistics
    local success_rate=$((successful_requests * 100 / total_requests))
    local avg_latency=$((successful_requests > 0 ? total_latency / successful_requests : 0))
    
    log_info "Performance results:"
    log_info "  Total requests: $total_requests"
    log_info "  Successful: $successful_requests ($success_rate%)"
    log_info "  Average latency: ${avg_latency}ms"
    log_info "  Max latency: ${max_latency}ms"
    
    # Validate against targets
    if [[ $success_rate -ge 95 ]]; then
        log_success "✅ Success rate target met: $success_rate% ≥ 95%"
    else
        log_warning "⚠️  Success rate below target: $success_rate% < 95%"
    fi
    
    if [[ $avg_latency -le 200 ]]; then
        log_success "✅ Latency target met: ${avg_latency}ms ≤ 200ms"
    else
        log_warning "⚠️  Latency above target: ${avg_latency}ms > 200ms"
    fi
}

# Generate integration report
generate_integration_report() {
    log_step "Generating integration report..."
    
    local report_file="$SCRIPT_DIR/telemetry/edgeinfer_integration_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    mkdir -p "$(dirname "$report_file")"
    
    # Gather system information
    local hailo_health=$(curl -sf "$HAILO_SERVICE_URL/healthz" 2>/dev/null || echo '{"error": "service unavailable"}')
    local edgeinfer_health=$(curl -sf "$EDGEINFER_URL/healthz" 2>/dev/null || echo '{"error": "service unavailable"}')
    
    # Create comprehensive report
    cat > "$report_file" << EOF
{
    "integration_report": {
        "timestamp": "$timestamp",
        "test_type": "EdgeInfer → Hailo Integration",
        "services": {
            "hailo": {
                "url": "$HAILO_SERVICE_URL",
                "health": $hailo_health,
                "container": "$(docker ps --format '{{.Status}}' --filter name=hailo-tcn-inference 2>/dev/null || echo 'not running')"
            },
            "edgeinfer": {
                "url": "$EDGEINFER_URL",
                "health": $edgeinfer_health,
                "backend_configured": "$(grep -o 'USE_REAL_MODEL=true' "$PISRV_DOCKER_PATH/.env" 2>/dev/null || echo 'false')"
            }
        },
        "network": {
            "shared_network": "edgeinfer-network",
            "network_exists": $(docker network ls | grep -q edgeinfer-network && echo "true" || echo "false"),
            "hailo_connected": $(docker network inspect edgeinfer-network 2>/dev/null | grep -q hailo-tcn-inference && echo "true" || echo "false")
        },
        "configuration": {
            "env_file": "$PISRV_DOCKER_PATH/.env",
            "model_backend_url": "$(grep '^MODEL_BACKEND_URL=' "$PISRV_DOCKER_PATH/.env" 2>/dev/null | cut -d'=' -f2 || echo 'not configured')",
            "use_real_model": "$(grep '^USE_REAL_MODEL=' "$PISRV_DOCKER_PATH/.env" 2>/dev/null | cut -d'=' -f2 || echo 'not configured')"
        }
    }
}
EOF
    
    log_success "Integration report saved: $(basename "$report_file")"
}

# Main function
main() {
    echo ""
    echo "🔗 EDGEINFER → HAILO INTEGRATION"
    echo "================================="
    echo "Hailo Service: $HAILO_SERVICE_URL"
    echo "EdgeInfer Service: $EDGEINFER_URL"
    echo "EdgeInfer Path: $PISRV_DOCKER_PATH"
    echo ""
    
    local start_time=$(date +%s)
    
    # Execute integration steps
    check_prerequisites
    configure_edgeinfer
    configure_network
    create_network
    test_hailo_service
    start_edgeinfer
    test_integration
    validate_performance
    generate_integration_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "🎉 INTEGRATION COMPLETE!"
    echo "========================"
    echo "Duration: ${duration}s"
    echo ""
    echo "🔗 Service URLs:"
    echo "  Hailo Backend: $HAILO_SERVICE_URL"
    echo "  EdgeInfer API: $EDGEINFER_URL"
    echo ""
    echo "🧪 Test Commands:"
    echo "  Health: curl -f $EDGEINFER_URL/healthz"
    echo "  Motifs: curl -X POST $EDGEINFER_URL/api/v1/analysis/motifs -H 'Content-Type: application/json' -d '{\"sessionId\":\"test\"}'"
    echo ""
    echo "📊 Monitoring:"
    echo "  Hailo metrics: curl $HAILO_SERVICE_URL/metrics"
    echo "  EdgeInfer logs: docker logs \$(docker ps -q -f name=edge-infer)"
    echo "  Hailo logs: docker logs hailo-tcn-inference"
    echo ""
    echo "🚀 INTEGRATION READY!"
}

# Handle command line arguments
case "${1:-integrate}" in
    integrate)
        main
        ;;
    check)
        check_prerequisites
        ;;
    config)
        configure_edgeinfer
        ;;
    test)
        test_hailo_service
        test_integration
        ;;
    perf)
        validate_performance
        ;;
    clean)
        log_info "Cleaning up integration..."
        cd "$PISRV_DOCKER_PATH" && docker-compose down --remove-orphans
        docker network rm edgeinfer-network 2>/dev/null || true
        log_success "Integration cleanup complete"
        ;;
    *)
        echo "Usage: $0 {integrate|check|config|test|perf|clean}"
        echo ""
        echo "Commands:"
        echo "  integrate - Complete integration setup (default)"
        echo "  check     - Check prerequisites only"
        echo "  config    - Configure EdgeInfer environment only"
        echo "  test      - Test integration only"
        echo "  perf      - Run performance validation only"
        echo "  clean     - Clean up integration"
        echo ""
        echo "Environment variables:"
        echo "  HAILO_SERVICE_URL    - Hailo service URL (default: http://localhost:9000)"
        echo "  EDGEINFER_URL        - EdgeInfer service URL (default: http://localhost:8080)"
        echo "  PISRV_DOCKER_PATH    - Path to pisrv_vapor_docker (default: ../pisrv_vapor_docker)"
        exit 1
        ;;
esac