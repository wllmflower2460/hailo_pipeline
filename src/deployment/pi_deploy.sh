#!/bin/bash
# HailoRT TCN Inference Sidecar - Raspberry Pi Deployment Script
# Automates deployment of the inference sidecar on Raspberry Pi + Hailo-8

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_NAME="hailo-tcn-inference"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Hailo device
    if [[ ! -e /dev/hailo0 ]]; then
        log_warning "Hailo device /dev/hailo0 not found. Sidecar will run in stub mode."
    else
        log_success "Hailo device detected: /dev/hailo0"
    fi
    
    # Check model artifact
    if [[ ! -f "$PROJECT_ROOT/artifacts/tcn_encoder.hef" ]]; then
        log_warning "HEF model not found at $PROJECT_ROOT/artifacts/tcn_encoder.hef"
        log_info "Sidecar will run in stub mode until model is provided."
    else
        log_success "HEF model found: $(du -h "$PROJECT_ROOT/artifacts/tcn_encoder.hef" | cut -f1)"
    fi
    
    # Check available memory
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEM_GB -lt 2 ]]; then
        log_warning "Low memory detected: ${MEM_GB}GB. Recommended: 4GB+ for optimal performance."
    fi
    
    log_success "Prerequisites check completed"
}

# Function to build the Docker image
build_image() {
    log_info "Building HailoRT sidecar Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build with progress output
    docker-compose -f "$COMPOSE_FILE" build --no-cache hailo-inference
    
    if [[ $? -eq 0 ]]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to deploy the service
deploy_service() {
    log_info "Deploying HailoRT inference sidecar..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing service if running
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Start the service
    docker-compose -f "$COMPOSE_FILE" up -d hailo-inference
    
    if [[ $? -eq 0 ]]; then
        log_success "Service deployed successfully"
    else
        log_error "Failed to deploy service"
        exit 1
    fi
}

# Function to wait for service to be healthy
wait_for_health() {
    log_info "Waiting for service to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f http://localhost:9000/healthz > /dev/null; then
            log_success "Service is healthy and ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for service..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Service failed to become healthy within $(($max_attempts * 2)) seconds"
    return 1
}

# Function to run smoke tests
run_smoke_tests() {
    log_info "Running deployment smoke tests..."
    
    # Test 1: Health check
    log_info "Test 1: Health check"
    if response=$(curl -s http://localhost:9000/healthz); then
        log_success "✓ Health check passed"
        echo "  Response: $response"
    else
        log_error "✗ Health check failed"
        return 1
    fi
    
    # Test 2: Root endpoint
    log_info "Test 2: Root endpoint"
    if curl -s http://localhost:9000/ > /dev/null; then
        log_success "✓ Root endpoint accessible"
    else
        log_error "✗ Root endpoint failed"
        return 1
    fi
    
    # Test 3: Metrics endpoint
    log_info "Test 3: Metrics endpoint"
    if curl -s http://localhost:9000/metrics | grep -q "hailo_"; then
        log_success "✓ Metrics endpoint working"
    else
        log_error "✗ Metrics endpoint failed"
        return 1
    fi
    
    # Test 4: Test inference (if available)
    log_info "Test 4: Test inference"
    if curl -s -X POST http://localhost:9000/test > /dev/null; then
        log_success "✓ Test inference working"
    else
        log_warning "⚠ Test inference not available (may be in stub mode)"
    fi
    
    log_success "All smoke tests completed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Container status
    echo "Container Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    # Service logs (last 20 lines)
    echo "Recent Logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 hailo-inference
    echo
    
    # Resource usage
    echo "Resource Usage:"
    docker stats "$SERVICE_NAME" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo
    
    # Service endpoints
    echo "Service Endpoints:"
    echo "  Health Check: http://localhost:9000/healthz"
    echo "  API Docs:     http://localhost:9000/docs"
    echo "  Metrics:      http://localhost:9000/metrics"
    echo "  Test:         http://localhost:9000/test"
    echo
    
    # EdgeInfer integration info
    echo "EdgeInfer Integration:"
    echo "  Set in EdgeInfer environment:"
    echo "    MODEL_BACKEND_URL=http://localhost:9000"
    echo "    USE_REAL_MODEL=true"
    echo
}

# Function to clean up deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Remove image (optional)
    if [[ "${1:-}" == "--remove-images" ]]; then
        docker-compose -f "$COMPOSE_FILE" down --rmi all
        log_info "Docker images removed"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "🚀 Starting HailoRT TCN Inference Sidecar Deployment"
    
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            build_image
            deploy_service
            wait_for_health
            run_smoke_tests
            show_status
            log_success "🎉 Deployment completed successfully!"
            ;;
        status)
            show_status
            ;;
        test)
            run_smoke_tests
            ;;
        cleanup)
            cleanup "${2:-}"
            ;;
        logs)
            docker-compose -f "$COMPOSE_FILE" logs -f hailo-inference
            ;;
        restart)
            docker-compose -f "$COMPOSE_FILE" restart hailo-inference
            wait_for_health
            log_success "Service restarted"
            ;;
        *)
            echo "Usage: $0 {deploy|status|test|cleanup|logs|restart}"
            echo
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  status   - Show current status"
            echo "  test     - Run smoke tests"
            echo "  cleanup  - Stop and remove containers"
            echo "  logs     - Follow service logs"
            echo "  restart  - Restart service"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"