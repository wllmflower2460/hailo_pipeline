#!/bin/bash
# HailoRT TCN Inference Sidecar API Test Script
# Usage: ./test_sidecar_api.sh [base_url]
# Default: http://localhost:9000

set -e

BASE_URL="${1:-http://localhost:9000}"
TIMEOUT=10

echo "🚀 Testing HailoRT TCN Inference Sidecar API"
echo "Base URL: $BASE_URL"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="${5:-200}"
    
    echo -e "\n${BLUE}Testing: $name${NC}"
    echo "Endpoint: $method $endpoint"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$BASE_URL$endpoint" 2>/dev/null || echo -e "\n000")
    else
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT -X "$method" "$BASE_URL$endpoint" \
                  -H "Content-Type: application/json" -d "$data" 2>/dev/null || echo -e "\n000")
    fi
    
    # Split response and status code
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}✅ SUCCESS${NC} (HTTP $status_code)"
        if [ "$endpoint" = "/healthz" ] && [ -n "$body" ]; then
            echo "Health Status: $(echo "$body" | jq -r '.ok // "unknown"' 2>/dev/null || echo "parsing error")"
            echo "Model: $(echo "$body" | jq -r '.model // "unknown"' 2>/dev/null || echo "parsing error")"
            echo "Uptime: $(echo "$body" | jq -r '.uptime_s // "unknown"' 2>/dev/null || echo "parsing error")s"
        elif [ "$endpoint" = "/infer" ] && [ -n "$body" ]; then
            latent_count=$(echo "$body" | jq '.latent | length' 2>/dev/null || echo "unknown")
            motif_count=$(echo "$body" | jq '.motif_scores | length' 2>/dev/null || echo "unknown")
            echo "Latent dimensions: $latent_count"
            echo "Motif scores: $motif_count"
        fi
    else
        echo -e "${RED}❌ FAILED${NC} (HTTP $status_code)"
        if [ "$status_code" = "000" ]; then
            echo "Connection failed or timeout"
        elif [ -n "$body" ]; then
            echo "Response: $body"
        fi
        return 1
    fi
}

# Test basic connectivity
test_endpoint "Service Discovery" "GET" "/"

# Test health endpoint
test_endpoint "Health Check" "GET" "/healthz"

# Test status endpoint
test_endpoint "Status Check" "GET" "/status"

# Test documentation
test_endpoint "API Documentation" "GET" "/docs"

# Test OpenAPI spec
test_endpoint "OpenAPI Specification" "GET" "/openapi.json"

# Test metrics
test_endpoint "Prometheus Metrics" "GET" "/metrics"

# Test simple test endpoint
test_endpoint "Simple Test Endpoint" "POST" "/test"

# Test inference with each sample file
echo -e "\n${BLUE}Testing Inference with Sample Data${NC}"

if [ -f "data/test_samples/realistic_imu_sample.json" ]; then
    echo -e "\n${BLUE}Testing with realistic IMU data${NC}"
    realistic_data=$(cat data/test_samples/realistic_imu_sample.json)
    test_endpoint "Realistic IMU Inference" "POST" "/infer" "$realistic_data"
else
    echo -e "${RED}❌ realistic_imu_sample.json not found${NC}"
fi

if [ -f "data/test_samples/static_imu_sample.json" ]; then
    echo -e "\n${BLUE}Testing with static pattern IMU data${NC}"
    static_data=$(cat data/test_samples/static_imu_sample.json)
    test_endpoint "Static Pattern Inference" "POST" "/infer" "$static_data"
else
    echo -e "${RED}❌ static_imu_sample.json not found${NC}"
fi

if [ -f "data/test_samples/random_imu_sample.json" ]; then
    echo -e "\n${BLUE}Testing with random IMU data${NC}"
    random_data=$(cat data/test_samples/random_imu_sample.json)
    test_endpoint "Random IMU Inference" "POST" "/infer" "$random_data"
else
    echo -e "${RED}❌ random_imu_sample.json not found${NC}"
fi

# Load test
echo -e "\n${BLUE}Running Mini Load Test (5 requests)${NC}"
start_time=$(date +%s.%N)
success_count=0

for i in {1..5}; do
    if [ -f "data/test_samples/realistic_imu_sample.json" ]; then
        realistic_data=$(cat data/test_samples/realistic_imu_sample.json)
        response=$(curl -s -w "%{http_code}" --max-time $TIMEOUT -X POST "$BASE_URL/infer" \
                  -H "Content-Type: application/json" -d "$realistic_data" 2>/dev/null || echo "000")
        status_code=$(echo "$response" | tail -c 4)
        if [ "$status_code" = "200" ]; then
            success_count=$((success_count + 1))
        fi
    fi
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "unknown")
avg_time=$(echo "scale=2; $duration / 5 * 1000" | bc -l 2>/dev/null || echo "unknown")

echo -e "${GREEN}Load Test Results:${NC}"
echo "Successful requests: $success_count/5"
if [ "$avg_time" != "unknown" ]; then
    echo "Average time per request: ${avg_time}ms"
fi

echo -e "\n========================================"
echo -e "${GREEN}🎯 API Testing Complete${NC}"
echo -e "\nFor PiSrv testing, use the same endpoints with:"
echo "Base URL: http://[GPUSrv-IP]:9000"
echo -e "\nSample curl command for PiSrv:"
echo -e "${BLUE}curl -X POST http://[GPUSrv-IP]:9000/infer \\"
echo "  -H \"Content-Type: application/json\" \\"
echo -e "  -d @data/test_samples/realistic_imu_sample.json${NC}"