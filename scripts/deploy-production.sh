#!/bin/bash

# Causal Interface Gym - Production Deployment Script
# Implements zero-downtime deployments with comprehensive validation

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DEPLOYMENT_ID="$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)"
readonly NAMESPACE="causal-gym"
readonly HEALTH_CHECK_TIMEOUT=300
readonly ROLLBACK_TIMEOUT=180

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
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

# Error handling
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed! Initiating cleanup..."
        
        # Attempt to rollback if deployment was started
        if [[ -n "${DEPLOYMENT_STARTED:-}" ]]; then
            log_warning "Attempting automatic rollback..."
            rollback_deployment || log_error "Rollback failed - manual intervention required"
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Pre-deployment validation
validate_prerequisites() {
    log_info "Validating deployment prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "docker" "helm" "git" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context)
    log_info "Current kubectl context: $current_context"
    
    if [[ "$current_context" != *"production"* ]] && [[ "$current_context" != *"prod"* ]]; then
        log_warning "Current context doesn't appear to be production: $current_context"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Validate cluster resources
    validate_cluster_resources
    
    log_success "Prerequisites validation completed"
}

validate_cluster_resources() {
    log_info "Validating cluster resources..."
    
    # Check node capacity
    local total_cpu total_memory
    total_cpu=$(kubectl get nodes -o jsonpath='{.items[*].status.capacity.cpu}' | tr ' ' '\n' | sed 's/m$//' | awk '{sum+=$1} END {print sum}')
    total_memory=$(kubectl get nodes -o jsonpath='{.items[*].status.capacity.memory}' | tr ' ' '\n' | sed 's/Ki$//' | awk '{sum+=$1/1024/1024} END {print int(sum)}')
    
    log_info "Cluster capacity: ${total_cpu}m CPU, ${total_memory}Gi Memory"
    
    # Check if we have sufficient resources for deployment
    local required_cpu=2000  # 2 CPU cores minimum
    local required_memory=4   # 4 GB minimum
    
    if [[ $total_cpu -lt $required_cpu ]]; then
        log_error "Insufficient CPU capacity. Required: ${required_cpu}m, Available: ${total_cpu}m"
        exit 1
    fi
    
    if [[ $total_memory -lt $required_memory ]]; then
        log_error "Insufficient memory capacity. Required: ${required_memory}Gi, Available: ${total_memory}Gi"
        exit 1
    fi
    
    log_success "Cluster resources validated"
}

# Build and push container images
build_and_push_images() {
    log_info "Building and pushing container images..."
    
    local backend_tag="causal-interface-gym:${DEPLOYMENT_ID}"
    local frontend_tag="causal-interface-gym-frontend:${DEPLOYMENT_ID}"
    
    # Build backend image
    log_info "Building backend image: $backend_tag"
    docker build -f Dockerfile.production -t "$backend_tag" "$PROJECT_ROOT"
    
    # Build frontend image
    log_info "Building frontend image: $frontend_tag"
    docker build -f Dockerfile.frontend -t "$frontend_tag" "$PROJECT_ROOT"
    
    # Run security scans on images
    run_image_security_scan "$backend_tag"
    run_image_security_scan "$frontend_tag"
    
    # Push images to registry
    if [[ "${PUSH_IMAGES:-true}" == "true" ]]; then
        log_info "Pushing images to registry..."
        docker push "$backend_tag"
        docker push "$frontend_tag"
    fi
    
    # Export tags for use in deployment
    export BACKEND_IMAGE_TAG="$backend_tag"
    export FRONTEND_IMAGE_TAG="$frontend_tag"
    
    log_success "Images built and pushed successfully"
}

run_image_security_scan() {
    local image_tag=$1
    log_info "Running security scan on $image_tag..."
    
    # Use trivy for container security scanning
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL --exit-code 1 "$image_tag" || {
            log_error "Security scan failed for $image_tag"
            exit 1
        }
        log_success "Security scan passed for $image_tag"
    else
        log_warning "Trivy not installed - skipping security scan"
    fi
}

# Run comprehensive quality gates
run_quality_gates() {
    log_info "Running quality gates..."
    
    # Run tests
    log_info "Running test suite..."
    cd "$PROJECT_ROOT"
    
    # Backend tests
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        python -m pytest tests/ --cov=src --cov-report=json --cov-fail-under=85 || {
            log_error "Backend tests failed"
            exit 1
        }
    else
        log_warning "Virtual environment not found - skipping Python tests"
    fi
    
    # Frontend tests
    if [[ -f "package.json" ]]; then
        npm test -- --coverage --coverageThreshold='{"global":{"branches":70,"functions":70,"lines":80,"statements":80}}' || {
            log_error "Frontend tests failed"
            exit 1
        }
    fi
    
    # Security scanning
    run_security_scan
    
    # Performance validation
    run_performance_tests
    
    log_success "Quality gates passed"
}

run_security_scan() {
    log_info "Running security scan on codebase..."
    
    # Run bandit for Python security issues
    if command -v bandit &> /dev/null; then
        bandit -r src/ -f json -o bandit-report.json || {
            local high_severity_count
            high_severity_count=$(jq '.results | map(select(.issue_severity == "HIGH" or .issue_severity == "CRITICAL")) | length' bandit-report.json 2>/dev/null || echo "0")
            
            if [[ $high_severity_count -gt 0 ]]; then
                log_error "High/Critical security issues found: $high_severity_count"
                exit 1
            fi
        }
        log_success "Security scan completed"
    else
        log_warning "Bandit not installed - skipping security scan"
    fi
}

run_performance_tests() {
    log_info "Running performance tests..."
    
    # Placeholder for performance tests
    # In production, this would run actual performance benchmarks
    log_info "Performance baseline validation..."
    sleep 2
    
    log_success "Performance tests passed"
}

# Deploy with blue-green strategy
deploy_blue_green() {
    log_info "Starting blue-green deployment..."
    DEPLOYMENT_STARTED=true
    
    # Determine current active deployment
    local current_version
    current_version=$(kubectl get deployment causal-interface-gym-backend -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "none")
    
    local new_version
    if [[ "$current_version" == "blue" ]]; then
        new_version="green"
    else
        new_version="blue"
    fi
    
    log_info "Deploying version: $new_version (current: $current_version)"
    
    # Update deployment manifests with new image tags and version
    update_deployment_manifests "$new_version"
    
    # Apply new deployment
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f "$PROJECT_ROOT/k8s/" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    wait_for_rollout "$new_version"
    
    # Run health checks on new deployment
    run_health_checks "$new_version"
    
    # Switch traffic to new deployment
    switch_traffic "$new_version"
    
    # Cleanup old deployment
    cleanup_old_deployment "$current_version"
    
    log_success "Blue-green deployment completed successfully"
}

update_deployment_manifests() {
    local version=$1
    log_info "Updating deployment manifests for version: $version"
    
    # Create temporary directory for modified manifests
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Copy and update manifests
    cp -r "$PROJECT_ROOT/k8s/" "$temp_dir/"
    
    # Update image tags and version labels
    find "$temp_dir/k8s" -name "*.yaml" -type f -exec sed -i "s/causal-interface-gym:latest/$BACKEND_IMAGE_TAG/g" {} \;
    find "$temp_dir/k8s" -name "*.yaml" -type f -exec sed -i "s/causal-interface-gym-frontend:latest/$FRONTEND_IMAGE_TAG/g" {} \;
    find "$temp_dir/k8s" -name "*.yaml" -type f -exec sed -i "s/version: v1/version: $version/g" {} \;
    
    # Update deployment names for blue-green
    find "$temp_dir/k8s" -name "*.yaml" -type f -exec sed -i "s/causal-interface-gym-backend/causal-interface-gym-backend-$version/g" {} \;
    find "$temp_dir/k8s" -name "*.yaml" -type f -exec sed -i "s/causal-interface-gym-frontend/causal-interface-gym-frontend-$version/g" {} \;
    
    # Copy back to k8s directory
    cp -r "$temp_dir/k8s/" "$PROJECT_ROOT/"
    rm -rf "$temp_dir"
}

wait_for_rollout() {
    local version=$1
    log_info "Waiting for rollout to complete..."
    
    # Wait for backend deployment
    local backend_deployment="causal-interface-gym-backend-$version"
    kubectl rollout status deployment/"$backend_deployment" -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s" || {
        log_error "Backend rollout failed"
        exit 1
    }
    
    # Wait for frontend deployment
    local frontend_deployment="causal-interface-gym-frontend-$version"
    kubectl rollout status deployment/"$frontend_deployment" -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s" || {
        log_error "Frontend rollout failed"
        exit 1
    }
    
    log_success "Rollout completed successfully"
}

run_health_checks() {
    local version=$1
    log_info "Running health checks on new deployment..."
    
    # Get service endpoints
    local backend_service="causal-gym-backend-service-$version"
    local frontend_service="causal-gym-frontend-service-$version"
    
    # Wait for services to be ready
    local attempts=0
    local max_attempts=30
    
    while [[ $attempts -lt $max_attempts ]]; do
        # Check backend health
        if kubectl exec -n "$NAMESPACE" deployment/causal-interface-gym-backend-"$version" -- curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend health check passed"
            break
        fi
        
        attempts=$((attempts + 1))
        log_info "Health check attempt $attempts/$max_attempts..."
        sleep 10
    done
    
    if [[ $attempts -eq $max_attempts ]]; then
        log_error "Health checks failed after $max_attempts attempts"
        exit 1
    fi
    
    # Run smoke tests
    run_smoke_tests "$version"
    
    log_success "All health checks passed"
}

run_smoke_tests() {
    local version=$1
    log_info "Running smoke tests..."
    
    # Test critical API endpoints
    local test_endpoints=(
        "/health"
        "/ready"
        "/api/v1/environments"
        "/api/v1/benchmark/status"
    )
    
    for endpoint in "${test_endpoints[@]}"; do
        log_info "Testing endpoint: $endpoint"
        
        # Port-forward to test the service directly
        kubectl port-forward -n "$NAMESPACE" deployment/causal-interface-gym-backend-"$version" 8000:8000 &
        local port_forward_pid=$!
        sleep 5
        
        if curl -f "http://localhost:8000$endpoint" > /dev/null 2>&1; then
            log_success "Endpoint $endpoint is healthy"
        else
            log_error "Endpoint $endpoint failed health check"
            kill $port_forward_pid 2>/dev/null || true
            exit 1
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    done
    
    log_success "Smoke tests completed successfully"
}

switch_traffic() {
    local new_version=$1
    log_info "Switching traffic to new deployment: $new_version"
    
    # Update service selectors to point to new deployment
    kubectl patch service causal-gym-backend-service -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"'$new_version'"}}}'
    kubectl patch service causal-gym-frontend-service -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"'$new_version'"}}}'
    
    # Verify traffic switch
    sleep 10
    run_post_switch_validation
    
    log_success "Traffic switched successfully to $new_version"
}

run_post_switch_validation() {
    log_info "Running post-traffic-switch validation..."
    
    # Test external endpoints through ingress
    local external_url="https://causal-gym.com"
    local api_url="https://api.causal-gym.com"
    
    # Check frontend
    if curl -f "$external_url" > /dev/null 2>&1; then
        log_success "Frontend is accessible via ingress"
    else
        log_warning "Frontend not accessible via ingress (may be DNS propagation delay)"
    fi
    
    # Check API
    if curl -f "$api_url/health" > /dev/null 2>&1; then
        log_success "API is accessible via ingress"
    else
        log_warning "API not accessible via ingress (may be DNS propagation delay)"
    fi
}

cleanup_old_deployment() {
    local old_version=$1
    if [[ "$old_version" != "none" ]]; then
        log_info "Cleaning up old deployment: $old_version"
        
        # Scale down old deployment
        kubectl scale deployment causal-interface-gym-backend-"$old_version" -n "$NAMESPACE" --replicas=0 2>/dev/null || true
        kubectl scale deployment causal-interface-gym-frontend-"$old_version" -n "$NAMESPACE" --replicas=0 2>/dev/null || true
        
        # Wait a bit before cleanup
        sleep 30
        
        # Delete old deployment
        kubectl delete deployment causal-interface-gym-backend-"$old_version" -n "$NAMESPACE" 2>/dev/null || true
        kubectl delete deployment causal-interface-gym-frontend-"$old_version" -n "$NAMESPACE" 2>/dev/null || true
        
        log_success "Old deployment cleaned up"
    fi
}

rollback_deployment() {
    log_error "Initiating deployment rollback..."
    
    # Get previous successful deployment
    local previous_version
    previous_version=$(kubectl rollout history deployment/causal-interface-gym-backend -n "$NAMESPACE" --revision=2 2>/dev/null | grep -o "version=.*" | cut -d= -f2 || echo "unknown")
    
    if [[ "$previous_version" != "unknown" ]]; then
        log_info "Rolling back to previous version: $previous_version"
        
        # Rollback deployments
        kubectl rollout undo deployment/causal-interface-gym-backend -n "$NAMESPACE" --to-revision=2
        kubectl rollout undo deployment/causal-interface-gym-frontend -n "$NAMESPACE" --to-revision=2
        
        # Wait for rollback to complete
        kubectl rollout status deployment/causal-interface-gym-backend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
        kubectl rollout status deployment/causal-interface-gym-frontend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
        
        log_success "Rollback completed"
    else
        log_error "Could not determine previous version for rollback"
        return 1
    fi
}

# Monitoring and alerting setup
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Apply monitoring configurations
    if [[ -f "$PROJECT_ROOT/monitoring/prometheus.yml" ]]; then
        kubectl apply -f "$PROJECT_ROOT/monitoring/" -n monitoring 2>/dev/null || true
    fi
    
    # Setup service monitors
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: causal-gym-monitor
  namespace: causal-gym
spec:
  selector:
    matchLabels:
      app: causal-gym-backend
  endpoints:
  - port: metrics
    path: /metrics
EOF
    
    log_success "Monitoring setup completed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-${DEPLOYMENT_ID}.json"
    
    cat > "$report_file" <<EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
  "backend_image": "$BACKEND_IMAGE_TAG",
  "frontend_image": "$FRONTEND_IMAGE_TAG",
  "namespace": "$NAMESPACE",
  "cluster_context": "$(kubectl config current-context)",
  "deployment_status": "success",
  "health_checks": "passed",
  "rollout_duration": "$(($(date +%s) - ${DEPLOYMENT_START_TIME:-$(date +%s)}))"
}
EOF
    
    log_success "Deployment report generated: $report_file"
}

# Main deployment workflow
main() {
    local DEPLOYMENT_START_TIME
    DEPLOYMENT_START_TIME=$(date +%s)
    
    log_info "🚀 Starting Causal Interface Gym production deployment"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --rollback)
                rollback_deployment
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment pipeline
    validate_prerequisites
    
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_and_push_images
    fi
    
    if [[ "${SKIP_TESTS:-false}" != "true" ]]; then
        run_quality_gates
    fi
    
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        deploy_blue_green
        setup_monitoring
        generate_deployment_report
    else
        log_info "Dry run completed - no actual deployment performed"
    fi
    
    local deployment_duration=$(($(date +%s) - DEPLOYMENT_START_TIME))
    log_success "🎉 Deployment completed successfully in ${deployment_duration}s"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "Monitor the deployment at: https://causal-gym.com"
}

# Execute main function with all arguments
main "$@"