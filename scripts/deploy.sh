#!/bin/bash

# Production deployment script for Causal Interface Gym
# Usage: ./scripts/deploy.sh [environment] [version]

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
PROJECT_NAME="causal-interface-gym"

echo "🚀 Deploying Causal Interface Gym"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker first."
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is not installed. Please install it first."
    fi
    
    # Check environment variables
    if [ "$ENVIRONMENT" = "production" ]; then
        required_vars=(
            "POSTGRES_PASSWORD"
            "REDIS_PASSWORD" 
            "GRAFANA_PASSWORD"
            "OPENAI_API_KEY"
        )
        
        for var in "${required_vars[@]}"; do
            if [ -z "${!var}" ]; then
                error "Required environment variable $var is not set"
            fi
        done
    fi
    
    log "✅ Prerequisites check passed"
}

# Build application image
build_image() {
    log "Building application image..."
    
    # Build production image
    docker build -f Dockerfile.production -t ${PROJECT_NAME}:${VERSION} .
    
    if [ "$VERSION" != "latest" ]; then
        docker tag ${PROJECT_NAME}:${VERSION} ${PROJECT_NAME}:latest
    fi
    
    log "✅ Image built successfully"
}

# Deploy with Docker Compose
deploy_compose() {
    log "Deploying with Docker Compose..."
    
    # Export environment variables
    export VERSION
    
    # Deploy based on environment
    case $ENVIRONMENT in
        "production")
            compose_file="docker-compose.production.yml"
            ;;
        "staging")
            compose_file="docker-compose.yml"
            ;;
        *)
            compose_file="docker-compose.dev.yml"
            ;;
    esac
    
    # Pull latest images (except our app image)
    docker-compose -f $compose_file pull postgres redis nginx prometheus grafana
    
    # Deploy services
    docker-compose -f $compose_file up -d
    
    log "✅ Services deployed"
}

# Deploy to Kubernetes
deploy_k8s() {
    log "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not configured"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace causal-gym > /dev/null 2>&1; then
        log "Creating namespace..."
        kubectl create namespace causal-gym
    fi
    
    # Apply secrets (if not exists)
    if ! kubectl get secret causal-gym-secrets -n causal-gym > /dev/null 2>&1; then
        log "Creating secrets..."
        kubectl create secret generic causal-gym-secrets \
            --from-literal=database-url="$CAUSAL_GYM_DATABASE_URL" \
            --from-literal=redis-url="$CAUSAL_GYM_REDIS_URL" \
            -n causal-gym
    fi
    
    if ! kubectl get secret llm-secrets -n causal-gym > /dev/null 2>&1; then
        kubectl create secret generic llm-secrets \
            --from-literal=openai-api-key="$OPENAI_API_KEY" \
            --from-literal=anthropic-api-key="${ANTHROPIC_API_KEY:-}" \
            -n causal-gym
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/causal-gym-api -n causal-gym --timeout=300s
    
    log "✅ Kubernetes deployment completed"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    local health_url="http://localhost:8000/health"
    
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        # Port forward for health check
        kubectl port-forward service/causal-gym-api-service 8000:8000 -n causal-gym &
        local port_forward_pid=$!
        sleep 5
    fi
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s $health_url > /dev/null; then
            log "✅ Health check passed"
            
            if [ "$DEPLOYMENT_TYPE" = "k8s" ] && [ ! -z "$port_forward_pid" ]; then
                kill $port_forward_pid
            fi
            
            return 0
        fi
        
        warn "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    error "Health check failed after $max_attempts attempts"
}

# Database migration
run_migrations() {
    log "Running database migrations..."
    
    if [ "$DEPLOYMENT_TYPE" = "compose" ]; then
        docker-compose exec causal-gym-api python -m alembic upgrade head
    elif [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        kubectl exec deployment/causal-gym-api -n causal-gym -- python -m alembic upgrade head
    fi
    
    log "✅ Database migrations completed"
}

# Backup before deployment
backup_data() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log "Creating backup..."
        
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_dir="backups/$timestamp"
        mkdir -p $backup_dir
        
        # Database backup
        if [ "$DEPLOYMENT_TYPE" = "compose" ]; then
            docker-compose exec postgres pg_dump -U causal_gym causal_gym_prod > $backup_dir/database.sql
        elif [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
            kubectl exec deployment/postgres -n causal-gym -- pg_dump -U causal_gym causal_gym_prod > $backup_dir/database.sql
        fi
        
        log "✅ Backup created at $backup_dir"
    fi
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    if [ "$DEPLOYMENT_TYPE" = "compose" ]; then
        # Get previous version
        previous_version=$(docker images ${PROJECT_NAME} --format "table {{.Tag}}" | sed -n '2p' | tr -d ' ')
        if [ ! -z "$previous_version" ]; then
            export VERSION=$previous_version
            docker-compose -f docker-compose.production.yml up -d causal-gym-api
            log "✅ Rolled back to version $previous_version"
        fi
    elif [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        kubectl rollout undo deployment/causal-gym-api -n causal-gym
        log "✅ Kubernetes rollback initiated"
    fi
}

# Monitoring setup
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create Grafana dashboards directory if not exists
    mkdir -p monitoring/dashboards
    
    # Generate Prometheus config if needed
    if [ ! -f monitoring/prometheus.yml ]; then
        cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules.yml"

scrape_configs:
  - job_name: 'causal-gym'
    static_configs:
      - targets: ['causal-gym-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:8080']
    metrics_path: '/nginx_status'
EOF
    fi
    
    log "✅ Monitoring configured"
}

# Main deployment flow
main() {
    # Parse deployment type from arguments or environment
    DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-compose}
    
    case "$3" in
        "k8s"|"kubernetes")
            DEPLOYMENT_TYPE="k8s"
            ;;
        "compose"|"docker-compose")
            DEPLOYMENT_TYPE="compose"
            ;;
    esac
    
    log "Deployment type: $DEPLOYMENT_TYPE"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup monitoring
    setup_monitoring
    
    # Backup data (production only)
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_data
    fi
    
    # Build image
    build_image
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        "k8s")
            deploy_k8s
            ;;
        "compose")
            deploy_compose
            ;;
        *)
            error "Unknown deployment type: $DEPLOYMENT_TYPE"
            ;;
    esac
    
    # Run migrations
    run_migrations
    
    # Health check
    health_check
    
    log "🎉 Deployment completed successfully!"
    log "Application is running at:"
    
    if [ "$DEPLOYMENT_TYPE" = "compose" ]; then
        log "  - API: https://localhost"
        log "  - Monitoring: http://localhost:3000 (Grafana)"
        log "  - Metrics: http://localhost:9090 (Prometheus)"
    else
        log "  - API: https://api.causal-gym.com"
        log "  - Use 'kubectl port-forward' for local access to monitoring"
    fi
}

# Trap for cleanup on error
trap 'error "Deployment failed! Use ./scripts/deploy.sh rollback to revert changes"' ERR

# Handle special commands
case "$1" in
    "rollback")
        rollback
        exit 0
        ;;
    "health")
        health_check
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac