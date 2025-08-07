# Deployment Guide - Causal Interface Gym

This guide covers production deployment of the Causal Interface Gym using Docker Compose and Kubernetes.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for development)
- kubectl (for Kubernetes deployment)
- SSL certificates (for production)

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_PASSWORD=your_redis_password

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# LLM APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Application
CAUSAL_GYM_ENVIRONMENT=production
CAUSAL_GYM_DATABASE_URL=postgresql://causal_gym:${POSTGRES_PASSWORD}@postgres:5432/causal_gym_prod
CAUSAL_GYM_REDIS_URL=redis://redis:6379
```

### Docker Compose Deployment

```bash
# Deploy to production
./scripts/deploy.sh production latest compose

# Deploy to staging
./scripts/deploy.sh staging latest compose

# Health check
./scripts/deploy.sh health
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy.sh production latest k8s

# Check status
kubectl get pods -n causal-gym

# View logs
kubectl logs deployment/causal-gym-api -n causal-gym
```

## 📊 Architecture

### Production Stack

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│     NGINX       │────│  Load        │────│   Causal Gym    │
│   (SSL/Proxy)   │    │  Balancer    │    │   API (x3)      │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                      │                    │
         │              ┌──────────────┐             │
         │              │  PostgreSQL  │             │
         │              │  (Primary)   │             │
         │              └──────────────┘             │
         │                      │                    │
         └──────────────────────┼────────────────────┘
                                │
                    ┌─────────────────┐
                    │      Redis      │
                    │    (Cache)      │
                    └─────────────────┘
```

### Monitoring Stack

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Nginx**: Request metrics and logs
- **Application**: Custom metrics and health checks

## 🔧 Configuration

### Security Configuration

```yaml
security:
  max_graph_nodes: 1000
  max_graph_edges: 5000
  rate_limit_calls: 100
  rate_limit_window: 60
  enable_input_sanitization: true
  enable_audit_logging: true
```

### Performance Configuration

```yaml
performance:
  cache_enabled: true
  cache_ttl: 3600
  max_concurrent_experiments: 20
  enable_performance_monitoring: true
  metric_retention_hours: 24
```

### Database Configuration

```yaml
database:
  url: "postgresql://user:pass@host:5432/db"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
```

## 📈 Scaling

### Horizontal Scaling

The application supports horizontal scaling through:

- **Docker Compose**: Increase replica count
- **Kubernetes**: HorizontalPodAutoscaler (HPA)
- **Load Balancing**: Nginx upstream configuration

### Auto-scaling Configuration

```yaml
# Kubernetes HPA
minReplicas: 3
maxReplicas: 20
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Resource Limits

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🔍 Monitoring & Observability

### Health Checks

- **Liveness Probe**: `/health` endpoint
- **Readiness Probe**: `/health` endpoint
- **Startup Probe**: Application initialization

### Metrics

Available at `/metrics` endpoint:

- Request duration and count
- Cache hit/miss rates
- Database connection pool metrics
- Causal computation performance
- System resource usage

### Logging

Structured logging with multiple levels:

- **Application logs**: `/app/logs/causal_gym.log`
- **Error logs**: `/app/logs/causal_gym_errors.log`
- **Security logs**: `/app/logs/causal_gym_security.log`
- **Performance logs**: `/app/logs/causal_gym_performance.log`

### Grafana Dashboards

Access monitoring dashboards:

- **URL**: `http://localhost:3000` (Docker) or configured domain
- **Username**: `admin`
- **Password**: Set via `GRAFANA_PASSWORD`

Key dashboards:
- Application Performance
- Database Metrics
- Cache Performance
- Request Analytics
- System Resources

## 🗄️ Database Management

### Migrations

```bash
# Run migrations
docker-compose exec causal-gym-api python -m alembic upgrade head

# Create migration
docker-compose exec causal-gym-api python -m alembic revision --autogenerate -m "description"
```

### Backup & Restore

```bash
# Backup
docker-compose exec postgres pg_dump -U causal_gym causal_gym_prod > backup.sql

# Restore
docker-compose exec -T postgres psql -U causal_gym causal_gym_prod < backup.sql
```

### Connection Pooling

PostgreSQL connection pooling is configured for optimal performance:

- **Pool Size**: 20 connections
- **Max Overflow**: 30 connections
- **Pool Timeout**: 30 seconds

## 🔐 Security

### SSL/TLS

SSL certificates should be placed in:
- `nginx/ssl/cert.pem`
- `nginx/ssl/key.pem`

For Let's Encrypt certificates:
```bash
# Generate certificates
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

### Network Security

- All services run in isolated Docker network
- Database and Redis not exposed externally
- Rate limiting on API endpoints
- Input validation and sanitization

### Secrets Management

**Docker Compose:**
Use environment variables and `.env` file

**Kubernetes:**
Use Kubernetes secrets:
```bash
kubectl create secret generic causal-gym-secrets \
  --from-literal=database-url="..." \
  --from-literal=redis-url="..."
```

## 🚨 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec causal-gym-api python -c "from causal_interface_gym.database import DatabaseManager; print('OK')"
```

#### Cache Issues
```bash
# Check Redis
docker-compose exec redis redis-cli ping

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Review slow queries
docker-compose logs causal-gym-api | grep "slow"
```

### Log Analysis

```bash
# Application logs
docker-compose logs -f causal-gym-api

# Nginx access logs
docker-compose logs nginx | grep "GET\|POST"

# Error logs only
docker-compose logs causal-gym-api | grep "ERROR"
```

## 🔄 Maintenance

### Rolling Updates

```bash
# Update to new version
./scripts/deploy.sh production v1.2.0 compose

# Rollback if needed
./scripts/deploy.sh rollback
```

### Database Maintenance

```bash
# Vacuum database
docker-compose exec postgres vacuumdb -U causal_gym causal_gym_prod

# Check database size
docker-compose exec postgres psql -U causal_gym -c "SELECT pg_size_pretty(pg_database_size('causal_gym_prod'));"
```

### Log Rotation

Logs are automatically rotated using Docker's logging driver:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
```

## 📞 Support

### Monitoring Alerts

Set up alerts for:
- High error rate (>5%)
- High response time (>2s)
- Low cache hit rate (<80%)
- High memory usage (>90%)
- Database connection issues

### Performance Tuning

1. **Database**: Optimize queries and indexes
2. **Cache**: Increase cache size if needed
3. **Application**: Adjust worker count
4. **Load Balancer**: Tune upstream configuration

### Capacity Planning

Monitor these metrics for scaling decisions:
- CPU utilization (target: <70%)
- Memory usage (target: <80%)
- Request latency (target: <500ms)
- Cache hit rate (target: >80%)

## 🎯 Production Checklist

Before going live:

- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database migrations run
- [ ] Health checks passing
- [ ] Monitoring dashboards configured
- [ ] Backup strategy implemented
- [ ] Security scan passed
- [ ] Load testing completed
- [ ] Rollback procedure tested
- [ ] Documentation updated

## 📚 Additional Resources

- [API Documentation](./docs/API.md)
- [Security Guide](./docs/SECURITY.md)
- [Performance Tuning](./docs/PERFORMANCE.md)
- [Development Setup](./CONTRIBUTING.md)

For support, please open an issue or contact the development team.