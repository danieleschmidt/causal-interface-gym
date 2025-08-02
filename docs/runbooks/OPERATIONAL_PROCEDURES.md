# Operational Procedures

*Last Updated: 2025-08-02*

## Overview

This document outlines standard operational procedures for maintaining and operating the Causal Interface Gym system in production environments.

## Daily Operations

### Morning Health Check (10 minutes)

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Daily Health Check - $(date) ==="

# 1. Service Status
echo "Checking service status..."
docker ps --filter "name=causal" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Health Endpoints
echo "Checking health endpoints..."
curl -s http://localhost:8501/health | jq '.status' || echo "❌ Health check failed"
curl -s http://localhost:8090/metrics | grep -c "causal_" || echo "❌ Metrics unavailable"

# 3. Resource Usage
echo "Checking resource usage..."
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# 4. Database Connectivity
echo "Checking database..."
docker exec postgres pg_isready -U causal_user || echo "❌ Database unavailable"
docker exec redis redis-cli ping || echo "❌ Redis unavailable"

# 5. Recent Errors
echo "Checking for recent errors..."
docker logs causal-interface-gym --since="24h" | grep -i error | wc -l

# 6. Backup Status
echo "Checking backup status..."
ls -la /backups/ | tail -5

echo "=== Health Check Complete ==="
```

### Log Review (15 minutes)

```bash
#!/bin/bash
# daily_log_review.sh

echo "=== Daily Log Review - $(date) ==="

# Check for errors in the last 24 hours
echo "Recent errors:"
docker logs causal-interface-gym --since="24h" | grep -E "(ERROR|CRITICAL|FATAL)" | tail -10

# Check for performance issues
echo "Slow requests (>5s):"
docker logs causal-interface-gym --since="24h" | grep "slow_request" | tail -5

# Check LLM API failures
echo "LLM API failures:"
docker logs causal-interface-gym --since="24h" | grep -E "(openai.*error|anthropic.*error)" | tail -5

# Check unusual patterns
echo "Connection timeouts:"
docker logs causal-interface-gym --since="24h" | grep -c "timeout"

echo "Database connection issues:"
docker logs causal-interface-gym --since="24h" | grep -c "connection.*failed"
```

## Weekly Operations

### Performance Review (30 minutes)

```bash
#!/bin/bash
# weekly_performance_review.sh

echo "=== Weekly Performance Review - $(date) ==="

# 1. Response Time Trends
echo "Average response times (last 7 days):"
docker exec prometheus promtool query instant \
  'avg_over_time(causal_inference_duration_seconds[7d])'

# 2. Error Rate Analysis
echo "Error rates by endpoint:"
docker exec prometheus promtool query instant \
  'rate(causal_inference_requests_total{status="error"}[7d])'

# 3. Resource Utilization
echo "Peak resource utilization:"
docker exec prometheus promtool query instant \
  'max_over_time(container_cpu_usage_seconds_total[7d])'

# 4. Database Performance
echo "Slowest database queries:"
docker exec postgres psql -U causal_user -d causal_experiments -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE calls > 10 
ORDER BY mean_time DESC 
LIMIT 10;"

# 5. Cache Hit Rates
echo "Redis cache performance:"
docker exec redis redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses)"

echo "=== Performance Review Complete ==="
```

### Security Audit (45 minutes)

```bash
#!/bin/bash
# weekly_security_audit.sh

echo "=== Weekly Security Audit - $(date) ==="

# 1. Dependency Vulnerabilities
echo "Checking for vulnerable dependencies..."
docker exec causal-interface-gym pip-audit --format=json > /tmp/pip-audit.json
python -c "
import json
with open('/tmp/pip-audit.json') as f:
    data = json.load(f)
    if data['vulnerabilities']:
        print(f'❌ Found {len(data[\"vulnerabilities\"])} vulnerabilities')
        for vuln in data['vulnerabilities'][:5]:  # Show top 5
            print(f'  - {vuln[\"package\"]} {vuln[\"installed_version\"]}: {vuln[\"id\"]}')
    else:
        print('✅ No vulnerabilities found')
"

# 2. Container Security
echo "Scanning container for security issues..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image causal-interface-gym:latest

# 3. Configuration Security
echo "Checking configuration security..."
# Check for exposed secrets
docker exec causal-interface-gym env | grep -E "(KEY|TOKEN|PASSWORD)" | grep -v "XXX" || echo "✅ No exposed secrets"

# Check file permissions
docker exec causal-interface-gym find /app -name "*.py" -perm 777 | wc -l

# 4. Access Log Analysis
echo "Analyzing access patterns..."
docker logs nginx --since="7d" | grep -E "(401|403|404)" | wc -l

echo "=== Security Audit Complete ==="
```

## Monthly Operations

### Capacity Planning (60 minutes)

```bash
#!/bin/bash
# monthly_capacity_planning.sh

echo "=== Monthly Capacity Planning - $(date) ==="

# 1. Usage Growth Analysis
echo "Analyzing usage growth..."
docker exec prometheus promtool query instant \
  'increase(causal_inference_requests_total[30d])'

# 2. Resource Trend Analysis
echo "Resource usage trends:"
docker exec prometheus promtool query instant \
  'avg_over_time(container_memory_usage_bytes[30d])'

# 3. Database Growth
echo "Database size trends:"
docker exec postgres psql -U causal_user -d causal_experiments -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY size_bytes DESC;"

# 4. Forecast Resource Needs
echo "Forecasting resource requirements..."
python scripts/capacity_forecast.py --period=30d --forecast=90d

echo "=== Capacity Planning Complete ==="
```

### Backup and Recovery Testing (90 minutes)

```bash
#!/bin/bash
# monthly_backup_recovery_test.sh

echo "=== Monthly Backup Recovery Test - $(date) ==="

BACKUP_DATE=$(date +%Y%m%d)
TEST_DB="causal_experiments_test_$BACKUP_DATE"

# 1. Create fresh backup
echo "Creating fresh backup..."
docker exec postgres pg_dump -U causal_user causal_experiments > "/backups/test_backup_$BACKUP_DATE.sql"

# 2. Test backup restoration
echo "Testing backup restoration..."
docker exec postgres createdb -U causal_user "$TEST_DB"
docker exec postgres psql -U causal_user -d "$TEST_DB" < "/backups/test_backup_$BACKUP_DATE.sql"

# 3. Validate restored data
echo "Validating restored data..."
ORIGINAL_COUNT=$(docker exec postgres psql -U causal_user -d causal_experiments -t -c "SELECT count(*) FROM experiments;")
RESTORED_COUNT=$(docker exec postgres psql -U causal_user -d "$TEST_DB" -t -c "SELECT count(*) FROM experiments;")

if [ "$ORIGINAL_COUNT" -eq "$RESTORED_COUNT" ]; then
    echo "✅ Backup restoration successful"
else
    echo "❌ Backup restoration failed: Original=$ORIGINAL_COUNT, Restored=$RESTORED_COUNT"
fi

# 4. Cleanup test database
docker exec postgres dropdb -U causal_user "$TEST_DB"

# 5. Test application startup with restored data
echo "Testing application startup..."
docker-compose -f docker-compose.test.yml up -d
sleep 30
curl -f http://localhost:8501/health || echo "❌ Application startup failed"
docker-compose -f docker-compose.test.yml down

echo "=== Backup Recovery Test Complete ==="
```

## Maintenance Procedures

### Routine Updates

#### Dependency Updates (Monthly)
```bash
#!/bin/bash
# update_dependencies.sh

echo "Updating Python dependencies..."
# Update requirements files
pip-tools compile --upgrade pyproject.toml
pip-tools compile --upgrade requirements-dev.in

# Update Docker base images
docker pull python:3.11-slim
docker pull postgres:15-alpine
docker pull redis:7-alpine

# Update pre-commit hooks
pre-commit autoupdate

# Test updates
make test
make lint

echo "Dependencies updated. Review changes and commit."
```

#### Security Patches (As needed)
```bash
#!/bin/bash
# apply_security_patches.sh

echo "Applying security patches..."

# Update system packages in containers
docker-compose build --no-cache --pull

# Update Python security patches
pip install --upgrade $(pip list --outdated --format=json | python -c "
import json, sys
packages = json.load(sys.stdin)
security_packages = ['requests', 'urllib3', 'certifi', 'cryptography']
for pkg in packages:
    if pkg['name'].lower() in security_packages:
        print(pkg['name'])
")

echo "Security patches applied. Test thoroughly before deployment."
```

### Configuration Management

#### Environment Variable Updates
```bash
#!/bin/bash
# update_environment.sh

# Backup current configuration
cp .env .env.backup.$(date +%Y%m%d)

# Update environment variables
echo "Current configuration:"
grep -E "API_KEY|DATABASE_URL|REDIS_URL" .env

echo "Update .env file with new values, then restart services:"
echo "docker-compose down && docker-compose up -d"
```

#### Secrets Rotation
```bash
#!/bin/bash
# rotate_secrets.sh

echo "Rotating secrets..."

# Generate new JWT secret
NEW_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "New JWT secret generated (update in .env): $NEW_JWT_SECRET"

# Generate new database password
NEW_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(16))")
echo "New database password generated: $NEW_DB_PASSWORD"

echo "Manual steps required:"
echo "1. Update .env with new secrets"
echo "2. Update database password: ALTER USER causal_user PASSWORD '$NEW_DB_PASSWORD';"
echo "3. Restart all services"
echo "4. Verify functionality"
```

## Scaling Operations

### Horizontal Scaling
```bash
#!/bin/bash
# scale_services.sh

DESIRED_REPLICAS=${1:-3}
echo "Scaling to $DESIRED_REPLICAS replicas..."

# Scale application containers
docker-compose up -d --scale app=$DESIRED_REPLICAS

# Update load balancer configuration
echo "Update nginx.conf with new backend servers:"
for i in $(seq 1 $DESIRED_REPLICAS); do
    echo "    server app$i:8501;"
done

# Reload nginx
docker-compose exec nginx nginx -s reload

echo "Scaling complete. Monitor performance and adjust as needed."
```

### Database Scaling
```bash
#!/bin/bash
# scale_database.sh

echo "Database scaling options:"
echo "1. Vertical scaling: Increase container resources"
echo "2. Connection pooling: Implement PgBouncer"
echo "3. Read replicas: Set up streaming replication"

# Example: Add connection pooling
cat << EOF > pgbouncer.ini
[databases]
causal_experiments = host=postgres port=5432 dbname=causal_experiments

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = trust
max_client_conn = 200
default_pool_size = 25
server_lifetime = 3600
server_idle_timeout = 600
EOF

echo "Add pgbouncer service to docker-compose.yml and update DATABASE_URL"
```

## Disaster Recovery

### Full System Recovery
```bash
#!/bin/bash
# disaster_recovery.sh

echo "=== Disaster Recovery Procedure ==="

# 1. Assess damage
echo "Step 1: Assessing system state..."
docker ps -a
docker images
df -h

# 2. Restore from backups
echo "Step 2: Restoring from backups..."
# Restore latest database backup
LATEST_BACKUP=$(ls -t /backups/*.sql | head -1)
echo "Restoring from: $LATEST_BACKUP"

# Rebuild containers
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d postgres redis

# Wait for database
sleep 30

# Restore data
docker exec postgres psql -U causal_user -d causal_experiments < "$LATEST_BACKUP"

# 3. Start application
echo "Step 3: Starting application services..."
docker-compose up -d

# 4. Verify functionality
echo "Step 4: Verifying system functionality..."
sleep 60
curl -f http://localhost:8501/health || echo "❌ Health check failed"

# 5. Notify stakeholders
echo "Step 5: System recovery complete. Notify stakeholders."
```

## Monitoring Dashboards

### Grafana Dashboard Setup
```bash
#!/bin/bash
# setup_dashboards.sh

echo "Setting up Grafana dashboards..."

# Import causal reasoning dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/dashboards/causal-reasoning.json

# Import system metrics dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/dashboards/system-metrics.json

echo "Dashboards imported. Access at http://localhost:3000"
```

### Custom Alerts Setup
```bash
#!/bin/bash
# setup_alerts.sh

echo "Configuring custom alerts..."

# Add alert rules to Prometheus
docker exec prometheus promtool check rules /etc/prometheus/alert.rules.yml

# Test alertmanager configuration
docker exec alertmanager amtool config check

# Send test alert
curl -XPOST http://localhost:9093/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert for operational verification"
    }
  }]'

echo "Alerts configured. Check Slack/email for test notification."
```

## Documentation Updates

### Runbook Maintenance
- Review and update procedures monthly
- Test all scripts in staging environment
- Document lessons learned from incidents
- Keep contact information current

### Knowledge Transfer
- Maintain on-call rotation documentation
- Create video recordings of complex procedures
- Regular training sessions for new team members
- Cross-train on different operational areas

---

*These operational procedures should be reviewed and updated regularly based on operational experience and system changes.*