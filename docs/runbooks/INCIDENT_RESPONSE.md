# Incident Response Runbook

*Last Updated: 2025-08-02*

## Overview

This runbook provides procedures for responding to incidents in the Causal Interface Gym system. It covers common issues, escalation procedures, and post-incident analysis.

## Incident Classification

### Severity Levels

**SEV1 - Critical**
- System completely unavailable
- Data loss or corruption
- Security breach
- Response Time: 15 minutes
- Escalation: Immediate

**SEV2 - High**
- Major functionality degraded
- Performance severely impacted
- API errors affecting multiple users
- Response Time: 1 hour
- Escalation: If not resolved in 2 hours

**SEV3 - Medium**
- Minor functionality issues
- Performance degradation
- Single user impact
- Response Time: 4 hours
- Escalation: If not resolved in 8 hours

**SEV4 - Low**
- Cosmetic issues
- Documentation errors
- Enhancement requests
- Response Time: 1 business day
- Escalation: None

## Contact Information

### On-Call Contacts
- **Primary**: +1-XXX-XXX-XXXX (oncall@causal-gym.org)
- **Secondary**: +1-XXX-XXX-XXXX (backup@causal-gym.org)
- **Manager**: +1-XXX-XXX-XXXX (manager@causal-gym.org)

### External Dependencies
- **AWS Support**: 1-206-266-4064
- **OpenAI Support**: support@openai.com
- **Anthropic Support**: support@anthropic.com

## Common Incidents

### 1. Service Unavailable (HTTP 503)

**Symptoms:**
- Health check endpoint returning 503
- Users unable to access application
- Load balancer showing backend servers as down

**Investigation Steps:**
```bash
# Check container status
docker ps | grep causal-interface-gym

# Check service logs
docker logs causal-interface-gym --tail=100

# Check system resources
docker stats

# Check network connectivity
curl -I http://localhost:8501/health
```

**Resolution Steps:**
1. **Restart Service**:
   ```bash
   docker-compose restart app
   ```

2. **Scale Up** (if resource constrained):
   ```bash
   docker-compose up -d --scale app=3
   ```

3. **Check Dependencies**:
   ```bash
   # Database connectivity
   docker exec app pg_isready -h postgres -p 5432
   
   # Redis connectivity
   docker exec app redis-cli -h redis ping
   ```

**Escalation Criteria:**
- Service doesn't recover after restart
- Database or external dependencies unavailable
- Resource constraints cannot be resolved

### 2. High Response Times

**Symptoms:**
- API response times > 5 seconds
- User complaints about slow performance
- High CPU/memory usage

**Investigation Steps:**
```bash
# Check current performance metrics
curl http://localhost:8090/metrics | grep response_time

# Check resource usage
docker exec app top
docker exec app ps aux --sort=-%cpu | head -10

# Check database performance
docker exec postgres psql -U causal_user -d causal_experiments -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

**Resolution Steps:**
1. **Scale Horizontally**:
   ```bash
   docker-compose up -d --scale app=5
   ```

2. **Check for Memory Leaks**:
   ```bash
   # Memory profiling
   docker exec app python -m memory_profiler --pdb-mmem=100 /app/examples/performance_test.py
   ```

3. **Database Optimization**:
   ```sql
   -- Check for slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   WHERE mean_time > 5000 
   ORDER BY mean_time DESC;
   
   -- Check for missing indexes
   SELECT schemaname, tablename, attname, n_distinct, correlation 
   FROM pg_stats 
   WHERE n_distinct > 100 AND correlation < 0.1;
   ```

### 3. LLM API Failures

**Symptoms:**
- Causal reasoning experiments failing
- High error rates from LLM providers
- Timeout errors in logs

**Investigation Steps:**
```bash
# Check API key configuration
docker exec app env | grep -E "(OPENAI|ANTHROPIC|GOOGLE)_API_KEY"

# Check recent LLM API calls
docker logs app | grep -E "(openai|anthropic|google)" | tail -20

# Test API connectivity
docker exec app python -c "
import openai
client = openai.OpenAI()
try:
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=5
    )
    print('OpenAI API: OK')
except Exception as e:
    print(f'OpenAI API Error: {e}')
"
```

**Resolution Steps:**
1. **Implement Fallback**:
   ```python
   # Switch to backup LLM provider
   export PRIMARY_LLM_PROVIDER=anthropic  # if OpenAI fails
   docker-compose restart app
   ```

2. **Adjust Rate Limits**:
   ```bash
   # Reduce concurrent requests
   export LLM_MAX_CONCURRENT_REQUESTS=5
   export LLM_REQUEST_DELAY=1.0
   docker-compose restart app
   ```

3. **Cache Previous Results**:
   ```bash
   # Enable aggressive caching
   export ENABLE_LLM_RESPONSE_CACHE=true
   export CACHE_TTL=3600
   docker-compose restart app
   ```

### 4. Data Inconsistencies

**Symptoms:**
- Causal inference results don't match expected values
- Test failures in causal reasoning validation
- User reports of incorrect behavior

**Investigation Steps:**
```bash
# Run validation suite
docker exec app python -m pytest tests/validation/ -v

# Check data integrity
docker exec postgres psql -U causal_user -d causal_experiments -c "
SELECT table_name, 
       pg_size_pretty(pg_total_relation_size(table_name)) as size,
       (SELECT count(*) FROM table_name) as row_count
FROM information_schema.tables 
WHERE table_schema = 'public';"

# Validate causal models
docker exec app python -c "
from causal_interface_gym.validation import validate_all_scenarios
results = validate_all_scenarios()
for scenario, result in results.items():
    if not result.is_valid:
        print(f'INVALID: {scenario} - {result.errors}')
"
```

**Resolution Steps:**
1. **Rollback Recent Changes**:
   ```bash
   # Rollback to last known good version
   git log --oneline -10  # Find last good commit
   docker build -t causal-interface-gym:rollback .
   docker-compose down
   docker-compose up -d
   ```

2. **Restore from Backup**:
   ```bash
   # Database restoration
   docker exec postgres pg_restore -U causal_user -d causal_experiments /backups/latest.dump
   ```

3. **Recalibrate Models**:
   ```bash
   # Recalibrate causal inference parameters
   docker exec app python scripts/recalibrate_models.py --force
   ```

## Escalation Procedures

### Internal Escalation
1. **Level 1**: On-call engineer
2. **Level 2**: Senior engineer or team lead
3. **Level 3**: Engineering manager
4. **Level 4**: CTO or equivalent

### External Escalation
1. **AWS Issues**: Open AWS Support case (Enterprise support)
2. **LLM Provider Issues**: Contact provider support with specific error codes
3. **Security Issues**: Contact security team immediately

### Communication Channels
- **Internal**: Slack #incidents channel
- **External**: Status page updates (status.causal-gym.org)
- **Users**: Email notifications for SEV1/SEV2 incidents

## Post-Incident Analysis

### Immediate Actions (Within 24 Hours)
1. **Timeline Creation**: Document incident timeline
2. **Root Cause Analysis**: Identify primary and contributing causes
3. **Impact Assessment**: Quantify user impact and business effects

### Follow-up Actions (Within 1 Week)
1. **Postmortem Meeting**: Schedule with all stakeholders
2. **Action Items**: Create specific, time-bound improvement tasks
3. **Documentation Updates**: Update runbooks and procedures

### Postmortem Template

```markdown
# Incident Postmortem: [INCIDENT_TITLE]

**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: SEVX
**Impact**: X users affected, Y minutes downtime

## Summary
Brief description of what happened.

## Timeline
- HH:MM - Incident began
- HH:MM - First alert
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix implemented
- HH:MM - Service restored

## Root Cause
What fundamentally caused this incident?

## Contributing Factors
What other factors made this incident worse or more likely?

## Resolution
How was the incident resolved?

## Action Items
1. [ ] Task 1 (Owner: @person, Due: YYYY-MM-DD)
2. [ ] Task 2 (Owner: @person, Due: YYYY-MM-DD)

## Lessons Learned
What did we learn from this incident?
```

## Monitoring and Alerting

### Critical Alerts
- **Service Down**: Health check failures
- **High Error Rate**: >5% error rate for 5 minutes
- **High Response Time**: >95th percentile >5s for 10 minutes
- **Resource Exhaustion**: CPU >90% or Memory >95% for 15 minutes

### Alert Channels
- **PagerDuty**: For SEV1/SEV2 incidents
- **Slack**: For all alerts
- **Email**: For daily/weekly summaries

### Alert Suppression
During maintenance windows or known issues:
```bash
# Suppress alerts for maintenance
curl -X POST http://alertmanager:9093/api/v1/silences \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [
      {"name": "alertname", "value": "ServiceDown"}
    ],
    "startsAt": "2025-08-02T10:00:00Z",
    "endsAt": "2025-08-02T12:00:00Z",
    "comment": "Scheduled maintenance",
    "createdBy": "oncall@causal-gym.org"
  }'
```

## Recovery Procedures

### Service Recovery
```bash
# Full system restart
docker-compose down
docker system prune -f
docker-compose up -d

# Database recovery
docker exec postgres pg_ctl restart -D /var/lib/postgresql/data

# Cache recovery
docker exec redis redis-cli flushall
```

### Data Recovery
```bash
# List available backups
docker exec postgres ls -la /backups/

# Restore from specific backup
docker exec postgres pg_restore -U causal_user -d causal_experiments /backups/backup-YYYYMMDD.dump

# Verify data integrity
docker exec app python scripts/validate_data_integrity.py
```

### Configuration Recovery
```bash
# Restore from git
git checkout main
git pull origin main

# Restore environment variables
cp .env.backup .env

# Rebuild and restart
docker-compose build
docker-compose up -d
```

## Preventive Measures

### Regular Health Checks
- **Daily**: Automated system health verification
- **Weekly**: Performance benchmark comparison
- **Monthly**: Security audit and dependency updates

### Backup Verification
- **Daily**: Automated backup creation and verification
- **Weekly**: Restore test in staging environment
- **Monthly**: Full disaster recovery drill

### Capacity Planning
- **Monitor Growth**: Track usage trends and capacity metrics
- **Scale Proactively**: Add resources before reaching 80% capacity
- **Load Testing**: Regular load testing to identify bottlenecks

---

*This runbook is a living document. Update it after each incident to improve response procedures.*