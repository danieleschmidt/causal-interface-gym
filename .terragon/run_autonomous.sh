#!/bin/bash
# Terragon Autonomous SDLC Execution Script

set -e  # Exit on error

# Configuration
TERRAGON_DIR=".terragon"
LOG_FILE="$TERRAGON_DIR/execution.log"
LOCK_FILE="$TERRAGON_DIR/execution.lock"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to cleanup on exit
cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT

# Check for lock file (prevent concurrent execution)
if [ -f "$LOCK_FILE" ]; then
    log "ERROR: Another autonomous execution is already running"
    exit 1
fi

# Create lock file
echo $$ > "$LOCK_FILE"

log "🚀 Starting Terragon Autonomous SDLC Execution"

# Ensure terragon directory exists
mkdir -p "$TERRAGON_DIR"

# Check if Python environment is available
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python3 not found"
    exit 1
fi

# Check if required Python packages are available
python3 -c "import yaml" 2>/dev/null || {
    log "Installing required packages..."
    pip3 install pyyaml
}

# Set environment variables
export TERRAGON_DRY_RUN="${TERRAGON_DRY_RUN:-false}"
export PYTHONPATH="$TERRAGON_DIR:$PYTHONPATH"

# Create virtual environment if it doesn't exist
if [ ! -d "$TERRAGON_DIR/venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$TERRAGON_DIR/venv"
fi

# Activate virtual environment
source "$TERRAGON_DIR/venv/bin/activate"

# Install dependencies
pip install pyyaml > /dev/null 2>&1

# Run the autonomous executor
log "🎯 Executing autonomous value discovery and delivery..."

cd "$TERRAGON_DIR"
python3 autonomous_executor.py 2>&1 | tee -a "../$LOG_FILE"
execution_result=$?

cd ..

if [ $execution_result -eq 0 ]; then
    log "✅ Autonomous execution completed successfully"
    
    # Update metrics and documentation
    if [ -f "AUTONOMOUS_BACKLOG.md" ]; then
        log "📊 Updated autonomous backlog documentation"
    fi
    
    # Check if we have changes to commit (in non-dry-run mode)
    if [ "$TERRAGON_DRY_RUN" != "true" ]; then
        if git diff --quiet && git diff --cached --quiet; then
            log "📋 No changes to commit"
        else
            log "📝 Changes detected - they should be committed by the executor"
        fi
    fi
    
else
    log "❌ Autonomous execution failed with exit code $execution_result"
fi

# Generate summary report
log "📈 Execution Summary:"
if [ -f "$TERRAGON_DIR/value-metrics.json" ]; then
    log "$(cat "$TERRAGON_DIR/value-metrics.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
metrics = data.get('backlog_metrics', {})
print(f\"  - Backlog Items: {metrics.get('total_items', 0)}\")
print(f\"  - Discovery Rate: {metrics.get('discovery_rate', 0):.1f} items/day\")
print(f\"  - Completion Rate: {metrics.get('completion_rate', 0):.1f}%\")
print(f\"  - Value Delivered: \${metrics.get('value_delivered', 0):,.0f}\")
" 2>/dev/null || echo "  - Metrics not available")"
fi

log "🔄 Next scheduled execution: $(date -d '+1 hour' '+%Y-%m-%d %H:%M:%S')"
log "📊 View backlog: cat AUTONOMOUS_BACKLOG.md"
log "📋 View logs: tail -f $LOG_FILE"

echo
echo "🎉 Terragon Autonomous SDLC execution complete!"
echo "   View detailed backlog: cat AUTONOMOUS_BACKLOG.md"
echo "   View execution logs: tail -f $LOG_FILE"
echo "   Schedule regular runs: Add to crontab"
echo
echo "Suggested cron entry (hourly execution):"
echo "0 * * * * cd $(pwd) && .terragon/run_autonomous.sh"