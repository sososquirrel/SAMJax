#!/bin/bash
# Monitor job log, alert on completion, auto-run compare + update progress

set -euo pipefail

SAMJAX=/glade/u/home/sabramian/SAMJax
AUTORESEARCH=$SAMJAX/autoresearch
CONFIG=${CONFIG:-$AUTORESEARCH/config.yaml}

# Parse config for log file path
LOG_FILE=$(grep "log_file:" "$CONFIG" | awk '{print $2}' | tr -d "'\"")
TIMEOUT_SEC=$(grep "timeout_sec:" "$CONFIG" | awk '{print $2}')

echo "[WATCH] Monitoring log: $LOG_FILE"
echo "[WATCH] Will timeout after $TIMEOUT_SEC seconds"
echo ""

# Watch log in real-time, show last 20 lines, wait for "done:"
start=$(date +%s)
tail -f "$LOG_FILE" --pid=$$ &
TAIL_PID=$!

while true; do
  now=$(date +%s)
  elapsed=$((now - start))

  # Check if log contains completion marker
  if grep -q "=== done:" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "[WATCH] ✓ Job completed!"
    kill $TAIL_PID 2>/dev/null || true
    break
  fi

  # Check timeout
  if [ $elapsed -gt "$TIMEOUT_SEC" ]; then
    echo ""
    echo "[WATCH] ✗ Timeout after ${elapsed}s"
    kill $TAIL_PID 2>/dev/null || true
    exit 2
  fi

  sleep 5
done

echo ""
echo "[COMPARE] Running comparison..."
cd "$AUTORESEARCH"
python compare.py

echo ""
echo "[DONE] Iteration complete. Progress updated."
