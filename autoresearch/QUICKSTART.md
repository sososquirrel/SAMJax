# Autoresearch Quickstart

## What Changed?

You now have 3 new tools + enhanced `compare.py`:

| File | Purpose |
|------|---------|
| `config.yaml` | Centralized config (paths, thresholds, timeouts) |
| `run_iteration.py` | Core automation: submit → poll → compare → commit → log |
| `watch-iteration.sh` | Real-time log monitoring (optional) |
| `compare.py` (enhanced) | File suggestions + JSON convergence tracking |

## Quick Start (5 minutes)

### 1. Single Iteration (Manual)

```bash
cd /glade/u/home/sabramian/SAMJax/autoresearch

# Dry-run first to see what would happen
python run_iteration.py --dry-run

# Then run for real
python run_iteration.py
```

This will:
1. Submit the PBS job (`qsub run_irma_debug5_gpu.pbs`)
2. Poll `qstat` until it completes (with timeout)
3. Run `compare.py` automatically
4. Suggest which files to fix based on first diverging stage
5. Update `progress.md` with results
6. Auto-commit code changes (with `--no-commit` to skip)

**Output**: Exits 0 if all match, 1 if mismatch

### 2. Real-Time Monitoring (Optional)

While the job is running, in another terminal:

```bash
cd /glade/u/home/sabramian/SAMJax/autoresearch
bash watch-iteration.sh
```

This tails the log file and alerts when done.

## Full Workflow

### With `/loop` (Self-Paced Iterations)

You provide the fixes, Claude automates the runs:

```
/loop python /glade/u/home/sabramian/SAMJax/autoresearch/run_iteration.py
```

This will:
- Run iteration 1 (submit → compare → log result)
- You read the result, decide what to fix
- Edit code in jsam/
- Loop fires again, runs iteration 2 automatically
- Repeat until all stages match

**Advantages:**
- You stay in control of what to fix
- Full context from prior analysis
- Session resets between iterations (no token bloat)

### With `/schedule` (Fully Autonomous, Hands-Off)

Set up a cron job that runs every hour:

```
/schedule "cd /glade/u/home/sabramian/SAMJax/autoresearch && python run_iteration.py" 0 */1 * * *
```

Then check back periodically:

```bash
tail -f /glade/u/home/sabramian/SAMJax/autoresearch/progress.md
```

## Configuration

Edit `config.yaml` to change defaults:

```yaml
compare:
  threshold: 1e-5          # Relative error tolerance
  max_rows: 38             # Step 1 only (19 stages × 2 icycles)
  
job:
  timeout_sec: 7200        # 2-hour max wait
  poll_interval_sec: 10    # Check job status every 10s

git:
  auto_commit: true        # Commit code changes automatically
```

## Understanding the Output

### Progress File

After each iteration, `progress.md` is updated:

```markdown
## Iteration 1 — 2026-04-16 14:30:42
**Changed**: (pending — fix needed for Stage 3 diverges...)
**Result**: Stage 3 diverges (max rel err: 1.23e-04)
**Root cause**: (to be identified)
**Next**: Investigate stage 3
```

You then edit the jsam code, and on next iteration, fill in **Changed**, **Root cause**, **Next**.

### Convergence JSON

After each run, `convergence.json` contains structured data:

```json
{
  "timestamp": "2026-04-16T14:30:42.123456",
  "threshold": 1e-5,
  "result": "MISMATCH",
  "first_diverging_stage": 3,
  "stages": {
    "0": {"name": "pre_step", "max_rel_err": 1.2e-14, "status": "OK", ...},
    "1": {"name": "forcing", "max_rel_err": 2.3e-15, "status": "OK", ...},
    "3": {"name": "buoyancy", "max_rel_err": 1.23e-4, "status": "MISMATCH", ...},
    ...
  }
}
```

This is useful for tracking convergence over iterations (Python/matplotlib can plot it).

### File Suggestions

When a stage diverges, `run_iteration.py` + `compare.py` suggest which files to read:

```
FIRST DIVERGENCE: stage 3 (step 1)
  - buoyancy_min(rel=1.23e-04)

Suggested files to review:
  - jsam: /glade/u/home/sabramian/SAMJax/jsam/core/physics/buoyancy.py
  - gSAM: /glade/u/home/sabramian/gSAM1.8.7/SRC/buoyancy.f90
```

## Typical Fix Workflow

1. **Run iteration**: `python run_iteration.py`
2. **Read output**: See which stage diverges, which files are suggested
3. **Compare code**: `diff jsam/core/physics/buoyancy.py gSAM/buoyancy.f90`
4. **Fix jsam**: Edit the file to match gSAM
5. **Commit**: Optionally `git commit -m "fix: stage 3 buoyancy"` (or use auto-commit)
6. **Next iteration**: `python run_iteration.py` again

The loop auto-commits with a message like:

```
fix: stage 3 buoyancy (iter 2)
```

So you can later `git log` to see what each iteration changed.

## Advanced: Batch Testing

To test multiple fixes in parallel:

```bash
# Iteration 1: test fix A
(cd /path/to/branch-A && python run_iteration.py)

# Iteration 2: test fix B (in another branch)
(cd /path/to/branch-B && python run_iteration.py)
```

Then compare convergence JSON outputs.

## Troubleshooting

### Job fails to submit
```bash
# Check cluster is responsive
qstat -u sabramian
```

### Job times out
Increase `timeout_sec` in `config.yaml`:
```yaml
job:
  timeout_sec: 10800  # 3 hours
```

### Output files missing
Check the log for crashes:
```bash
tail -50 /glade/u/home/sabramian/SAMJax/outputs/debug5_gpu.log
```

### Auto-commit fails
Disable it:
```bash
python run_iteration.py --no-commit
```

## Key Files

| Path | Purpose |
|------|---------|
| `config.yaml` | All configuration |
| `run_iteration.py` | Main orchestration script |
| `watch-iteration.sh` | Real-time log monitoring |
| `compare.py` | Enhanced comparison (with file suggestions + JSON) |
| `design.md` | Maps stages to jsam/gSAM modules |
| `progress.md` | Experiment log (auto-updated) |
| `convergence.json` | Structured convergence data (per iteration) |

## Session Management

Your concern about growing sessions: use `/loop` or `/schedule`.

- **`/loop`**: Each iteration is isolated. You edit code between iterations. Claude submits next iteration automatically. **Best for interactive debugging.**

- **`/schedule`**: Fully autonomous. Claude submits jobs on a cron schedule. You check `progress.md` when you want. **Best for long-running campaigns.**

Either way, session context stays clean because iterations are independent tasks.

---

**Ready to go?** Start with:

```bash
python run_iteration.py --dry-run
```

Then run for real and watch the magic! 🚀
