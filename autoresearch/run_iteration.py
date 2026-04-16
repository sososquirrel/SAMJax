#!/usr/bin/env python3
"""Automated iteration loop: submit job → poll → compare → update progress → commit.

Usage:
    python run_iteration.py [--config CONFIG] [--dry-run] [--no-commit] [--iteration N]

Exits 0 if all stages match, 1 if mismatch found, 2+ for errors.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

CONFIG_DEFAULT = Path(__file__).parent / "config.yaml"
AUTORESEARCH_ROOT = Path(__file__).parent


def load_config(config_path):
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd, check=True, capture=True):
    """Run shell command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=capture,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"ERROR running command: {e}")
        return 999, "", str(e)


def submit_job(config, dry_run=False):
    """Submit PBS job. Returns job ID or None on failure."""
    script = config["job"]["script"]
    cmd = f"qsub {script}"

    print(f"[SUBMIT] {cmd}")
    if dry_run:
        print("  (dry-run, not submitted)")
        return "DRY_RUN_JOB_ID"

    rc, stdout, stderr = run_cmd(cmd)
    if rc != 0:
        print(f"ERROR submitting job: {stderr}")
        return None

    job_id = stdout.strip()
    print(f"  → Job ID: {job_id}")
    return job_id


def poll_job(job_id, config, dry_run=False):
    """Poll qstat until job completes. Returns (completed, elapsed_sec)."""
    timeout_sec = config["job"]["timeout_sec"]
    poll_interval = config["job"]["poll_interval_sec"]
    job_name = config["job"]["name"]

    print(f"[POLL] Waiting for job {job_id}...")
    start = time.time()

    if dry_run:
        print("  (dry-run, skipping poll)")
        return True, 0

    while True:
        rc, stdout, _ = run_cmd(f"qstat -u sabramian")
        if rc != 0:
            print(f"  WARNING: qstat failed, retrying...")
            time.sleep(poll_interval)
            continue

        # Check if job still in queue
        if job_id not in stdout and job_name not in stdout:
            elapsed = time.time() - start
            print(f"  → Job completed in {elapsed:.0f}s")
            return True, elapsed

        elapsed = time.time() - start
        if elapsed > timeout_sec:
            print(f"  ERROR: Job timeout after {elapsed:.0f}s")
            return False, elapsed

        # Print progress every 30s
        if int(elapsed) % 30 == 0:
            print(f"  ... still running ({elapsed:.0f}s)")

        time.sleep(poll_interval)


def check_outputs(config, dry_run=False):
    """Check if output files exist."""
    jsam_csv = Path(config["jsam"]["debug_dir"]) / "globals.csv"

    print(f"[CHECK] Looking for {jsam_csv}")
    if dry_run:
        print("  (dry-run, skipping check)")
        return True

    if not jsam_csv.exists():
        print(f"  ERROR: {jsam_csv} not found")
        return False

    print(f"  ✓ Found")
    return True


def run_compare(config, dry_run=False):
    """Run compare.py and return (exit_code, first_bad_stage, convergence_json)."""
    oracle_csv = Path(config["oracle"]["debug_dir"]) / "globals.csv"
    jsam_csv = Path(config["jsam"]["debug_dir"]) / "globals.csv"
    json_out = Path(config["output"]["convergence_json"])
    threshold = config["compare"]["threshold"]
    max_rows = config["compare"]["max_rows"]
    design = AUTORESEARCH_ROOT / "design.md"

    cmd = (
        f"python {AUTORESEARCH_ROOT}/compare.py "
        f"--oracle {oracle_csv} --jsam {jsam_csv} "
        f"--max-rows {max_rows} --threshold {threshold} "
        f"--json-out {json_out} --design {design}"
    )

    print(f"[COMPARE]")
    if dry_run:
        print("  (dry-run, not running)")
        return 0, None, None

    rc, stdout, stderr = run_cmd(cmd)
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    # Parse JSON output
    first_bad_stage = None
    convergence_json = None
    if json_out.exists():
        with open(json_out) as f:
            convergence_json = json.load(f)
        first_bad_stage = convergence_json.get("first_diverging_stage")

    return rc, first_bad_stage, convergence_json


def read_progress(config):
    """Read existing progress.md and count iterations."""
    progress_file = Path(config["output"]["progress_file"])
    if not progress_file.exists():
        return 0, ""

    with open(progress_file) as f:
        content = f.read()

    # Count "## Iteration N" headers
    iteration_count = content.count("## Iteration")
    return iteration_count, content


def update_progress(config, iteration_num, first_bad_stage, convergence_json, dry_run=False):
    """Append entry to progress.md."""
    progress_file = Path(config["output"]["progress_file"])

    # Format result summary
    if convergence_json is None:
        result_str = "ERROR: no comparison data"
    elif convergence_json.get("result") == "MATCH":
        result_str = "✓ ALL STAGES MATCH"
    else:
        bad_stage = convergence_json.get("first_diverging_stage")
        bad_data = convergence_json.get("stages", {}).get(str(bad_stage), {})
        max_rel = bad_data.get("max_rel_err", "?")
        result_str = f"Stage {bad_stage} diverges (max rel err: {max_rel:.2e})"

    entry = f"""
## Iteration {iteration_num} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Changed**: (pending — fix needed for {result_str})
**Result**: {result_str}
**Root cause**: (to be identified)
**Next**: Investigate stage {first_bad_stage if first_bad_stage is not None else '?'}
"""

    print(f"[UPDATE] Appending to {progress_file}")
    if not dry_run:
        with open(progress_file, "a") as f:
            f.write(entry)
        print(f"  ✓ Updated")
    else:
        print("  (dry-run, not updated)")


def auto_commit(config, iteration_num, first_bad_stage, dry_run=False):
    """Commit code changes with auto-generated message."""
    if not config["git"]["auto_commit"]:
        return True

    # Check if there are changes to commit
    rc, stdout, _ = run_cmd("git status --porcelain jsam/")
    if rc != 0 or not stdout.strip():
        print(f"[COMMIT] No changes to commit")
        return True

    stage_name = {
        0: "pre_step", 1: "forcing", 2: "nudging", 3: "buoyancy",
        4: "radiation", 5: "surface", 6: "advect_mom", 7: "coriolis",
        8: "sgs_proc", 9: "sgs_mom", 10: "adamsA", 11: "damping",
        12: "adamsB", 13: "pressure", 14: "advect_scalars", 15: "sgs_scalars",
        16: "upperbound", 17: "micro", 18: "diagnose",
    }.get(first_bad_stage, f"stage_{first_bad_stage}")

    msg = f"fix: stage {first_bad_stage} {stage_name} (iter {iteration_num})"

    print(f"[COMMIT] {msg}")
    if dry_run:
        print("  (dry-run, not committed)")
        return True

    # Stage changes
    rc, _, stderr = run_cmd("git add jsam/")
    if rc != 0:
        print(f"  ERROR staging changes: {stderr}")
        return False

    # Commit
    rc, stdout, stderr = run_cmd(f"git commit -m '{msg}'")
    if rc != 0 and "nothing to commit" not in stderr:
        print(f"  WARNING: commit failed: {stderr}")
        return False

    print(f"  ✓ Committed")
    return True


def print_context():
    """Print instructions, design, and progress for loop context."""
    files = {
        "INSTRUCTIONS": AUTORESEARCH_ROOT / "instructions.md",
        "DESIGN (Stage Mapping)": AUTORESEARCH_ROOT / "design.md",
        "PROGRESS (Prior Iterations)": AUTORESEARCH_ROOT / "progress.md",
    }

    for label, fpath in files.items():
        if not fpath.exists():
            continue

        print(f"\n{'─'*70}")
        print(f"[{label}]")
        print(f"{'─'*70}")

        with open(fpath) as f:
            content = f.read()

        # For design.md, only print the stage mapping section
        if "design" in label.lower():
            if "## 19 Stages" in content:
                start = content.find("## 19 Stages")
                end = content.find("## Oracle", start)
                if end == -1:
                    end = len(content)
                content = content[start:end]

        # For progress.md, print it all (but cap at 5000 chars if it gets huge)
        if len(content) > 5000 and "progress" in label.lower():
            content = "...[earlier iterations omitted]...\n\n" + content[-4000:]

        print(content)

    print(f"\n{'='*70}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    p.add_argument("--dry-run", action="store_true", help="Show what would happen")
    p.add_argument("--no-commit", action="store_true", help="Skip auto-commit")
    p.add_argument("--iteration", type=int, default=None, help="Override iteration number")
    p.add_argument("--no-context", action="store_true", help="Skip printing context files")
    args = p.parse_args()

    # Print context (instructions, design, progress) for loop iterations
    if not args.no_context:
        print_context()

    # Load config
    if not args.config.exists():
        print(f"ERROR: config not found at {args.config}")
        return 2

    config = load_config(args.config)
    if args.no_commit:
        config["git"]["auto_commit"] = False

    # Determine iteration number
    iter_num, _ = read_progress(config)
    if args.iteration is not None:
        iter_num = args.iteration
    else:
        iter_num += 1

    print(f"\n{'='*70}")
    print(f"ITERATION {iter_num}")
    print(f"{'='*70}\n")

    # Step 1: Submit job
    job_id = submit_job(config, dry_run=args.dry_run)
    if job_id is None:
        return 2

    # Step 2: Poll for completion
    completed, elapsed = poll_job(job_id, config, dry_run=args.dry_run)
    if not completed:
        return 2

    # Step 3: Check outputs exist
    if not check_outputs(config, dry_run=args.dry_run):
        return 2

    # Step 4: Run comparison
    rc, first_bad_stage, convergence_json = run_compare(config, dry_run=args.dry_run)

    # Step 5: Update progress.md
    update_progress(config, iter_num, first_bad_stage, convergence_json, dry_run=args.dry_run)

    # Step 6: Auto-commit if enabled
    if not args.no_commit:
        auto_commit(config, iter_num, first_bad_stage, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*70}")
    if convergence_json and convergence_json.get("result") == "MATCH":
        print(f"✓ ITERATION {iter_num}: SUCCESS — All stages match!")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"✗ ITERATION {iter_num}: MISMATCH at stage {first_bad_stage}")
        if convergence_json:
            stage_data = convergence_json.get("stages", {}).get(str(first_bad_stage), {})
            print(f"  Max rel err: {stage_data.get('max_rel_err', '?'):.2e}")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
