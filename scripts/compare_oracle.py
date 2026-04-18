#!/usr/bin/env python3
"""
compare_oracle.py — compare jsam debug globals.csv against gSAM oracle.

Usage:
    python scripts/compare_oracle.py [--steps N] [--stages s0,s1,...] [--vars v0,v1,...]

Outputs a per-step × per-stage × per-variable table of ratios and relative
differences between jsam and gSAM.
"""
import argparse
import sys
import numpy as np
import pandas as pd

JSAM_CSV  = "/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug/globals.csv"
GSAM_CSV  = "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv"

STAT_VARS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
STATS     = ["min", "max", "mean"]
COLS      = [f"{v}_{s}" for v in STAT_VARS for s in STATS]

STAGE_NAMES = {
    0: "pre_step", 1: "forcing",  2: "nudging",   3: "buoyancy",
    4: "radiation",5: "surface",  6: "advect_mom", 7: "coriolis",
    8: "sgs_proc", 9: "sgs_mom", 10: "adamsA",    11: "damping",
   12: "adamsB",  13: "pressure",14: "advect_scalars",15:"sgs_scalars",
   16: "upperbound",17:"micro",  18: "diagnose",
}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # For gSAM step 1, there are 3 icycle entries — keep only the last one per (nstep, stage_id)
    df = df.drop_duplicates(subset=["nstep", "stage_id"], keep="last").reset_index(drop=True)
    return df.set_index(["nstep", "stage_id"])


def rel_diff(jsam_val: float, gsam_val: float) -> str:
    """Return relative difference as percentage string."""
    if gsam_val == 0:
        if jsam_val == 0:
            return "  0.000%"
        return f"{jsam_val/1e-10:+10.3f}x(abs)"
    rd = (jsam_val - gsam_val) / abs(gsam_val) * 100.0
    return f"{rd:+9.3f}%"


def ratio(jsam_val: float, gsam_val: float) -> str:
    if gsam_val == 0:
        return "       N/A" if jsam_val == 0 else f"{jsam_val:.3e}(abs)"
    r = jsam_val / gsam_val
    return f"{r:10.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsam-csv", type=str, default=JSAM_CSV,
                    help="Path to jsam globals.csv (default: hardcoded path).")
    ap.add_argument("--gsam-csv", type=str, default=GSAM_CSV,
                    help="Path to gSAM oracle globals.csv (default: hardcoded path).")
    ap.add_argument("--steps",  type=str, default=None,
                    help="Comma-separated step numbers, or 'all'. Default: first 10.")
    ap.add_argument("--stages", type=str, default=None,
                    help="Comma-separated stage ids. Default: all 19.")
    ap.add_argument("--vars",   type=str, default=None,
                    help="Comma-separated variable names from UVWTABSQCQVQI. Default: all.")
    ap.add_argument("--no-mean", action="store_true",
                    help="Suppress mean column (show only min/max).")
    ap.add_argument("--max-reldiff", type=float, default=None,
                    help="Only show rows with |reldiff| > this threshold (%).")
    args = ap.parse_args()

    print(f"Loading jsam CSV …  {args.jsam_csv}")
    jsam = load_csv(args.jsam_csv)
    print(f"Loading gSAM CSV …  {args.gsam_csv}")
    gsam = load_csv(args.gsam_csv)

    jsam_steps = sorted(jsam.index.get_level_values("nstep").unique())
    gsam_steps = sorted(gsam.index.get_level_values("nstep").unique())
    common_steps = sorted(set(jsam_steps) & set(gsam_steps))

    if args.steps and args.steps != "all":
        want = [int(s) for s in args.steps.split(",")]
        common_steps = [s for s in common_steps if s in want]
    elif args.steps != "all":
        common_steps = common_steps[:10]  # default: first 10

    want_stages = list(range(19))
    if args.stages:
        want_stages = [int(s) for s in args.stages.split(",")]

    want_vars = STAT_VARS
    if args.vars:
        want_vars = [v.upper() for v in args.vars.split(",")]

    stats = ["min", "max"] if args.no_mean else STATS

    print(f"\nComparing {len(common_steps)} steps × {len(want_stages)} stages × "
          f"{len(want_vars)} vars × {len(stats)} stats")
    print(f"Steps:  {common_steps[:5]}{'…' if len(common_steps)>5 else ''}")
    print()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = f"{'step':>5} {'stage':>15}  {'variable':>10}  {'stat':>4}  "
    hdr += f"{'jsam':>14}  {'gSAM':>14}  {'ratio':>10}  {'reldiff':>9}"
    print(hdr)
    print("-" * len(hdr))

    rows_shown = 0
    max_reldiffs: dict[str, float] = {}  # track worst reldiff per variable

    for step in common_steps:
        for sid in want_stages:
            if (step, sid) not in jsam.index or (step, sid) not in gsam.index:
                continue
            jrow = jsam.loc[(step, sid)]
            grow = gsam.loc[(step, sid)]
            stage_name = STAGE_NAMES.get(sid, str(sid))

            for var in want_vars:
                for stat in stats:
                    col = f"{var}_{stat}"
                    if col not in COLS:
                        continue
                    jval = float(jrow[col])
                    gval = float(grow[col])

                    # Compute relative diff for filtering
                    if gval != 0:
                        rd_pct = abs((jval - gval) / abs(gval)) * 100.0
                    else:
                        rd_pct = 0.0 if jval == 0 else float("inf")

                    # Track max reldiff
                    key = f"{var}_{stat}"
                    if rd_pct > max_reldiffs.get(key, 0.0):
                        max_reldiffs[key] = rd_pct

                    if args.max_reldiff is not None and rd_pct < args.max_reldiff:
                        continue

                    r_str  = ratio(jval, gval)
                    rd_str = rel_diff(jval, gval)
                    print(f"{step:>5} {stage_name:>15}  {var:>10}  {stat:>4}  "
                          f"{jval:>14.6e}  {gval:>14.6e}  {r_str}  {rd_str}")
                    rows_shown += 1

    print(f"\n{'─'*80}")
    print(f"Rows shown: {rows_shown}")
    print()

    # ── Summary: worst relative diff per variable ──────────────────────────
    print("=== Worst relative difference per variable (over shown steps/stages) ===")
    print(f"{'variable':>12}  {'max_reldiff':>12}")
    print("-" * 28)
    for key in sorted(max_reldiffs):
        rd = max_reldiffs[key]
        flag = " ****" if rd > 5.0 else (" **" if rd > 1.0 else "")
        print(f"{key:>12}  {rd:>11.4f}%{flag}")


if __name__ == "__main__":
    main()
