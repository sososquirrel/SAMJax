#!/usr/bin/env python3
"""
compare_oracle.py — compare jsam debug globals.csv against gSAM oracle.

Usage:
    python scripts/compare_oracle.py --log outputs/run_<RUN_ID>.log
    python scripts/compare_oracle.py --jsam-csv <path>/globals.csv
    python scripts/compare_oracle.py --log <log> --steps 1,2,3 --vars W --stages 10,11,12,13

Outputs a per-step × per-stage × per-variable table of ratios and relative
differences between jsam and gSAM.
"""
import argparse
import re
import sys
from pathlib import Path

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
    # Multiple icycles per (nstep, stage_id) are possible; keep only the first (pre-blowup)
    df = df.drop_duplicates(subset=["nstep", "stage_id"], keep="first").reset_index(drop=True)
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


def _read_bin_header(fp):
    """Parse the 128-byte header of a rank_NNNNNN.bin file."""
    nx_gl, ny_gl, nzm = np.fromfile(fp, "<i4", 3)
    _ = np.fromfile(fp, "<f4", 4)  # latlon box (unused here)
    i_min_gl, i_max_gl, j_min_gl, j_max_gl = np.fromfile(fp, "<i4", 4)
    it_off, jt_off = np.fromfile(fp, "<i4", 2)
    i1_loc, i2_loc, j1_loc, j2_loc = np.fromfile(fp, "<i4", 4)
    nfields = int(np.fromfile(fp, "<i4", 1)[0])
    fp.read(8 * nfields)  # field names (unused — known order)
    return dict(
        nx_gl=int(nx_gl), ny_gl=int(ny_gl), nzm=int(nzm),
        i_min_gl=int(i_min_gl), i_max_gl=int(i_max_gl),
        j_min_gl=int(j_min_gl), j_max_gl=int(j_max_gl),
        it_off=int(it_off), jt_off=int(jt_off),
        i1_loc=int(i1_loc), i2_loc=int(i2_loc),
        j1_loc=int(j1_loc), j2_loc=int(j2_loc),
        nfields=nfields,
    )


def load_irma_subbox(debug_dir: str, step: int, sid: int):
    """Load the full (7, nzm, nj, ni) IRMA-subbox tensor at (step, sid).

    Works for jsam (single rank) or gSAM (144 ranks, stitched). Returns
    (tensor, box_info) or (None, None) if no record found.
    """
    rank_files = sorted(Path(debug_dir).glob("rank_*.bin"))
    if not rank_files:
        return None, None

    full = None
    box_info = None
    for rfile in rank_files:
        with open(rfile, "rb") as f:
            hdr = _read_bin_header(f)
            ni_loc = hdr["i2_loc"] - hdr["i1_loc"] + 1
            nj_loc = hdr["j2_loc"] - hdr["j1_loc"] + 1
            if ni_loc <= 0 or nj_loc <= 0:
                continue  # this rank doesn't overlap the IRMA box

            if full is None:
                ni_full = hdr["i_max_gl"] - hdr["i_min_gl"] + 1
                nj_full = hdr["j_max_gl"] - hdr["j_min_gl"] + 1
                nzm     = hdr["nzm"]
                full = np.full((7, nzm, nj_full, ni_full), np.nan, dtype=np.float32)
                box_info = dict(
                    i_min_gl=hdr["i_min_gl"], j_min_gl=hdr["j_min_gl"],
                    ni_full=ni_full, nj_full=nj_full, nzm=nzm,
                )
            nzm = hdr["nzm"]
            payload_floats = 7 * nzm * nj_loc * ni_loc
            payload_bytes  = payload_floats * 4

            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    break
                nstep_rec, sid_rec = np.frombuffer(chunk, "<i4")
                if int(nstep_rec) == step and int(sid_rec) == sid:
                    buf  = np.fromfile(f, "<f4", payload_floats)
                    # Fortran [ni, nj, nzm, 7] == C (7, nzm, nj, ni)
                    tile = buf.reshape((7, nzm, nj_loc, ni_loc))
                    box_i0 = (hdr["it_off"] + hdr["i1_loc"]) - hdr["i_min_gl"]
                    box_j0 = (hdr["jt_off"] + hdr["j1_loc"]) - hdr["j_min_gl"]
                    full[:, :, box_j0:box_j0+nj_loc, box_i0:box_i0+ni_loc] = tile
                    break
                else:
                    f.seek(payload_bytes, 1)

    if full is None or np.all(np.isnan(full)):
        return None, None
    return full, box_info


def report_max_deviation_cells(jsam_dir: str, gsam_dir: str,
                                step: int, sid: int, stage_name: str):
    """Print per-field max-|abs-diff| cell locations at (step, sid)."""
    print()
    print(f"=== Max per-cell |jsam − gSAM| at step {step}, stage '{stage_name}' "
          f"(id={sid}) ===")

    jt, jbox = load_irma_subbox(jsam_dir, step, sid)
    gt, gbox = load_irma_subbox(gsam_dir, step, sid)
    if jt is None:
        print(f"  (no jsam binary record found for step {step}, stage {sid})")
        return
    if gt is None:
        print(f"  (no gSAM binary record found for step {step}, stage {sid})")
        return
    if jt.shape != gt.shape:
        print(f"  (shape mismatch: jsam {jt.shape} vs gSAM {gt.shape})")
        return

    i_min_gl = jbox["i_min_gl"]
    j_min_gl = jbox["j_min_gl"]
    hdr = (f"{'variable':>8}  {'abs_diff':>12}  {'jsam':>14}  {'gSAM':>14}  "
           f"{'reldiff':>10}  {'(k,  j_box,  i_box)':>22}  "
           f"{'(k,  j_gl ,  i_gl )':>22}")
    print(hdr)
    print("-" * len(hdr))
    for fi, var in enumerate(STAT_VARS):
        jf = jt[fi]
        gf = gt[fi]
        mask = ~np.isnan(gf)
        if not mask.any():
            print(f"  {var:>8}  (no overlap)")
            continue
        diff = np.where(mask, np.abs(jf - gf), -1.0)
        k, j, i = np.unravel_index(np.argmax(diff), diff.shape)
        ad = float(diff[k, j, i])
        jv = float(jf[k, j, i])
        gv = float(gf[k, j, i])
        denom = max(abs(jv), abs(gv), 1e-30)
        rd = (jv - gv) / denom * 100.0
        i_gl = int(i) + i_min_gl   # 1-based global i
        j_gl = int(j) + j_min_gl   # 1-based global j
        print(f"  {var:>8}  {ad:12.4e}  {jv:14.6e}  {gv:14.6e}  "
              f"{rd:+9.3f}%  ({int(k):>3}, {int(j):>5}, {int(i):>5})  "
              f"({int(k):>3}, {j_gl:>5}, {i_gl:>5})")

    print()
    print(f"  Indices: k is 0-based vertical level. "
          f"(j_box, i_box) are 0-based within the IRMA subbox "
          f"[ni={jbox['ni_full']}, nj={jbox['nj_full']}]. "
          f"(j_gl, i_gl) are 1-based global grid indices.")


def extract_debug_dir_from_log(log_path: str) -> str:
    """Parse a jsam run log and extract the debug_dump directory path."""
    pattern = re.compile(r"\[debug_dump\]\s+enabled\s+→\s+(\S+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return m.group(1)
    raise RuntimeError(
        f"No '[debug_dump] enabled → <path>' line in {log_path}. "
        "Was the run launched with --debug-dump-dir?"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None,
                    help="Path to jsam run log — auto-extracts debug dir and "
                         "sets --jsam-csv to <debug_dir>/globals.csv.")
    ap.add_argument("--jsam-csv", type=str, default=JSAM_CSV,
                    help="Path to jsam globals.csv (default: hardcoded path). "
                         "Ignored if --log is given.")
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
    ap.add_argument("--first_failing_only", action="store_true",
                    help="Print a single compact recap table at the first (step, stage) "
                         "where any tensor has |reldiff| > threshold, then stop. "
                         "Also reports the FOCUS variable (worst tensor) for the "
                         "autoresearch agent.")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="Relative-diff threshold in %% used by --first_failing_only "
                         "to define the frontier (default: 0.1).")
    args = ap.parse_args()

    if args.log:
        debug_dir = extract_debug_dir_from_log(args.log)
        jsam_csv_path = str(Path(debug_dir) / "globals.csv")
        print(f"Log:        {args.log}")
        print(f"Debug dir:  {debug_dir}")
    else:
        jsam_csv_path = args.jsam_csv

    print(f"Loading jsam CSV …  {jsam_csv_path}")
    jsam = load_csv(jsam_csv_path)
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

    # ── First-failing-only compact mode ──────────────────────────────────────
    if args.first_failing_only:
        for step in common_steps:
            for sid in want_stages:
                if (step, sid) not in jsam.index or (step, sid) not in gsam.index:
                    continue
                jrow = jsam.loc[(step, sid)]
                grow = gsam.loc[(step, sid)]

                # Check if any tensor at this (step, sid) has |reldiff| > threshold
                any_failing = False
                for var in want_vars:
                    for stat in stats:
                        col = f"{var}_{stat}"
                        if col not in COLS:
                            continue
                        jval = float(jrow[col])
                        gval = float(grow[col])
                        if gval != 0:
                            rd_pct = abs((jval - gval) / abs(gval)) * 100.0
                        else:
                            rd_pct = 0.0 if jval == 0 else float("inf")
                        if rd_pct > args.threshold:
                            any_failing = True
                            break
                    if any_failing:
                        break

                if not any_failing:
                    continue

                # Print full recap at this (step, sid) and stop
                stage_name = STAGE_NAMES.get(sid, str(sid))
                print(hdr)
                print("-" * len(hdr))
                # Track per-variable worst |reldiff| across its stats, plus the
                # count of tensors (var × stat) with max-error < threshold — the
                # autoresearch metric uses both to decide keep/reject.
                per_var_worst: dict[str, tuple[float, str, float, float]] = {}
                n_under_threshold = 0
                n_tensors_total = 0
                for var in want_vars:
                    for stat in stats:
                        col = f"{var}_{stat}"
                        if col not in COLS:
                            continue
                        jval = float(jrow[col])
                        gval = float(grow[col])
                        r_str  = ratio(jval, gval)
                        rd_str = rel_diff(jval, gval)
                        print(f"{step:>5} {stage_name:>15}  {var:>10}  {stat:>4}  "
                              f"{jval:>14.6e}  {gval:>14.6e}  {r_str}  {rd_str}")

                        if gval != 0:
                            rd_pct = abs((jval - gval) / abs(gval)) * 100.0
                        else:
                            rd_pct = 0.0 if jval == 0 else float("inf")
                        n_tensors_total += 1
                        if rd_pct < args.threshold:
                            n_under_threshold += 1
                        prev = per_var_worst.get(var)
                        if prev is None or rd_pct > prev[0]:
                            per_var_worst[var] = (rd_pct, stat, jval, gval)

                # Per-cell max deviation from binary dumps
                jsam_debug_dir = str(Path(jsam_csv_path).parent)
                gsam_debug_dir = str(Path(args.gsam_csv).parent)
                try:
                    report_max_deviation_cells(jsam_debug_dir, gsam_debug_dir,
                                               step, sid, stage_name)
                except Exception as e:
                    print(f"\n  (per-cell analysis failed: {e})")

                # ── FOCUS: worst variable at the frontier step ──────────────
                focus_var = max(per_var_worst, key=lambda v: per_var_worst[v][0])
                f_rd, f_stat, f_jv, f_gv = per_var_worst[focus_var]
                print()
                print("=" * 78)
                print(f"  FOCUS — autoresearch should work on this tensor:")
                print("=" * 78)
                print(f"    frontier step        : {step}")
                print(f"    frontier stage       : {stage_name} (id={sid})")
                print(f"    threshold used       : {args.threshold:.3f} %")
                print(f"    tensors < threshold  : {n_under_threshold} / {n_tensors_total}")
                print(f"    → FOCUS VARIABLE     : {focus_var}")
                print(f"      worst stat         : {f_stat}")
                print(f"      jsam               : {f_jv:.6e}")
                print(f"      gSAM               : {f_gv:.6e}")
                print(f"      |reldiff|          : {f_rd:.4f} %")
                print()
                print("  Metric recap (keep/reject a modification):")
                print(f"    1. Frontier step index > before? → KEEP")
                print(f"    2. Same index: more tensors < {args.threshold:.2f}% → KEEP, "
                      f"else REJECT")
                print(f"    3. Same index + same count: worst-variable deviation "
                      f"improved? → KEEP (equal → KEEP, worse → REJECT)")
                return

        print(f"No (step, stage) found with any tensor deviation > {args.threshold:.2f}%.")
        return

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
