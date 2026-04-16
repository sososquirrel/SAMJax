#!/usr/bin/env python3
"""Compare jsam globals.csv against gSAM oracle globals.csv.

Usage:
    python compare.py [--oracle PATH] [--jsam PATH] [--max-rows N] [--threshold T]

Exits 0 if all fields match within threshold, 1 otherwise.
Prints per-stage summary: first diverging stage highlighted.
"""
import argparse
import sys

import numpy as np
import pandas as pd

ORACLE_DEFAULT = "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv"
JSAM_DEFAULT   = "/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug/globals.csv"

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
STATS  = ["min", "max", "mean"]

# QC/QV/QI are stale before stage 17 — skip comparison
MICRO_FIELDS = {"QC", "QV", "QI"}
MICRO_STAGE_MIN = 17


def load(path):
    df = pd.read_csv(path)
    # strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def compare(oracle_path, jsam_path, max_rows=None, threshold=1e-5):
    odf = load(oracle_path)
    jdf = load(jsam_path)

    if max_rows:
        odf = odf.head(max_rows)
        jdf = jdf.head(max_rows)

    # Join on (nstep, stage_id)
    merged = odf.merge(jdf, on=["nstep", "stage_id"], suffixes=("_oracle", "_jsam"))
    if len(merged) == 0:
        print("ERROR: no matching (nstep, stage_id) rows found.")
        return 1

    print(f"Comparing {len(merged)} rows (steps {merged.nstep.min()}-{merged.nstep.max()})\n")

    all_ok = True
    first_bad_stage = None
    first_bad_step = None

    # Per-stage aggregation
    stages = sorted(merged.stage_id.unique())
    results = []

    for sid in stages:
        sname = merged.loc[merged.stage_id == sid, "stage_name_oracle"].iloc[0]
        sm = merged[merged.stage_id == sid]
        stage_max_err = 0.0
        stage_max_rel = 0.0
        bad_fields = []

        for field in FIELDS:
            if field in MICRO_FIELDS and sid < MICRO_STAGE_MIN:
                continue
            for stat in STATS:
                col = f"{field}_{stat}"
                col_o = f"{col}_oracle"
                col_j = f"{col}_jsam"
                if col_o not in sm.columns or col_j not in sm.columns:
                    continue
                diff = np.abs(sm[col_o].values - sm[col_j].values)
                denom = np.maximum(np.abs(sm[col_o].values), 1e-30)
                rel = diff / denom
                max_abs = diff.max()
                max_rel = rel.max()
                if max_abs > stage_max_err:
                    stage_max_err = max_abs
                if max_rel > stage_max_rel:
                    stage_max_rel = max_rel
                if max_rel > threshold:
                    bad_fields.append(f"{col}(rel={max_rel:.2e})")

        status = "OK" if not bad_fields else "MISMATCH"
        if bad_fields and first_bad_stage is None:
            first_bad_stage = sid
            first_bad_step = int(sm.nstep.iloc[0])
            all_ok = False

        results.append((sid, sname, stage_max_err, stage_max_rel, status, bad_fields))

    # Print table
    print(f"{'ID':>3} {'Stage':<16} {'MaxAbsErr':>12} {'MaxRelErr':>12} {'Status':<10}")
    print("-" * 60)
    for sid, sname, mae, mre, status, bfields in results:
        marker = " <<<" if sid == first_bad_stage else ""
        print(f"{sid:>3} {sname:<16} {mae:>12.4e} {mre:>12.4e} {status:<10}{marker}")

    if not all_ok:
        print(f"\nFIRST DIVERGENCE: stage {first_bad_stage} (step {first_bad_step})")
        _, _, _, _, _, bfields = results[first_bad_stage]
        for bf in bfields[:10]:
            print(f"  - {bf}")
        print("\nFix this stage first — downstream errors may resolve automatically.")
        return 1
    else:
        print("\nALL STAGES MATCH within threshold.")
        return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", default=ORACLE_DEFAULT)
    p.add_argument("--jsam", default=JSAM_DEFAULT)
    p.add_argument("--max-rows", type=int, default=None, help="Limit to first N rows")
    p.add_argument("--threshold", type=float, default=1e-5, help="Relative error threshold")
    args = p.parse_args()
    sys.exit(compare(args.oracle, args.jsam, args.max_rows, args.threshold))
