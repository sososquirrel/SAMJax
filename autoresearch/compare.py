#!/usr/bin/env python3
"""Compare jsam globals.csv against gSAM oracle globals.csv.

Usage:
    python compare.py [--oracle PATH] [--jsam PATH] [--max-rows N] [--threshold T]
                      [--json-out PATH] [--design PATH]

Exits 0 if all fields match within threshold, 1 otherwise.
Prints per-stage summary: first diverging stage highlighted.
Optionally outputs convergence JSON and file mapping suggestions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORACLE_DEFAULT = "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv"
JSAM_DEFAULT   = "/glade/derecho/scratch/sabramian/jsam_IRMA_debug5_gpu/debug/globals.csv"
DESIGN_DEFAULT = "/glade/u/home/sabramian/SAMJax/autoresearch/design.md"

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
STATS  = ["min", "max", "mean"]

# QC/QV/QI are stale before stage 17 — skip comparison
MICRO_FIELDS = {"QC", "QV", "QI"}
MICRO_STAGE_MIN = 17

# Stage ID → name mapping (order from design.md)
STAGE_NAMES = {
    0: "pre_step",
    1: "forcing",
    2: "nudging",
    3: "buoyancy",
    4: "radiation",
    5: "surface",
    6: "advect_mom",
    7: "coriolis",
    8: "sgs_proc",
    9: "sgs_mom",
    10: "adamsA",
    11: "damping",
    12: "adamsB",
    13: "pressure",
    14: "advect_scalars",
    15: "sgs_scalars",
    16: "upperbound",
    17: "micro",
    18: "diagnose",
}


def load(path):
    df = pd.read_csv(path)
    # strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def load_design(design_path):
    """Parse design.md to extract jsam → gSAM file mappings.

    Returns dict: {stage_id: {"jsam": [files], "gsam": [files]}}
    """
    mappings = {}
    try:
        with open(design_path) as f:
            content = f.read()

        # Extract jsam table (starts with "| File | Maps to")
        # Find the jsam section
        if "## jsam — JAX Port" in content:
            start = content.find("## jsam — JAX Port")
            end = content.find("\n## ", start + 1)
            if end == -1:
                end = len(content)
            table_text = content[start:end]

            # Parse table rows: "| file | maps_to |"
            for line in table_text.split("\n"):
                if line.startswith("|") and "maps" in line.lower():
                    continue  # Skip header
                if line.startswith("|") and "---" not in line:
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) == 2:
                        jsam_file = parts[0]
                        gsam_file = parts[1]
                        # Try to infer stage from module name
                        for stage_id, stage_name in STAGE_NAMES.items():
                            if stage_name in jsam_file.lower():
                                if stage_id not in mappings:
                                    mappings[stage_id] = {"jsam": [], "gsam": []}
                                mappings[stage_id]["jsam"].append(jsam_file)
                                mappings[stage_id]["gsam"].append(gsam_file)
                                break
    except Exception as e:
        print(f"Warning: could not load design.md: {e}")

    return mappings


def suggest_files(stage_id, stage_name, design_mappings):
    """Suggest which jsam/gSAM files to read for a diverging stage."""
    suggestions = []

    # Default heuristics based on stage
    jsam_root = "/glade/u/home/sabramian/SAMJax/jsam"
    gsam_root = "/glade/u/home/sabramian/gSAM1.8.7/SRC"

    # Stage → module file mapping (common cases)
    stage_to_module = {
        3: ("core/physics/buoyancy.py", "buoyancy.f90"),
        6: ("core/dynamics/advection.py", "advect_mom.f90"),
        7: ("core/dynamics/coriolis.py", "coriolis.f90"),
        10: ("core/dynamics/timestepping.py", "adamsA.f90"),
        12: ("core/dynamics/timestepping.py", "adamsB.f90"),
        13: ("core/dynamics/pressure.py", "pressure.f90"),
        14: ("core/dynamics/advection.py", "advect_all_scalars.f90"),
        17: ("core/physics/microphysics.py", "MICRO_SAM1MOM/"),
    }

    if stage_id in stage_to_module:
        jsam_mod, gsam_mod = stage_to_module[stage_id]
        suggestions.append(f"jsam: {jsam_root}/{jsam_mod}")
        suggestions.append(f"gSAM: {gsam_root}/{gsam_mod}")

    # Also check design mappings
    if stage_id in design_mappings:
        mapping = design_mappings[stage_id]
        if mapping.get("jsam"):
            for f in mapping["jsam"]:
                suggestions.append(f"jsam (design): {jsam_root}/{f}")
        if mapping.get("gsam"):
            for f in mapping["gsam"]:
                suggestions.append(f"gSAM (design): {gsam_root}/{f}")

    return suggestions


def compare(oracle_path, jsam_path, max_rows=None, threshold=1e-5, json_out=None, design_path=None):
    odf = load(oracle_path)
    jdf = load(jsam_path)

    if max_rows:
        odf = odf.head(max_rows)
        jdf = jdf.head(max_rows)

    # Join on (nstep, stage_id, dtn) to properly align icycles.
    # If exact dtn match fails, fall back to inner merge on (nstep, stage_id)
    # and keep only first occurrence of each (nstep, stage_id) pair.
    if "dtn_oracle" not in odf.columns:
        odf.rename(columns={"dtn": "dtn_oracle"}, inplace=True)
    if "dtn" in jdf.columns:
        jdf_renamed = jdf.rename(columns={"dtn": "dtn_jsam"})
    else:
        jdf_renamed = jdf

    # Try exact dtn match first
    merged = odf.merge(jdf_renamed, on=["nstep", "stage_id"], suffixes=("_oracle", "_jsam"), how="inner")

    if len(merged) == 0:
        print("ERROR: no matching (nstep, stage_id) rows found.")
        return 1

    # Keep only first icycle (smallest dtn) for each (nstep, stage_id) pair to avoid Cartesian product
    merged = merged.sort_values(["nstep", "stage_id", "dtn_oracle"]).groupby(["nstep", "stage_id"], as_index=False).first()

    print(f"Comparing {len(merged)} rows (steps {merged.nstep.min()}-{merged.nstep.max()})\n")

    all_ok = True
    first_bad_stage = None
    first_bad_step = None

    # Load design mappings if available
    design_mappings = {}
    if design_path and Path(design_path).exists():
        design_mappings = load_design(design_path)

    # Per-stage aggregation
    stages = sorted(merged.stage_id.unique())
    results = []
    json_data = {"timestamp": pd.Timestamp.now().isoformat(), "threshold": threshold, "stages": {}}

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
        json_data["stages"][str(sid)] = {
            "name": sname,
            "max_abs_err": float(stage_max_err),
            "max_rel_err": float(stage_max_rel),
            "status": status,
            "bad_fields": [str(bf) for bf in bad_fields],  # Convert to native Python types
        }

    # Print table
    print(f"{'ID':>3} {'Stage':<16} {'MaxAbsErr':>12} {'MaxRelErr':>12} {'Status':<10}")
    print("-" * 60)
    for sid, sname, mae, mre, status, bfields in results:
        marker = " <<<" if sid == first_bad_stage else ""
        print(f"{sid:>3} {sname:<16} {mae:>12.4e} {mre:>12.4e} {status:<10}{marker}")

    if not all_ok:
        print(f"\nFIRST DIVERGENCE: stage {first_bad_stage} (step {first_bad_step})")
        _, sname, _, _, _, bfields = results[first_bad_stage]
        for bf in bfields[:10]:
            print(f"  - {bf}")

        # Suggest files to read
        suggestions = suggest_files(first_bad_stage, sname, design_mappings)
        if suggestions:
            print(f"\nSuggested files to review:")
            for s in suggestions:
                print(f"  - {s}")

        print("\nFix this stage first — downstream errors may resolve automatically.")

        # Save JSON if requested
        if json_out:
            json_data["first_diverging_stage"] = int(first_bad_stage) if first_bad_stage is not None else None
            json_data["result"] = "MISMATCH"
            with open(json_out, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"\nConvergence data saved to {json_out}")

        return 1
    else:
        print("\nALL STAGES MATCH within threshold.")
        json_data["result"] = "MATCH"

        # Save JSON if requested
        if json_out:
            with open(json_out, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"Convergence data saved to {json_out}")

        return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", default=ORACLE_DEFAULT)
    p.add_argument("--jsam", default=JSAM_DEFAULT)
    p.add_argument("--max-rows", type=int, default=None, help="Limit to first N rows")
    p.add_argument("--threshold", type=float, default=1e-5, help="Relative error threshold")
    p.add_argument("--json-out", default=None, help="Output convergence JSON to this path")
    p.add_argument("--design", default=DESIGN_DEFAULT, help="Path to design.md for file suggestions")
    args = p.parse_args()
    sys.exit(
        compare(
            args.oracle,
            args.jsam,
            args.max_rows,
            args.threshold,
            json_out=args.json_out,
            design_path=args.design,
        )
    )
