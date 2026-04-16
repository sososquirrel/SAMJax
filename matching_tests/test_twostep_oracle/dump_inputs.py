"""End-to-end 2-step comparison of jsam vs gSAM oracle debug dumps.

Reads the globals.csv files from both pipelines and produces per-stage,
per-field error reports. Also reads rank-0 binary tensor for spatial
error analysis.

Cases
-----
oracle_globals_step1     — all 19 stages of step 1
oracle_globals_step2     — all 19 stages of step 2
oracle_tensor_step1_adamsA — 3D tensor at step 1 stage 10
oracle_summary           — summary table across all available steps
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
SAMJAX_ROOT = MT_ROOT.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, str(SAMJAX_ROOT))

GSAM_GLOBALS = Path("/glade/derecho/scratch/sabramian/"
                    "gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv")
JSAM_GLOBALS = Path("/glade/derecho/scratch/sabramian/"
                    "jsam_IRMA_debug500/debug/globals.csv")
GSAM_RANK0   = Path("/glade/derecho/scratch/sabramian/"
                    "gSAM_IRMA_500timesteps_DEBUG/debug/rank_000000.bin")
JSAM_RANK0   = Path("/glade/derecho/scratch/sabramian/"
                    "jsam_IRMA_debug500/debug/rank_000000.bin")

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
METRICS = ["min", "max", "mean"]
STAGES = [
    (0, "pre_step"), (1, "forcing"), (2, "nudging"), (3, "buoyancy"),
    (4, "radiation"), (5, "surface"), (6, "advect_mom"), (7, "coriolis"),
    (8, "sgs_proc"), (9, "sgs_mom"), (10, "adamsA"), (11, "damping"),
    (12, "adamsB"), (13, "pressure"), (14, "advect_scalars"),
    (15, "sgs_scalars"), (16, "upperbound"), (17, "micro"), (18, "diagnose"),
]


def _load_globals():
    """Load and align the two globals CSVs."""
    import pandas as pd
    if not GSAM_GLOBALS.exists() or not JSAM_GLOBALS.exists():
        print("ERROR: globals.csv not found. Run debug500 jobs first.")
        sys.exit(1)

    g = pd.read_csv(GSAM_GLOBALS)
    j = pd.read_csv(JSAM_GLOBALS)
    g.columns = [c.strip() for c in g.columns]
    j.columns = [c.strip() for c in j.columns]
    g = g.drop_duplicates(subset=["nstep", "stage_id"], keep="last")
    j = j.drop_duplicates(subset=["nstep", "stage_id"], keep="last")
    g = g.set_index(["nstep", "stage_id"])
    j = j.set_index(["nstep", "stage_id"])
    common = g.index.intersection(j.index)
    return g.loc[common].sort_index(), j.loc[common].sort_index()


def _compare_step(g, j, step: int, outfile: str):
    """Compare all stages for a single step."""
    print(f"\n{'='*72}")
    print(f"  STEP {step} — stage-by-stage comparison")
    print(f"{'='*72}")
    print(f"{'sid':>3} {'stage':<14} {'worst_field':>20} {'rel_err':>10} {'abs_err':>12}")
    print("-" * 72)

    with open(outfile, "w") as f:
        f.write(f"step,sid,stage,field,metric,gsam,jsam,abs_err,rel_err\n")
        for sid, sname in STAGES:
            if (step, sid) not in g.index:
                continue
            gr = g.loc[(step, sid)]
            jr = j.loc[(step, sid)]
            worst_rel = 0.0
            worst_label = ""
            for fld in FIELDS:
                for m in METRICS:
                    col = f"{fld}_{m}"
                    a = float(gr[col])
                    b = float(jr[col])
                    ae = abs(a - b)
                    denom = max(abs(a), abs(b), 1e-30)
                    re = ae / denom
                    f.write(f"{step},{sid},{sname},{fld},{m},{a:.8e},{b:.8e},{ae:.4e},{re:.4e}\n")
                    if re > worst_rel:
                        worst_rel = re
                        worst_label = f"{col}(g={a:.3e},j={b:.3e})"
            flag = "***" if worst_rel > 0.01 else "   "
            print(f"{sid:>3} {sname:<14} {worst_label:>40} {worst_rel:>10.3e} {flag}")


def _oracle_summary(g, j):
    """Print per-step max relative error summary."""
    print(f"\n{'='*72}")
    print(f"  SUMMARY — per-step max relative error")
    print(f"{'='*72}")
    steps = sorted(set(idx[0] for idx in g.index))
    for step in steps[:10]:
        max_rel = 0.0
        for sid, _ in STAGES:
            if (step, sid) not in g.index:
                continue
            gr = g.loc[(step, sid)]
            jr = j.loc[(step, sid)]
            for fld in FIELDS:
                for m in METRICS:
                    col = f"{fld}_{m}"
                    a = float(gr[col]); b = float(jr[col])
                    denom = max(abs(a), abs(b), 1e-30)
                    rel = abs(a - b) / denom
                    max_rel = max(max_rel, rel)
        print(f"  step {step:>3}: max_rel = {max_rel:.4e}")


def _tensor_compare(step: int, sid: int):
    """Compare rank-0 3D tensors if binary files exist."""
    if not GSAM_RANK0.exists():
        print(f"WARNING: gSAM rank_000000.bin not found, skipping tensor comparison")
        return
    if not JSAM_RANK0.exists():
        print(f"WARNING: jsam rank_000000.bin not found, skipping tensor comparison")
        return

    print(f"\n--- Tensor comparison: step {step}, stage {sid} ---")
    print(f"  (Rank-0 binary loading not yet implemented — see TODO in driver)")
    print(f"  gSAM: {GSAM_RANK0}")
    print(f"  jsam: {JSAM_RANK0}")
    # TODO: Implement full tensor loading using the Oracle_Tensor_Structure.md
    # format description. For now, report existence only.


def main() -> int:
    case = sys.argv[1]
    g, j = _load_globals()

    if case == "oracle_globals_step1":
        _compare_step(g, j, 1, "step1_detail.csv")
    elif case == "oracle_globals_step2":
        _compare_step(g, j, 2, "step2_detail.csv")
    elif case == "oracle_tensor_step1_adamsA":
        _compare_step(g, j, 1, "step1_detail.csv")
        _tensor_compare(1, 10)
    elif case == "oracle_summary":
        _oracle_summary(g, j)
    else:
        raise ValueError(f"unknown case: {case}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
