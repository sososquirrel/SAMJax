"""
Detailed field-by-field diagnostics for ERA5 init comparison.

Reads gsam_{field}.bin and jsam_{field}.bin from the work directory
and prints per-field statistics: max abs error, max relative error,
RMSE, correlation, and identifies the vertical level(s) where
discrepancies are largest.

Usage
-----
    python diagnose_init_diff.py <workdir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import read_bin  # noqa: E402

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
# IRMA box dimensions from the debug dump header
NZM = 74
NJ_IRMA = 150   # j_max_gl - j_min_gl + 1 = 510 - 361 + 1
NI_IRMA = 320   # i_max_gl - i_min_gl + 1 = 1361 - 1042 + 1


def diagnose_field(name: str, gsam: np.ndarray, jsam: np.ndarray) -> dict:
    """Compute comparison metrics for one field."""
    diff = jsam - gsam
    abs_diff = np.abs(diff)

    # Denominator for relative error: max of absolute values, avoid /0
    denom = np.maximum(np.abs(gsam), np.abs(jsam))
    denom = np.where(denom < 1e-30, 1.0, denom)
    rel_diff = abs_diff / denom

    # Per-level statistics
    nz = gsam.shape[0]
    level_max_abs = np.array([abs_diff[k].max() for k in range(nz)])
    level_rmse = np.array([np.sqrt((diff[k] ** 2).mean()) for k in range(nz)])
    level_max_rel = np.array([rel_diff[k].max() for k in range(nz)])
    worst_level = int(np.argmax(level_max_abs))

    # Global correlation
    gsam_flat = gsam.ravel()
    jsam_flat = jsam.ravel()
    if gsam_flat.std() > 0 and jsam_flat.std() > 0:
        corr = float(np.corrcoef(gsam_flat, jsam_flat)[0, 1])
    else:
        corr = float("nan")

    return {
        "name": name,
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "correlation": corr,
        "gsam_range": (float(gsam.min()), float(gsam.max())),
        "jsam_range": (float(jsam.min()), float(jsam.max())),
        "worst_level": worst_level,
        "worst_level_max_abs": float(level_max_abs[worst_level]),
        "worst_level_rmse": float(level_rmse[worst_level]),
        "level_max_abs": level_max_abs,
        "level_max_rel": level_max_rel,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workdir>")
        return 1

    workdir = Path(sys.argv[1])

    print("=" * 80)
    print("ERA5 INIT TENSOR COMPARISON — DETAILED DIAGNOSTICS")
    print("=" * 80)
    print(f"Grid: {NZM} levels x {NJ_IRMA} lat x {NI_IRMA} lon (IRMA sub-region)")
    print()

    all_results = []
    for name in FIELDS:
        gsam_path = workdir / f"gsam_{name}.bin"
        jsam_path = workdir / f"jsam_{name}.bin"
        if not gsam_path.exists() or not jsam_path.exists():
            print(f"  [SKIP] {name}: missing bin file")
            continue

        gsam_flat = read_bin(gsam_path)
        jsam_flat = read_bin(jsam_path)

        expected_size = NZM * NJ_IRMA * NI_IRMA
        if gsam_flat.size != expected_size:
            print(f"  [WARN] {name}: gsam size={gsam_flat.size}, expected={expected_size}")
            # Try to infer shape
            nz = NZM
            total = gsam_flat.size
            nj_ni = total // nz
            print(f"         Inferring nj*ni={nj_ni}")

        gsam = gsam_flat.reshape(NZM, NJ_IRMA, NI_IRMA)
        jsam = jsam_flat.reshape(NZM, NJ_IRMA, NI_IRMA)

        r = diagnose_field(name, gsam, jsam)
        all_results.append(r)

        print(f"--- {name} ---")
        print(f"  gSAM range:   [{r['gsam_range'][0]:.6e}, {r['gsam_range'][1]:.6e}]")
        print(f"  jsam range:   [{r['jsam_range'][0]:.6e}, {r['jsam_range'][1]:.6e}]")
        print(f"  Max |diff|:   {r['max_abs']:.6e}")
        print(f"  Mean |diff|:  {r['mean_abs']:.6e}")
        print(f"  Max rel diff: {r['max_rel']:.6e}")
        print(f"  Mean rel:     {r['mean_rel']:.6e}")
        print(f"  RMSE:         {r['rmse']:.6e}")
        print(f"  Correlation:  {r['correlation']:.8f}")
        print(f"  Worst level:  k={r['worst_level']} "
              f"(max|diff|={r['worst_level_max_abs']:.6e}, "
              f"rmse={r['worst_level_rmse']:.6e})")
        # Show top-5 worst levels
        top5 = np.argsort(r["level_max_abs"])[::-1][:5]
        print(f"  Top-5 levels: {', '.join(f'k={k}(err={r[\"level_max_abs\"][k]:.3e})' for k in top5)}")
        print()

    # Summary table
    if all_results:
        print("=" * 80)
        print("SUMMARY")
        print(f"{'Field':<6} {'Max|diff|':>12} {'RMSE':>12} {'MaxRel':>12} {'Corr':>10}")
        print("-" * 56)
        for r in all_results:
            print(f"{r['name']:<6} {r['max_abs']:>12.4e} {r['rmse']:>12.4e} "
                  f"{r['max_rel']:>12.4e} {r['correlation']:>10.6f}")
        print("=" * 80)

        # Verdict
        worst_rel = max(r["max_rel"] for r in all_results)
        worst_corr = min(r["correlation"] for r in all_results
                         if not np.isnan(r["correlation"]))
        print()
        if worst_corr < 0.99:
            print("VERDICT: SIGNIFICANT structural mismatch detected "
                  f"(worst correlation={worst_corr:.4f})")
            print("  -> This indicates a likely bug in the init pipeline.")
        elif worst_rel > 0.5:
            print(f"VERDICT: Large relative differences (max={worst_rel:.2e}) "
                  "but fields are correlated.")
            print("  -> Likely interpolation chain difference, not a structural bug.")
            print("  -> Investigate per-level breakdown above for root cause.")
        elif worst_rel > 0.05:
            print(f"VERDICT: Moderate differences (max_rel={worst_rel:.2e}). "
                  "Expected from different interpolation pipelines.")
        else:
            print(f"VERDICT: Good agreement (max_rel={worst_rel:.2e}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
