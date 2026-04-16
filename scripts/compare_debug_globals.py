#!/usr/bin/env python
"""Compare jsam vs gSAM oracle debug globals.csv stage-by-stage."""
import sys
import numpy as np
import pandas as pd

GSAM = "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv"
JSAM = "/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug/globals.csv"

g = pd.read_csv(GSAM)
j = pd.read_csv(JSAM)
g.columns = [c.strip() for c in g.columns]
j.columns = [c.strip() for c in j.columns]

# gSAM has 3 extra icycle-1 dumps for stages 6,10,12 at step 1 (per the md).
# Keep the LAST occurrence (final icycle) so nstep/stage_id is unique.
g = g.drop_duplicates(subset=["nstep", "stage_id"], keep="last")
j = j.drop_duplicates(subset=["nstep", "stage_id"], keep="last")
g = g.set_index(["nstep", "stage_id"])
j = j.set_index(["nstep", "stage_id"])
common = g.index.intersection(j.index)
g = g.loc[common].sort_index()
j = j.loc[common].sort_index()

print(f"common records: {len(common)}  (jsam has {len(j)}, gsam has {len(common)})")
print(f"step range: {common.get_level_values(0).min()}..{common.get_level_values(0).max()}\n")

fields = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
metrics = ["min", "max", "mean"]

# Walk records in order; report first-divergence stats per stage.
rows = []
for (step, sid) in common:
    gr = g.loc[(step, sid)]
    jr = j.loc[(step, sid)]
    stage = gr["stage_name"].strip() if isinstance(gr["stage_name"], str) else str(gr["stage_name"]).strip()
    rec = {"step": step, "sid": sid, "stage": stage}
    max_rel = 0.0
    worst = ""
    for f in fields:
        for m in metrics:
            col = f"{f}_{m}"
            a = float(gr[col]); b = float(jr[col])
            denom = max(abs(a), abs(b), 1e-30)
            rel = abs(a - b) / denom
            if rel > max_rel:
                max_rel = rel
                worst = f"{col}(g={a:.4e},j={b:.4e})"
            rec[col] = rel
    rec["max_rel"] = max_rel
    rec["worst"] = worst
    rows.append(rec)

df = pd.DataFrame(rows)

# Summary: per-step max diff, showing where divergence first crosses thresholds
print("=== Per-step summary (max relative diff across all 21 global reductions) ===")
per_step = df.groupby("step")["max_rel"].max()
print(per_step.to_string())

# First stage with any noticeable divergence
print("\n=== First record above each threshold ===")
for thr in [1e-6, 1e-4, 1e-2, 1e-1, 0.5]:
    hit = df[df["max_rel"] > thr]
    if len(hit):
        r = hit.iloc[0]
        print(f"  >{thr:<8g}  step={r['step']:>3} sid={r['sid']:>2} {r['stage']:<14} "
              f"max_rel={r['max_rel']:.3e}  {r['worst']}")
    else:
        print(f"  >{thr:<8g}  (not reached)")

# Show stage-0 (pre_step) state at step 1 — initial condition match
print("\n=== Step 1 stage 0 (pre_step) ===")
r = df[(df["step"] == 1) & (df["sid"] == 0)].iloc[0]
for f in fields:
    print(f"  {f:5}  min rel={r[f+'_min']:.2e}  max rel={r[f+'_max']:.2e}  mean rel={r[f+'_mean']:.2e}")

# Track W_max through first 5 steps (most diagnostic)
print("\n=== W_max and TABS_min trajectory (all stages, first 5 steps) ===")
print(f"{'step':>4} {'sid':>3} {'stage':<14} {'gW_max':>12} {'jW_max':>12} "
      f"{'gT_min':>10} {'jT_min':>10}")
for (step, sid) in common:
    if step > 5:
        break
    gr = g.loc[(step, sid)]
    jr = j.loc[(step, sid)]
    stage = str(gr["stage_name"]).strip()
    print(f"{step:>4} {sid:>3} {stage:<14} "
          f"{float(gr['W_max']):>12.4e} {float(jr['W_max']):>12.4e} "
          f"{float(gr['TABS_min']):>10.2f} {float(jr['TABS_min']):>10.2f}")
