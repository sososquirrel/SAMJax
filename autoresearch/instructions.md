# Auto-Research Pipeline — jsam Continuous Matching

## Goal

Iterate until jsam output tensors match gSAM oracle tensors (see `Oracle_Tensor_Structure.md`) to get max deviation below 1e-4

## Pipeline (each iteration)

### Step 1 — Read context
- Read `autoresearch/design.md` (repo map)
- Read `autoresearch/instructions.md` (this file)
- Read `.claude/Oracle_Tensor_Structure.md` (tensor schema)
- Read `autoresearch/progress.md` (experiment log)

### Step 2 — Analyze & fix
- From `progress.md`, identify the **first diverging stage** (earliest stage with mismatch).
- Read the relevant jsam source and its gSAM Fortran counterpart (see design.md mapping).
- Identify the root cause. Fix one or a few related things per iteration — do not shotgun.
- Edit jsam code. Do not touch gSAM.

### Step 3 — Run
- Submit: `qsub /glade/u/home/sabramian/SAMJax/scripts/run_irma_debug500.pbs`
- Wait for job to complete (check `qstat -u sabramian` or watch the log file).
- jsam output lands in `/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug/`
- Oracle is at `/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/`

### Step 4 — Compare
- Run: `python autoresearch/compare.py` (defaults point to both debug dirs)
- Use `--max-rows 38` to limit to step 1 only (19 stages × 2 icycles) for fast iteration.
- Use `--threshold 1e-5` (default) or tighten to `1e-7` for near-ULP matching.
- Script exits 0 if all match, 1 if mismatch — prints first diverging stage.
- If IRMA-box binary comparison is needed, reassemble rank files per `Oracle_Tensor_Structure.md`.
- Focus on the **first stage that diverges** — upstream errors cascade.

### Step 5 — Update progress.md
Append an entry to `progress.md` with:
```
## Iteration N — YYYY-MM-DD
**Changed**: [files changed, 1-line summary]
**Result**: [which stages match, which diverge, error magnitudes]
**Root cause**: [what was wrong]
**Next**: [what to investigate next]
```

### Step 6 — Loop
Go to Step 1.

## Rules
- Fix the **earliest diverging stage first**. Downstream mismatches may resolve automatically.
- Keep edits minimal and targeted. One logical fix per iteration.
- Always check if jsam produces a `globals.csv` — if not, check for crashes/errors in the log first.
- Stage 12 (adamsB) W spike is intentional in gSAM — don't flag it.
- QC/QV/QI are stale before stage 17 — only compare at stages 17-18.

## Key Paths
| What | Path |
|------|------|
| jsam code | `/glade/u/home/sabramian/SAMJax/jsam/` |
| gSAM code | `/glade/u/home/sabramian/gSAM1.8.7/SRC/` |
| Run script | `/glade/u/home/sabramian/SAMJax/scripts/run_irma_debug500.pbs` |
| jsam debug output | `/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug/` |
| Oracle debug output | `/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/` |
| Oracle schema | `/glade/u/home/sabramian/SAMJax/.claude/Oracle_Tensor_Structure.md` |
