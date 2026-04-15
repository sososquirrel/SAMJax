#!/bin/bash -l
#
# jsam IRMA 500-step debug run — mirror of gSAM run_IRMA_debug500.pbs
# ---------------------------------------------------------------------
# Purpose: reproduce the exact same configuration as the Fortran gSAM
# oracle dump at /glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG
# so jsam tensors can be compared stage-by-stage, step-by-step, to the
# last bit.
#
# Matches these gSAM prm_debug500 settings:
#   - Grid: lat_720_dyvar native (1440 x 720 x 74), dt=10s, 500 steps
#   - RRTMG radiation, nrad=90 (gSAM uses LW+SW; jsam has LW only)
#   - o3file: ozone_era5_monthly_201709-201709_GLOBAL.bin
#   - docloud, doprecip, dosurface, dosgs, docoriolis = true
#   - dolargescale = false, donudging_uv/tq = false
#   - damping_u_cu=0.25, damping_w_cu=0.3
#   - doupperbound = false
#   - Precision: float32 (gSAM internal precision, matches oracle dumps)
#
# The oracle tensors live at:
#   /glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/
# See .claude/Oracle_Tensor_Structure.md for the on-disk schema.

set -euo pipefail

SAMJAX=/glade/u/home/sabramian/SAMJax
OUTDIR=/glade/derecho/scratch/sabramian/jsam_IRMA_500timesteps_DEBUG
DEBUGDIR=$OUTDIR/debug
O3FILE=/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D/ozone_era5_monthly_201709-201709_GLOBAL.bin

mkdir -p "$OUTDIR" "$DEBUGDIR"

cd "$SAMJAX"

python scripts/run_irma.py \
    --nsteps          500  \
    --output-interval 500  \
    --out-dir         "$OUTDIR" \
    --casename        jsam_IRMA_debug500 \
    --rad             rrtmg \
    --nrad            90 \
    --co2             400.0 \
    --o3file          "$O3FILE" \
    --debug-dump-dir  "$DEBUGDIR" \
    --float32 \
    --no-polar-filter \
    --sponge-tau      0 \
    --slm \
    "$@"
