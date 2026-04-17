# gSAM IRMA 500-step debug dump — schema

Oracle tensors for bit-level comparison against jsam. Produced on 2026-04-15 by:

- `/glade/u/home/sabramian/gSAM1.8.7/SRC/debug_dump.f90`       (dump module)
- `/glade/u/home/sabramian/gSAM1.8.7/SRC/main.f90`             (calls `dump_stage` at each of the 19 stages)
- `/glade/u/home/sabramian/gSAM1.8.7/run_IRMA_debug500.pbs`    (job script)
- `/glade/u/home/sabramian/gSAM1.8.7/CASES/IRMA/prm_debug500`  (namelist)
- `/glade/u/home/sabramian/gSAM1.8.7/gSAM`                      (binary, rebuilt 2026-04-15)

## Paths

```
/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/globals.csv
    rank-0 global min/max/mean per (step,stage), 9538 rows + header

/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/rank_NNNNNN.bin
    per-rank stream-binary, 144 files, rank_000000.bin .. rank_000143.bin

/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/logs/run.log
    PBS stdout (log has 3 runs appended; last run begins near line 1320)

/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/OUT_2DL/IRMA_diag1h_2017-09-05-01-23-20_0000000500.2D_lnd.nc
    SLM 2D fields, 249 MB, converted via /glade/u/home/sabramian/gSAM1.8.7/UTIL/2D2nc

/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/OUT_STAT/IRMA_diag1h.nc
    column statistics (hbuffer .stat converted via /glade/u/home/sabramian/gSAM1.8.7/UTIL/stat2nc)
```

## Run config

- Grid: `nx_gl=1440, ny_gl=720, nzm=74`, lat-lon (lat_720_dyvar), dt=10 s
- Decomposition: 16×9 = 144 MPI ranks, local `nx=90, ny=80`
- 500 steps. Step 1 subcycles (3 icycles with dt≈4.81 s), steps 2..500 use dt=10 s
- Precision: all field data is **float32** (matches gSAM internal precision)

## Stages (19 per step, in call order)

```
id  name              id  name              id  name
 0  pre_step           7  coriolis          14  advect_scalars
 1  forcing            8  sgs_proc          15  sgs_scalars
 2  nudging            9  sgs_mom           16  upperbound
 3  buoyancy          10  adamsA            17  micro
 4  radiation         11  damping           18  diagnose
 5  surface           12  adamsB
 6  advect_mom        13  pressure
```

Total records = 500 steps × 19 stages + 3 extra icycle-1 dumps = **9538** rows/records.

## Fields (7, in this fixed order)

Index | Name | Source array | Notes
------|------|--------------|------
1 | U    | `u(1:nx,1:ny,1:nzm)` | staggered, dumped at raw indices
2 | V    | `v(1:nx,1:ny,1:nzm)` | staggered, dumped at raw indices
3 | W    | `w(1:nx,1:ny,1:nzm)` | staggered, top half-level dropped
4 | TABS | recomputed locally  | `t - gamaz(k) + fac_cond*(qcl+qpl) + fac_sub*(qci+qpi)`
5 | QC   | `qcl(1:nx,1:ny,1:nzm)` | **stale between micro calls** (only refreshed in `micro_proc`/`diagnose`)
6 | QV   | `qv(1:nx,1:ny,1:nzm)`  | stale between micro calls
7 | QI   | `qci(1:nx,1:ny,1:nzm)` | stale between micro calls

Note: QV/QC/QI only change at stages 17 (micro) and 18 (diagnose) within a step —
all earlier stages show the previous step's post-diagnose values.

## `globals.csv` (rank-0 only, 9538 rows + header)

25 columns, per-row reduction over the **full global domain** (not IRMA box):
```
nstep, stage_id, stage_name, dtn,
U_min, U_max, U_mean, V_min, V_max, V_mean, W_min, W_max, W_mean,
TABS_min, TABS_max, TABS_mean,
QC_min, QC_max, QC_mean, QV_min, QV_max, QV_mean, QI_min, QI_max, QI_mean
```
Means are unweighted volume averages: `sum / (nx_gl*ny_gl*nzm)`.

## `rank_NNNNNN.bin` — stream-binary, little-endian

Each rank writes its own file. Data only covers the **IRMA subregion**:
```
lat ∈ [  0,  35] deg
lon ∈ [260, 340] deg   (i.e. 100W..20W)
```

A rank writes per-step data **only if its local subdomain overlaps the IRMA box**.
Header is written by all 144 ranks; per-step records only by overlapping ranks.

### Header (written once, at `dump_init`)

Fortran `access='stream' form='unformatted'`. Sequence of raw little-endian bytes,
no record markers:

```
i4  nx_gl           # 1440
i4  ny_gl           #  720
i4  nzm             #   74
r4  lat_min         #    0.0
r4  lat_max         #   35.0
r4  lon_min         #  260.0
r4  lon_max         #  340.0
i4  i_min_gl, i_max_gl, j_min_gl, j_max_gl   # 1-based global indices of box (inclusive)
i4  it_off, jt_off                           # this rank's (it,jt) subdomain offset
i4  i1_loc, i2_loc, j1_loc, j2_loc           # 1-based LOCAL indices of box overlap
                                             #   invalid (no overlap) iff i1>i2 or j1>j2
i4  nfields                                  # 7
char[8] * 7   field_names                    # ('U       ','V       ','W       ',
                                             #  'TABS    ','QC      ','QV      ','QI      ')
```

### Per-dump record (only if `rank_has_overlap`)

Written once per `dump_stage` call:
```
i4    nstep
i4    stage_id
r4[ni, nj, nzm, 7]   out_box         # Fortran order: fastest is i, then j, then k, then field
                                     # ni = i2_loc - i1_loc + 1
                                     # nj = j2_loc - j1_loc + 1
```

Field slice order inside `out_box`: `(U, V, W, TABS, QC, QV, QI)`.