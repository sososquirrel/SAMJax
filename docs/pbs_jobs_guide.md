# PBS Job Submission on Derecho and Casper

Official NCAR docs:
- [PBS overview](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/)
- [PBS job scripts — directives reference](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/)
- [Casper job script examples](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/casper-job-script-examples/)
- [Starting Casper jobs](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/starting-casper-jobs/)
- [Casper overview](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/)
- [Derecho overview](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)
- [Queues and charging](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/charging/)

---

## Which system to use?

| Situation | Use |
|-----------|-----|
| GPU compute (PyTorch, JAX, CUDA) | **Derecho** `main` or **Casper** `casper@casper-pbs` |
| CPU-only data analysis (xarray, dask) | **Casper** — large-memory nodes available |
| MPI jobs, production numerical models | **Derecho** — `main` queue |
| Interactive GPU debugging | **Casper** — `casper@casper-pbs` with `qsub -I` |
| Interactive CPU debugging | **Derecho** — `develop` queue |
| Large-memory preprocessing (>100 GB RAM) | **Casper** — up to ~700 GB per node |

**Key rule of thumb:**
- GPU inference/training → Derecho `main` (A100s available there too) or Casper.
- Large-memory CPU analysis (CMIP6, xarray/dask) → Casper.
- MPI numerical models → Derecho.

### Hardware summary

| System | CPUs/node | RAM/node | GPUs/node |
|--------|-----------|----------|-----------|
| Derecho | 128 (AMD EPYC 7763) | 256 GB | up to 4× A100 (GPU nodes) |
| Casper | 36–64 | 256–700 GB | up to 4× A100 or V100 |

---

## Key concepts: nodes, chunks, tasks, CPUs

PBS uses a **chunk-based** resource model. The `-l select=` line defines how many chunks to allocate and what each chunk contains.

```
#PBS -l select=<N>:ncpus=<C>:mpiprocs=<M>:mem=<RAM>:ngpus=<G>:gpu_type=<T>
```

| Term | What it means |
|------|---------------|
| `select=N` | Number of **chunks** (roughly: number of nodes, or node-like allocations) |
| `ncpus=C` | CPU cores per chunk |
| `mpiprocs=M` | MPI ranks launched per chunk (must be ≤ `ncpus`) |
| `mem=RAM` | RAM per chunk (e.g. `64GB`) |
| `ngpus=G` | GPUs per chunk |
| `ompthreads=T` | OpenMP threads per MPI rank (optional; `ncpus` = `mpiprocs` × `ompthreads`) |

**Total MPI ranks** = `select` × `mpiprocs`
**Total CPUs allocated** = `select` × `ncpus`

### Example breakdowns

| `select=` line | Meaning |
|----------------|---------|
| `select=1:ncpus=8:mem=64GB:ngpus=1:gpu_type=a100` | 1 node, 8 CPUs, 64 GB, 1 GPU — single-GPU inference |
| `select=1:ncpus=16:mem=128GB` | 1 node, 16 CPUs, 128 GB — large-memory CPU analysis |
| `select=4:ncpus=128:mpiprocs=128` | 4 Derecho nodes, 512 MPI ranks total |
| `select=2:ncpus=128:mpiprocs=16:ompthreads=8` | 2 nodes, 32 MPI ranks, 8 threads each (hybrid MPI+OpenMP) |

---

## How many nodes to request?

### For GPU jobs
- Most single-model inference jobs need **1 chunk, 1 GPU**. The framework (PyTorch, JAX) manages parallelism within one GPU.
- Multi-GPU: request `ngpus=N` up to 4 per node. Each Casper A100 node: 4 GPUs, 64 CPUs, ~700 GB RAM.
- Start small, scale up only if you hit memory limits or runtime is too long.

### For CPU analysis jobs (CMIP6, xarray/dask)
- Size by **memory first**: estimate your dataset size in RAM (e.g. CESM2 daily `tas` for one SSP ~10–50 GB), then pick `mem=` accordingly.
- CPUs control dask parallelism: `ncpus=8–16` is usually enough for xarray/dask workloads. More CPUs help if you have many `.compute()` calls or heavy map operations.
- One Casper node is almost always sufficient for CMIP analysis.

### For MPI jobs
- `nodes = ceil(total_mpi_ranks / 128)` on Derecho (128 CPUs/node).
- Don't over-request — more nodes = longer queue wait and more communication overhead.

### Rules of thumb
1. **Single Python/GPU script** → `select=1:ncpus=8:ngpus=1`
2. **Memory-bound analysis** → size `mem=` first, then choose CPUs.
3. **MPI job** → match ranks to domain decomposition, divide by 128 for node count.
4. **Always test in `develop` queue** with 1 node before scaling.

---

## GPU job on Derecho (Aurora inference)

The Aurora inference job runs on Derecho's A100 GPU nodes, queue `main`:

```bash
#!/bin/bash
#PBS -N aurora_inference
#PBS -A UCLB0065
#PBS -q main
#PBS -l select=1:ncpus=8:mem=64GB:ngpus=1:gpu_type=a100
#PBS -l walltime=04:00:00
#PBS -o logs/aurora_inference.out
#PBS -e logs/aurora_inference.err

set -euo pipefail

PROJ=/glade/u/home/sabramian/aurora-era5
cd "$PROJ"

# Conda must be initialised explicitly — PBS does not source ~/.bashrc
__conda_setup="$('/glade/u/apps/jupyterhub/jh-23.11/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate aurora-era5

DATE=${DATE:-2017-09-05}
STEPS=${STEPS:-20}
TC=${TC:-irma}
SCRATCH=/glade/derecho/scratch/sabramian/aurora-era5
export HF_HOME=$SCRATCH/hf_cache

mkdir -p "$SCRATCH/data" "$SCRATCH/forecasts/$TC" logs/

python -u -m aurora_era5.data_prep --date "$DATE" --outdir "$SCRATCH/data"
python -u -m aurora_era5.data_prep --static --outdir "$SCRATCH/data"
python -u -m aurora_era5.inference \
    --surface "$SCRATCH/data/era5_surface_${DATE}.nc" \
    --atmos   "$SCRATCH/data/era5_atmos_${DATE}.nc" \
    --steps   "$STEPS" \
    --outdir  "$SCRATCH/forecasts/$TC" \
    --device  cuda
```

Submit with environment variable overrides:
```bash
qsub scripts/run_inference.pbs                          # IRMA defaults
DATE=2017-08-25 STEPS=40 TC=harvey qsub scripts/run_inference.pbs  # Harvey, 10 days
```

See [aurora-era5/scripts/run_inference.pbs](../../aurora-era5/scripts/run_inference.pbs) for the full working script.

### GPU type options

| `gpu_type=` | Memory | Notes |
|-------------|--------|-------|
| `a100` | 40 GB | Standard; works for Aurora inference |
| `a100_80gb` | 80 GB | For larger batches or longer rollouts |
| `v100` | 32 GB | Older; available as fallback on Casper |

---

## CPU job on Casper (CMIP6 / xarray analysis)

CMIP6 analysis is CPU and memory intensive (large xarray datasets, dask compute). Use Casper large-memory nodes:

```bash
#!/bin/bash
#PBS -N cmip6_analysis
#PBS -A UCLB0065
#PBS -q casper@casper-pbs
#PBS -l select=1:ncpus=16:mem=128GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/cmip6_analysis.log

set -euo pipefail

cd /glade/u/home/sabramian/<your-cmip-repo>

__conda_setup="$('/glade/u/apps/jupyterhub/jh-23.11/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate <your-env>

python -u analysis/run_cmip_analysis.py
```

**Memory sizing for CMIP6:** a single CESM2 daily variable (`tas`, 192×288 grid) over 85 years (ssp585, 2015–2100) is ~6 GB in memory per variable. For 7 variables across 3 SSPs: ~130 GB. Request `mem=128GB` to `mem=200GB` depending on how much you load at once.

**Dask workers:** set `ncpus` to match the number of dask workers you want. A good default for CMIP analysis is `ncpus=8` to `ncpus=16`.

The CESM2 CMIP6 catalog lives at:
```
/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json
```

---

## Derecho queue options

| Queue | Max walltime | Max CPUs | Notes |
|-------|-------------|----------|-------|
| `main` | 12 h | unlimited | Standard production |
| `preempt` | 24 h | unlimited | Cheap but can be killed |
| `develop` | 1 h | 256 CPUs | Interactive/debugging |

## Casper queue

All Casper jobs go to `casper@casper-pbs`. There is one queue; walltime max is 24 h.

---

## Project codes

| Code | Project | Used for |
|------|---------|----------|
| `UCLB0065` | Aurora / CMIP | Aurora GPU inference (Derecho), CMIP CPU analysis (Casper) |
| `UCLB0066` | SAMJax | SAMJax GPU runs (Casper) |
| `UMIC0004` | SAM / gSAM | Derecho CPU runs |

---

## Submitting and monitoring

```bash
# Submit
qsub my_job.pbs

# Check status (all your jobs)
qstat -u $USER

# Detailed info on a specific job
qstat -f <jobid>

# Delete a job
qdel <jobid>

# Watch log in real time
tail -f logs/aurora_inference.out
```

---

## Common pitfalls

- **Conda activation** — PBS does not source `~/.bashrc`, so `conda activate` won't work unless you first initialise the conda hook (see the `__conda_setup` block in the examples above). Alternatively, use the full path to the env's Python: `/glade/u/home/sabramian/.conda/envs/<env>/bin/python`.
- **Do not add `module load`** in Casper scripts — `ncarenv/25.10` is sticky.
- **Scratch is fast, home is slow** — write output to `/glade/derecho/scratch/sabramian/`, not home.
- **HuggingFace cache** — set `HF_HOME` to scratch so model weights aren't re-downloaded each run.
- **CMIP `decode_times`** — always use `decode_times=False` + per-file `preprocess=xr.decode_cf` with `open_mfdataset`; direct CF decode across CESM2 files fails due to inconsistent time units.
- **`-j oe`** merges stdout/stderr into one log — easier to tail. Or use separate `-o` / `-e` to split them.

---

## Interactive sessions

Interactive sessions give you a shell on a compute node — useful for debugging, profiling, or iterative development.

### Casper — interactive GPU session

```bash
qsub -I -A UCLB0065 -q casper@casper-pbs \
     -l select=1:ncpus=4:mem=32GB:ngpus=1:gpu_type=a100 \
     -l walltime=01:00:00
```

Or use the NCAR shortcut (Casper only):

```bash
qinteractive -A UCLB0065 -q casper@casper-pbs \
             -l select=1:ncpus=4:mem=32GB:ngpus=1:gpu_type=a100 \
             -l walltime=01:00:00
```

`qinteractive` with no arguments gives 1 CPU, 10 GB RAM on Casper as a quick scratch session.

### Derecho — interactive CPU session

```bash
qsub -I -A UCLB0065 -q develop \
     -l select=1:ncpus=8:mem=64GB \
     -l walltime=01:00:00
```

### Tips

- Keep walltime short (1–2 h) — longer requests wait longer in the queue.
- Activate your conda env explicitly once on the node: `conda activate aurora-era5`
- Type `exit` when done — releases the node immediately instead of waiting for walltime.

---

## Notebooks in VS Code via Remote SSH (login node)

This is the simplest setup: VS Code connects to the login node over SSH and runs the notebook kernel there. Good for light development, data exploration, and CMIP catalog browsing. **Do not run heavy compute on the login node** — submit a batch job or start an interactive session for that.

### One-time setup

1. **Install extensions** in VS Code (local machine):
   - [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
   - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

2. **Add SSH config** on your laptop (`~/.ssh/config`):
   ```
   Host casper
       HostName casper.hpc.ucar.edu
       User sabramian
       ForwardAgent yes

   Host derecho
       HostName derecho.hpc.ucar.edu
       User sabramian
       ForwardAgent yes
   ```

3. **Connect**: Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → `Remote-SSH: Connect to Host` → select `casper` or `derecho`. VS Code installs a small server on the remote automatically.

4. **Open your repo**: `File → Open Folder` → navigate to e.g. `/glade/u/home/sabramian/aurora-era5`.

### Running a notebook

1. Open any `.ipynb` file in VS Code.
2. Click **Select Kernel** (top right) → **Python Environments** → pick your conda env (e.g. `aurora-era5`).
   - If the env doesn't appear, choose **Enter interpreter path** and paste the full path:
     ```
     /glade/u/home/sabramian/.conda/envs/aurora-era5/bin/python
     ```
3. Run cells normally. The kernel runs on the login node.

### Login node vs compute node

| | Login node (VS Code SSH) | Compute node (interactive PBS) |
|-|--------------------------|-------------------------------|
| GPU access | No | Yes (Casper/Derecho with `ngpus=1`) |
| RAM | ~a few GB shared | Up to 256 GB (Derecho) / 700 GB (Casper) |
| CPU time | Short tasks only | Full walltime allocation |
| Good for | Editing, CMIP catalog queries, small plots | Aurora inference, large dask `.compute()` |

For GPU-backed notebook work (e.g. running Aurora steps interactively), start an interactive Casper session first, then connect VS Code to the **compute node hostname** using the same Remote-SSH steps — the compute node will appear as a new SSH target once you're logged in.
