# matching_tests

Per-module behavior comparison between **gSAM** (`/glade/u/home/sabramian/gSAM1.8.7`)
and **jsam** (the JAX port at `/glade/u/home/sabramian/SAMJax/jsam`).
Each subdirectory tests one module in isolation and asserts agreement to
**4 decimal places** (`atol=5e-5`, `rtol=1e-4` by default — override per
test).

No pytest. Each module is a self-contained bash script that:

1. Compiles a tiny Fortran driver against the relevant gSAM `.f90` files.
2. Runs the driver — it dumps deterministic outputs to `work/fortran_out.bin`.
3. Runs `dump_inputs.py` (or `test_jsam.py`) — same numbers from the jsam side.
4. Calls `python -m common.compare` to diff them at 4 decimals.
