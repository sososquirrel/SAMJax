# Key Findings — gSAM Behaviour

## Newton satadj at init time (2026-04-18)
The ERA5 binary init file holds raw ERA5 fields written *before* gSAM's `micro_init()` runs (`setdata.f90:349` reads, `setdata.f90:500` calls `micro_init`). Both gSAM and jSAM must run Newton satadj at init to convert ERA5's `(QV, QC, QI)` into SAM1MOM's self-consistent thermodynamic state where `q = QV+QC+QI` is the prognostic. This is not redundant — it is required.
