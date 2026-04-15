#!/bin/bash
# scaffold_run.sh — placeholder for unimplemented matching tests.
#
# Exits 77 so run_all.sh counts this test as SKIP (not FAIL).
# Replace this script (by updating <module>/run.sh) once driver.f90 and
# dump_inputs.py are written for the module.

mod="$(basename "$(dirname "$0")")"
echo "SKIP [${mod}]: driver not yet implemented — see ${mod}/TODO.md"
exit 77
