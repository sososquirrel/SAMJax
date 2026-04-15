#!/bin/bash
# matching_tests top-level runner.
#
# Iterates every module subdir, runs its run.sh, prints a one-line status,
# and at the end prints a pass/skip/fail summary table. Exit code is the
# number of FAILs (0 = all pass).
#
# Usage:
#   ./run_all.sh                # run all modules
#   ./run_all.sh sat consts     # run only listed modules
#
# Skipped modules (driver.f90 not yet written) exit 77 — counted separately.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ $# -gt 0 ]]; then
  MODS=("$@")
else
  MODS=()
  for d in */; do
    [[ "$d" == "common/" || "$d" == "logs/" ]] && continue
    [[ ! -x "${d}run.sh" ]] && continue
    MODS+=("${d%/}")
  done
fi

PASS=()
SKIP=()
FAIL=()
LOGDIR="${HERE}/logs"
mkdir -p "$LOGDIR"

for mod in "${MODS[@]}"; do
  if [[ ! -x "${HERE}/${mod}/run.sh" ]]; then
    echo "[$mod] no run.sh — skipping"
    SKIP+=("$mod")
    continue
  fi
  log="${LOGDIR}/${mod}.log"
  printf "[%-14s] running... " "$mod"
  bash "${HERE}/${mod}/run.sh" >"$log" 2>&1
  rc=$?
  case $rc in
    0)  echo "PASS"; PASS+=("$mod") ;;
    77) echo "SKIP (see ${mod}/TODO.md)"; SKIP+=("$mod") ;;
    *)  echo "FAIL (rc=$rc, see logs/${mod}.log)"; FAIL+=("$mod") ;;
  esac
done

echo
echo "================ matching_tests summary ================"
echo "PASS: ${#PASS[@]}    SKIP: ${#SKIP[@]}    FAIL: ${#FAIL[@]}"
[[ ${#PASS[@]} -gt 0 ]] && echo "  pass: ${PASS[*]:-}"
[[ ${#SKIP[@]} -gt 0 ]] && echo "  skip: ${SKIP[*]:-}"
[[ ${#FAIL[@]} -gt 0 ]] && echo "  fail: ${FAIL[*]:-}"
exit ${#FAIL[@]}
