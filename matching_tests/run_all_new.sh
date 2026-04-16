#!/bin/bash
# Run ALL matching tests in matching_tests/, collecting results.
# Does not stop on failure — reports everything.
#
# Each test_*/run.sh is run as a black box:
#   exit 0  → each PASS/FAIL line is parsed from stdout
#   exit 77 → SKIP (scaffold placeholder)
#   other   → ERROR
#
# Tests that use the standard run_case+run_compare pattern emit
# "PASS [label]" or "FAIL [label]" lines.  Tests using scaffold_run.sh
# emit "SKIP [label]".

export PYTHON="${PYTHON:-/glade/u/home/sabramian/.conda/envs/jsam/bin/python}"
export JAX_PLATFORMS=cpu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Auto-discover every test_* directory that has a run.sh
TESTS=()
for d in "$HERE"/test_*/run.sh; do
  TESTS+=("$(basename "$(dirname "$d")")")
done

# Sort alphabetically
IFS=$'\n' TESTS=($(sort <<<"${TESTS[*]}")); unset IFS

total_pass=0
total_fail=0
total_skip=0
total_error=0

# Collect per-test verdicts for the final table
declare -a VERDICTS=()

for t in "${TESTS[@]}"; do
  echo ""
  echo "============================================"
  echo "  $t"
  echo "============================================"

  # Run the test's own run.sh, capture output
  output=$(
    export PYTHON="$PYTHON"
    export JSAM_ROOT="$(cd "$HERE/.." && pwd)"
    cd "$HERE"
    bash "$HERE/$t/run.sh" 2>&1
  )
  rc=$?

  echo "$output"

  if [ $rc -eq 77 ]; then
    # Scaffold / not-yet-implemented
    total_skip=$((total_skip + 1))
    VERDICTS+=("SKIP    $t")
    continue
  fi

  # Count PASS / FAIL lines in the output
  p=$(echo "$output" | grep -c "^PASS " || true)
  f=$(echo "$output" | grep -c "^FAIL " || true)

  if [ $rc -ne 0 ] && [ $f -eq 0 ]; then
    # Non-zero exit but no FAIL lines → compile or runtime error
    total_error=$((total_error + 1))
    VERDICTS+=("ERROR   $t  (exit $rc)")
  elif [ $f -gt 0 ]; then
    total_pass=$((total_pass + p))
    total_fail=$((total_fail + f))
    VERDICTS+=("FAIL    $t  ($p pass, $f fail)")
  else
    total_pass=$((total_pass + p))
    VERDICTS+=("PASS    $t  ($p cases)")
  fi
done

echo ""
echo ""
echo "################################################################"
echo "#                    FULL SUMMARY TABLE                        #"
echo "################################################################"
echo ""
printf "%-8s %-38s %s\n" "Verdict" "Test" "Detail"
printf "%-8s %-38s %s\n" "-------" "----" "------"
for v in "${VERDICTS[@]}"; do
  verdict="${v%% *}"
  rest="${v#* }"
  tname="${rest%% *}"
  detail="${rest#* }"
  [ "$detail" = "$tname" ] && detail=""
  printf "%-8s %-38s %s\n" "$verdict" "$tname" "$detail"
done

echo ""
echo "============================================"
echo "  TOTALS"
echo "============================================"
echo "  PASS:  $total_pass cases"
echo "  FAIL:  $total_fail cases"
echo "  SKIP:  $total_skip tests"
echo "  ERROR: $total_error tests"
echo "============================================"
