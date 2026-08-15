#!/usr/bin/env bash
# The pre-push gate for the C++ half. CI proves only the USD-free wire on a bare
# runner; everything needing USD, clang-tidy's compile database, a GPU, or a
# display is proven here. Run from anywhere; stops at the first failure.
#
#   scripts/hydra-check.sh
#
# Needs: build/hydra configured (see hydra/README.md), the USD prefix on
# PATH/PYTHONPATH, clang-format/clang-tidy 22.1.8, a built cenote-server, and a
# display for the interactive test.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== build, install, ctest =="
cmake --build build/hydra --parallel
cmake --install build/hydra
ctest --test-dir build/hydra --output-on-failure

echo "== clang-format =="
find hydra \( -name '*.cpp' -o -name '*.hpp' \) -print0 |
  xargs -0r clang-format --dry-run -Werror

echo "== clang-tidy =="
find hydra -name '*.cpp' -print0 |
  xargs -0 -P"$(nproc)" -n1 clang-tidy -p build/hydra --quiet

for test in usdrecord_smoke render_settings_test curves_test flip_golden interactive_test; do
  echo "== $test =="
  python3 "hydra/tests/$test.py"
done
