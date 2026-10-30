#!/bin/bash
# 把 51 个 Alpha101 因子分成每批 6 个跑，避免内存不够
cd "$(dirname "$0")"
keys=$(python -c "import sys; sys.path.insert(0,'..'); import alpha101; print(' '.join(alpha101.ALPHAS))")
batch=()
for k in $keys; do
  batch+=("alpha101:$k")
  if [ ${#batch[@]} -eq 6 ]; then python -W ignore run_round.py 4 "${batch[@]}" || exit 1; batch=(); fi
done
[ ${#batch[@]} -gt 0 ] && python -W ignore run_round.py 4 "${batch[@]}"
echo ALL_DONE
