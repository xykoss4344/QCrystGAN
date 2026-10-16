#!/bin/sh
# Benchmark each run (n=300) as soon as its target checkpoint exists, one at a time,
# Wave 4 fine-tunes, n=300, as each reaches ep550.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
items="mgmno_stable:s4_relax10_ft_s1:550 mgmno_stable:s4_ms3x_ft_s1:550 mgmno_stable:s4_symdata_ft_s1:550"
while [ -n "$items" ]; do
  left=""
  for it in $items; do
    chem=${it%%:*}; rest=${it#*:}; run=${rest%%:*}; ep=${rest#*:}
    if [ -f runs/$run/checkpoint_$ep.pt ]; then
      echo "$(date +%m-%d_%H:%M) start $run ep$ep"
      QGAN_CHEM=$chem $P benchmark.py --ckpt runs/$run/checkpoint_$ep.pt --n 300 --cif_dir bench300 --cif_only 2>&1 | tail -1
      mv bench300/${run}_$ep bench300/${run}_${ep}_n300
      sh bench_full.sh bench300/${run}_${ep}_n300 || echo "FAILED $run"
      QGAN_CHEM=mgmno $P chgnet_screen.py bench300/${run}_${ep}_n300_relaxed --k 100 2>&1 | tail -1
      sh lemat_only.sh bench300/${run}_${ep}_n300_screen100 || echo "FAILED screen $run"
    else left="$left $it"; fi
  done
  items=$(echo $left); [ -n "$items" ] && sleep 300
done
echo "$(date +%m-%d_%H:%M) queue done"
