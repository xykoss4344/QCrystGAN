#!/bin/sh
# Benchmark each run (n=300) as soon as its target checkpoint exists, one at a time,
# for the oxide-pretrained fine-tune. Items: chem run epoch.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
until grep -q "queue done" bench_queue2.log && grep -q "queue done" bench_queue4.log; do sleep 120; done
items="mgmno_stable:p2_pretrain_ft_stable_s1:190"
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
