#!/bin/sh
# Benchmark each run (n=300) as soon as its target checkpoint exists, one at a time,
# Items: chem run epoch. (Runs alongside bench_queue2: 2 LeMat at once is OK at >15 GB free.)
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
items="mgmno_stable:s3_vaegan_ft_s1:550 mgmno_lemat:g1_mgmno_lemat_classical_s1:200 mgmno_stable:s1_stable10_w23_s2:440 mgmno_stable:s1_stable10_classical_s1:440 mgmno_stable:s1_stable10_w23_s0:440 mgmno_stable:s1_stable10_nosplit_s1:440"
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
