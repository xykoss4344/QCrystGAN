#!/bin/sh
# Seed 2 rerun (its first LeMat run was stopped: started before the UTF-8 fix).
until grep -q "queue done" bench_queue8.log; do sleep 120; done
d=s1_stable10_w23_s2_440_n300
sh bench_full.sh bench300/$d || echo "FAILED $d"
sh lemat_only.sh bench300/${d}_screen100 || echo "FAILED screen $d"
echo "$(date +%m-%d_%H:%M) queue done"
