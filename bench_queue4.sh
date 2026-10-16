#!/bin/sh
# Screened sets (CHGNet top 100 of 300), after the MatterGen n=300 benchmark finishes.
until grep -q "mattergen_mgmno_n300_relaxed timeouts" bench_queue3.log; do sleep 60; done
for d in s1_stable10_w23_s1_440_n300_screen100 mattergen_mgmno_n300_screen100 w23_relax3_s1_200_screen100; do
  echo "$(date +%H:%M) start $d"; sh lemat_only.sh bench300/$d || echo "FAILED $d"; done
echo "$(date +%H:%M) queue done"
