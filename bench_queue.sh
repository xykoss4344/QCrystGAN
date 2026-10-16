#!/bin/sh
# Overnight benchmark queue, one at a time (RAM budget). Log: bench_queue.log
for d in bench_out/s2_stable_ft_s1_490 bench_out/s2_stable20_ft_s1_490 \
         bench_out/s1_stable10_w23_s1_440 bench_out/s1_stable20_w23_s1_440 \
         bench300/w23_relax3_s1_200; do
  echo "$(date +%H:%M) start $d"; sh bench_full.sh $d || echo "FAILED $d"; done
echo "$(date +%H:%M) queue done"
