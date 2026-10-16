#!/bin/sh
# Overnight benchmark queue, one at a time (RAM budget). Log: bench_queue2.log
for d in bench300/s1_stable10_w23_s1_440_n300 bench300/s2_stable20_ft_s1_490_n300 bench300/s2_stable_ft_s1_490_n300; do
  echo "$(date +%H:%M) start $d"; sh bench_full.sh $d || echo "FAILED $d"; done
echo "$(date +%H:%M) queue done"
