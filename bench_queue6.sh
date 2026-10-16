#!/bin/sh
# Substitution baseline at n=300, after bench_queue2 and bench_queue4 finish.
until grep -q "queue done" bench_queue2.log && grep -q "queue done" bench_queue4.log; do sleep 120; done
sh bench_full.sh bench300/substitution_n300
/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe chgnet_screen.py bench300/substitution_n300_relaxed --k 100 2>&1 | tail -1
sh lemat_only.sh bench300/substitution_n300_screen100
echo "$(date +%m-%d_%H:%M) queue done"
