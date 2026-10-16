#!/bin/sh
# p2 (oxide pretrain + stable fine-tune) rerun: first run died on the emoji bug.
until grep -q "queue done" bench_queue9.log 2>/dev/null; do sleep 120; done
d=p2_pretrain_ft_stable_s1_190_n300
sh bench_full.sh bench300/$d || echo "FAILED $d"
[ -d bench300/${d}_screen100 ] || /c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe chgnet_screen.py bench300/${d}_relaxed --k 100 2>&1 | tail -1
sh lemat_only.sh bench300/${d}_screen100 || echo "FAILED screen $d"
echo "$(date +%m-%d_%H:%M) queue done"
