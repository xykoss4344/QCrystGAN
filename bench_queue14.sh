#!/bin/sh
# Seed 2 of the relax10 recipe, in parallel with bench_queue13. The checkpoint is renamed
# after export so bench_queue13 does not score it a second time (q13 then never prints
# "queue done"; watch for this file's "queue done" instead).
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
r=s5_relax10_ft_s2; d=bench300/${r}_550_n300
QGAN_CHEM=mgmno_stable $P benchmark.py --ckpt runs/$r/checkpoint_550.pt --n 300 --cif_dir bench300/_q14 --cif_only 2>&1 | tail -1
mv runs/$r/checkpoint_550.pt runs/$r/checkpoint_550_scored.pt
mv bench300/_q14/${r}_550 $d && rmdir bench300/_q14
sh bench_full.sh $d || echo "FAILED $r"
QGAN_CHEM=mgmno $P chgnet_screen.py ${d}_relaxed --k 100 2>&1 | tail -1
sh lemat_only.sh ${d}_screen100 || echo "FAILED screen $r"
echo "$(date +%m-%d_%H:%M) queue done"
