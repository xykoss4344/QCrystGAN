#!/bin/sh
# Quantum g1 (100 LeMat crystals) at n=300, to match its classical twin.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
QGAN_CHEM=mgmno_lemat $P benchmark.py --ckpt runs/g1_mgmno_lemat_w23_s1/checkpoint_200.pt --n 300 --cif_dir bench300 --cif_only 2>&1 | tail -1
mv bench300/g1_mgmno_lemat_w23_s1_200 bench300/g1_mgmno_lemat_w23_s1_200_n300
sh bench_full.sh bench300/g1_mgmno_lemat_w23_s1_200_n300 || echo FAILED
echo "$(date +%m-%d_%H:%M) queue done"
