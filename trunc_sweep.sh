#!/bin/sh
# Truncation sweep on s1_stable10_w23_s1 ep440: CHGNet-proxy metastability + uniqueness.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
for z in 0.5 0.7; do
  d=bench300/trunc${z}_s1_440
  QGAN_ZSCALE=$z QGAN_CHEM=mgmno_stable $P benchmark.py --ckpt runs/s1_stable10_w23_s1/checkpoint_440.pt --n 300 --cif_dir bench300/_t --cif_only 2>&1 | tail -1
  mv bench300/_t/s1_stable10_w23_s1_440 $d && rmdir bench300/_t
  $P relax_cifs.py $d ${d}_relaxed 2>&1 | grep -v CHGNet
  $P chgnet_screen.py ${d}_relaxed --k 100 2>&1 | tail -1
  $P uniq_check.py ${d}_relaxed
done
$P uniq_check.py bench300/s1_stable10_w23_s1_440_n300_relaxed
echo "$(date +%m-%d_%H:%M) sweep done"
