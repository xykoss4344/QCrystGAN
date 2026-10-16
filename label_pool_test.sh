#!/bin/sh
# Same model (s4_relax10 ep550), compositions drawn from a wider label pool. Proxy only.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
for pool in mgmno_lemat mgmno; do
  d=bench300/pool_${pool}_s4_550
  ds=datasets/$pool/mgmno_100.pickle; [ $pool = mgmno ] && ds=datasets/mgmno_100.pickle
  QGAN_CHEM=mgmno_stable $P benchmark.py --ckpt runs/s4_relax10_ft_s1/checkpoint_550.pt --n 300 --dataset $ds --cif_dir bench300/_lp --cif_only 2>&1 | tail -1
  mv bench300/_lp/s4_relax10_ft_s1_550 $d && rmdir bench300/_lp
  $P relax_cifs.py $d ${d}_relaxed 2>&1 | grep relaxed
  $P chgnet_screen.py ${d}_relaxed --k 100 2>&1 | grep scored
  $P uniq_check.py ${d}_relaxed
done
echo "reference s4_relax10 (stable pool): $($P chgnet_screen.py bench300/s4_relax10_ft_s1_550_n300_relaxed --k 100 2>&1 | grep scored)"
$P uniq_check.py bench300/s4_relax10_ft_s1_550_n300_relaxed
echo "$(date +%m-%d_%H:%M) pool test done"
