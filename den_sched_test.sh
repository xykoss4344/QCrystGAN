#!/bin/sh
# Gentler denoising schedules with the 120-epoch denoiser on s4_relax10 ep550. Proxy + uniqueness.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
for sc in 0.2,0.1,0.05 0.1,0.05 0.4,0.3,0.2,0.1,0.05; do
  d=bench300/den120_$(echo $sc | tr ',.' '_p')
  QGAN_CHEM=mgmno_stable $P denoiser.py sample runs/denoiser_st20_e120.pt runs/s4_relax10_ft_s1/checkpoint_550.pt $d --n 300 --schedule $sc 2>&1 | tail -1
  $P relax_cifs.py $d ${d}_relaxed 2>&1 | grep relaxed
  $P chgnet_screen.py ${d}_relaxed --k 100 2>&1 | grep scored | sed 's/kept.*//'
  $P uniq_check.py ${d}_relaxed
done
echo "$(date +%m-%d_%H:%M) sched test done"
