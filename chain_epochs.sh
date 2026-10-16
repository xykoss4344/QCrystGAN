#!/bin/sh
# Sample-quality convergence: M.S.U.N. of a0_full_s1 (40 crystals, quantum, seed 1) at epochs 300/400/500,
# same pipeline as chain_ds.sh (n=300, CPU CHGNet relax, LeMat orb+MACE). Epoch 550 is already benchmarked.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
L=/c/Users/Adminb/lemat-genbench
r=a0_full_s1
for e in 300 400 500; do (
  n=${r}_${e}_n300
  rm -rf bench300/_ep$e bench300/$n bench300/${n}_relaxed
  QGAN_CHEM=mgmno_stable $P benchmark.py --ckpt runs/$r/checkpoint_$e.pt --n 300 --cif_dir bench300/_ep$e --cif_only | tail -1
  mv bench300/_ep$e/${r}_$e bench300/$n && rmdir bench300/_ep$e
  CUDA_VISIBLE_DEVICES=-1 $P relax_cifs.py bench300/$n bench300/${n}_relaxed | tail -1
) & done
wait
for e in 300 400 500; do
  n=${r}_${e}_n300
  ls "$PWD/bench300/${n}_relaxed"/*.cif | sed 's#^/c/#C:/#' > $L/list_${n}_relaxed.txt
  (cd $L && CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/run_benchmarks.py --cifs list_${n}_relaxed.txt \
     --config orb_mace_multi_mlip_hull --name ${n}_relaxed > run_${n}_relaxed.log 2> run_${n}_relaxed_err.log)
  echo "$(date '+%m-%d %H:%M') $n done, timeouts: $(grep -c 'Timeout processing' $L/run_${n}_relaxed.log $L/run_${n}_relaxed_err.log | paste -sd' ')"
done
echo "$(date '+%m-%d %H:%M') all done"
