#!/bin/sh
# Kim et al. 2020 GAN baseline: when a seed's epoch-500 generator lands, sample 300, relax, benchmark.
# Same relaxation and LeMat config as every other method in Table 1.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2  # oversubscribed CPU stalled LeMat 7 h (10-14)
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 QGAN_CHEM=mgmno_stable
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
L=/c/Users/Adminb/lemat-genbench
K=baselines/kim2020
todo="1 0 2"
while [ -n "$(echo $todo)" ]; do
  left=""
  for s in $todo; do
    if [ ! -f $K/runs_s$s/generator_500 ]; then left="$left $s"; continue; fi
    n=kim2020_s${s}_500_n300
    grep -q " $n done" chain_kim.log && continue  # already benchmarked (restart-safe)
    if [ ! -f bench300/${n}_relaxed/relax_log.csv ]; then  # relax already done -> straight to LeMat
    $P $K/sample.py $K/runs_s$s/generator_500 bench300/$n --seed $s | tail -1
    # CPU relax: on cuda next to the sweep it pushed VRAM to 0.6 GB free (10-13)
    CUDA_VISIBLE_DEVICES=-1 $P relax_cifs.py bench300/$n bench300/${n}_relaxed | tail -1
    fi
    # one LeMat at a time; LeMat runs on CPU (8 trainings leave ~3.7 GB VRAM, 10-13)
    while ps -ef | grep -q '[r]un_benchmarks'; do sleep 120; done
    # LeMat on GPU when >=5 GB VRAM is free (~20 min), else CPU (~4 h at 100% CPU, 10-14)
    dev=-1; [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') -ge 5000 ] && dev=0
    ls "$PWD/bench300/${n}_relaxed"/*.cif | sed 's#^/c/#C:/#' > $L/list_${n}_relaxed.txt
    (cd $L && CUDA_VISIBLE_DEVICES=$dev .venv/Scripts/python.exe scripts/run_benchmarks.py --cifs list_${n}_relaxed.txt \
       --config orb_mace_multi_mlip_hull --name ${n}_relaxed > run_${n}_relaxed.log 2> run_${n}_relaxed_err.log)
    echo "$(date '+%m-%d %H:%M') $n done, decoded $(cat bench300/$n/decoded.txt), timeouts: $(grep -c 'Timeout processing' $L/run_${n}_relaxed.log $L/run_${n}_relaxed_err.log | paste -sd' ')"
  done
  todo=$left
  [ -n "$(echo $todo)" ] && sleep 300
done
