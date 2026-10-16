#!/bin/sh
# Data-size sweep: benchmark each ds* run as soon as its epoch-550 checkpoint lands (one at a time).
# Relaxed set only (the raw set is not used in the paper's quantum-vs-classical section).
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2  # oversubscribed CPU stalled LeMat 7 h (10-14)
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
L=/c/Users/Adminb/lemat-genbench
todo="ds40_c_s1 ds63_q_s1 ds63_c_s1 ds107_q_s1 ds107_c_s1 ds40_q_s0 ds40_c_s0 ds63_q_s0 ds63_c_s0 ds107_q_s0 ds107_c_s0
      ds40_q_s2 ds40_c_s2 ds63_q_s2 ds63_c_s2 ds107_q_s2 ds107_c_s2"
while [ -n "$(echo $todo)" ]; do
  left=""
  for r in $todo; do
    if [ ! -f runs/$r/checkpoint_550.pt ]; then left="$left $r"; continue; fi
    case $r in ds40_*) c=mgmno_stable;; ds63_*) c=mgmno_stable20;; *) c=mgmno;; esac
    n=${r}_550_n300
    grep -q " $n done" chain_ds.log && continue  # already benchmarked (restart-safe)
    if [ ! -f bench300/${n}_relaxed/relax_log.csv ]; then  # relax already done -> straight to LeMat
    rm -rf bench300/_ds bench300/$n bench300/${n}_relaxed
    QGAN_CHEM=$c $P benchmark.py --ckpt runs/$r/checkpoint_550.pt --n 300 --cif_dir bench300/_ds --cif_only | tail -1
    mv bench300/_ds/${r}_550 bench300/$n && rmdir bench300/_ds
    # CPU relax: on cuda next to 8 trainings it left 3.9 GB VRAM free (10-13)
    CUDA_VISIBLE_DEVICES=-1 $P relax_cifs.py bench300/$n bench300/${n}_relaxed | tail -1
    fi
    # one LeMat at a time; LeMat runs on CPU (8 trainings leave ~3.7 GB VRAM, 10-13)
    while ps -ef | grep -q '[r]un_benchmarks'; do sleep 120; done
    # LeMat on GPU when >=5 GB VRAM is free (~20 min), else CPU (~4 h at 100% CPU, 10-14)
    dev=-1; [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') -ge 5000 ] && dev=0
    ls "$PWD/bench300/${n}_relaxed"/*.cif | sed 's#^/c/#C:/#' > $L/list_${n}_relaxed.txt
    (cd $L && CUDA_VISIBLE_DEVICES=$dev .venv/Scripts/python.exe scripts/run_benchmarks.py --cifs list_${n}_relaxed.txt \
       --config orb_mace_multi_mlip_hull --name ${n}_relaxed > run_${n}_relaxed.log 2> run_${n}_relaxed_err.log)
    echo "$(date '+%m-%d %H:%M') $n done, timeouts: $(grep -c 'Timeout processing' $L/run_${n}_relaxed.log $L/run_${n}_relaxed_err.log | paste -sd' ')"
  done
  todo=$left
  [ -n "$(echo $todo)" ] && sleep 600
done
