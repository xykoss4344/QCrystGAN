#!/bin/sh
# Li generalisation of the final recipe (same as chain_camno.sh), one system per call:
#   sh chain_li.sh limno LiMnO 8.9 14.0      sh chain_li.sh licoo LiCoO 7.9 12.3
# Data: LeMat Li-B-O crystals with benchmark-MLIP E_hull <= 0.1 (LeMat DFT hull is broken for Li).
# note: mkdir lock = one LeMat at a time across both chains.
S=$1; MG=$2; P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
C=${S}_stable
lemat() { until mkdir /tmp/lemat_lock 2>/dev/null; do sleep 120; done; "$@"; rmdir /tmp/lemat_lock; }
bench() {   # dir -> relax+LeMat, CHGNet screen, LeMat on screened
  lemat sh bench_full.sh $1 || echo FAILED $1
  QGAN_CHEM=$C $P chgnet_screen.py ${1}_relaxed --k 100 | tail -1
  lemat sh lemat_only.sh ${1}_screen100 || echo FAILED $1 screen
}
QGAN_CHEM=$C $P - <<PY
import os, pickle, numpy as np, torch
from chemistry import data_path
from eval_wave import one_structure
from baselines.substitution import generate as substitute
np.random.seed(0); torch.manual_seed(0)
raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
rc = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
rl = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
idx = np.random.choice(len(rl), 300)
sub, kept = substitute(rc, rl, rl[idx])
out = 'bench300/${S}_substitution_n300'; os.makedirs(out, exist_ok=True)
k = 0
for c, l in zip(sub, rl[idx[kept]]):
    st = one_structure(c, l)
    if st is not None:
        st.to(filename=f'{out}/${S}_substitution_n300_{k:04d}.cif'); k += 1
print(k, 'substitution CIFs')
PY
bench bench300/${S}_substitution_n300
until [ -f /c/Users/Adminb/mattergen/results/chemsys_$MG/generated_crystals_cif.zip ]; do sleep 300; done
QGAN_CHEM=$C $P baselines/mattergen_export.py /c/Users/Adminb/mattergen/results/chemsys_$MG --out bench300/${S}_mattergen_n300 --n 300
bench bench300/${S}_mattergen_n300
until [ -f runs/l1_${C}_s1/checkpoint_440.pt ]; do sleep 300; done
$P launch_like.py runs/s4_relax10_ft_s1 runs/l2_${S}_final_s1 chem=$C vpa_lo=$3 vpa_hi=$4 \
   resume_checkpoint=runs/l1_${C}_s1/checkpoint_440.pt n_epochs=560
until [ -f runs/l2_${S}_final_s1/checkpoint_550.pt ]; do sleep 300; done
QGAN_CHEM=$C $P benchmark.py --ckpt runs/l2_${S}_final_s1/checkpoint_550.pt --n 300 --cif_dir bench300/_$S --cif_only | tail -1
mv bench300/_$S/l2_${S}_final_s1_550 bench300/l2_${S}_final_s1_550_n300 && rmdir bench300/_$S
bench bench300/l2_${S}_final_s1_550_n300
echo "$(date +%m-%d_%H:%M) $S done"
