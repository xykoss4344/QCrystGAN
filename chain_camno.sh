#!/bin/sh
# Ca-Mn-O generalisation of the final recipe: base (c1) -> relax10 stage (c2) -> n=300 benchmark,
# plus MatterGen (Ca-Mn-O) and substitution on the same 37 stable crystals. One LeMat at a time.
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
until [ -f runs/c1_camno_stable_s1/checkpoint_440.pt ]; do sleep 300; done
$P launch_like.py runs/s4_relax10_ft_s1 runs/c2_camno_final_s1 chem=camno_stable vpa_lo=11.3 vpa_hi=17.7 \
   resume_checkpoint=runs/c1_camno_stable_s1/checkpoint_440.pt n_epochs=560
# substitution while c2 trains
QGAN_CHEM=camno_stable $P - <<'PY'
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
out = 'bench300/camno_substitution_n300'; os.makedirs(out, exist_ok=True)
k = 0
for c, l in zip(sub, rl[idx[kept]]):
    st = one_structure(c, l)
    if st is not None:
        st.to(filename=f'{out}/camno_substitution_n300_{k:04d}.cif'); k += 1
print(k, 'substitution CIFs')
PY
sh bench_full.sh bench300/camno_substitution_n300 || echo FAILED sub
QGAN_CHEM=camno_stable $P chgnet_screen.py bench300/camno_substitution_n300_relaxed --k 100 | tail -1
sh lemat_only.sh bench300/camno_substitution_n300_screen100 || echo FAILED sub screen
until [ -f /c/Users/Adminb/mattergen/results/chemsys_CaMnO/generated_crystals_cif.zip ]; do sleep 300; done
QGAN_CHEM=camno_stable $P baselines/mattergen_export.py /c/Users/Adminb/mattergen/results/chemsys_CaMnO --out bench300/camno_mattergen_n300 --n 300
sh bench_full.sh bench300/camno_mattergen_n300 || echo FAILED mg
QGAN_CHEM=camno_stable $P chgnet_screen.py bench300/camno_mattergen_n300_relaxed --k 100 | tail -1
sh lemat_only.sh bench300/camno_mattergen_n300_screen100 || echo FAILED mg screen
until [ -f runs/c2_camno_final_s1/checkpoint_550.pt ]; do sleep 300; done
QGAN_CHEM=camno_stable $P benchmark.py --ckpt runs/c2_camno_final_s1/checkpoint_550.pt --n 300 --cif_dir bench300/_c --cif_only | tail -1
mv bench300/_c/c2_camno_final_s1_550 bench300/c2_camno_final_s1_550_n300 && rmdir bench300/_c
sh bench_full.sh bench300/c2_camno_final_s1_550_n300 || echo FAILED ours
QGAN_CHEM=camno_stable $P chgnet_screen.py bench300/c2_camno_final_s1_550_n300_relaxed --k 100 | tail -1
sh lemat_only.sh bench300/c2_camno_final_s1_550_n300_screen100 || echo FAILED ours screen
echo "$(date +%m-%d_%H:%M) camno done"
