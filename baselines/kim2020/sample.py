"""Sample the Kim et al. 2020 GAN for the benchmark: 300 crystals, compositions drawn from the
training labels exactly as for SH-QGAN (eval_wave.sample), written as CIFs.

    QGAN_CHEM=mgmno_stable python baselines/kim2020/sample.py runs_s1/generator_500 OUT_DIR --seed 1

Occupied slots are the rows the generator does not leave at the zero padding (Kim's
remove_zero_padding rule, coordinate sum > 0.4). Undecodable samples are not written; the
decoded count is printed and saved so they can be counted as failures (n stays 300).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from models import Generator  # noqa: E402
from chemistry import data_path  # noqa: E402
from eval_wave import one_structure  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('out')
ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed)

gen = Generator(argparse.Namespace(latent_dim=512, input_dim=512 + 28 + 1))
gen.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
gen.eval()

raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)[np.random.choice(len(raw), a.n)]
cnt = np.stack([labels[:, :8].sum(1), labels[:, 8:16].sum(1), labels[:, 16:].sum(1)], 1).astype(int)
onehot = lambda k, w: torch.eye(w)[torch.tensor(k - 1)]
with torch.no_grad():
    img = gen(torch.randn(a.n, 512), onehot(cnt[:, 0], 8), onehot(cnt[:, 1], 8), onehot(cnt[:, 2], 12),
              torch.tensor(cnt.sum(1) / 28.0, dtype=torch.float32)[:, None]).numpy().reshape(a.n, 30, 3)

os.makedirs(a.out, exist_ok=True)
name = os.path.basename(os.path.normpath(a.out))
k = 0
for x in img:
    occ = (x[2:].sum(1) > 0.4).astype(np.float32)   # Kim's remove_zero_padding rule
    st = one_structure(x.flatten(), occ)
    if st is not None:
        st.to(filename=os.path.join(a.out, f'{name}_{k:04d}.cif')); k += 1
open(os.path.join(a.out, 'decoded.txt'), 'w').write(f'{k} {a.n}\n')
print(f'{name}: {k}/{a.n} decoded -> {a.out}')
