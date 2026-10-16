"""Learned geometry denoiser: a data prior applied to generated crystals before relaxation.

Diagnosis (runs/PLAN.md, 2026-10-06): every pair distance in the generator's raw output is
0.2-0.4 A too short (Mn-O median 1.80 vs real 1.96) and the CHGNet relaxer removes ~0.9
eV/atom, against ~0.0 for MatterGen. Penalising short contacts made the generator inflate
the cell instead. This learns the fix from data instead: an E(3)-equivariant periodic
message-passing network (models.QINR_Crystal.PeriodicRefiner) trained to remove Gaussian
position noise (0.02-0.5 A, Cartesian) from REAL crystals, conditioned on the noise level.
At sampling time it is applied to the generator's output with a decreasing noise schedule.
Cell parameters are untouched; only atom positions move.

    QGAN_CHEM=mgmno_stable20 python denoiser.py train runs/denoiser_st20.pt
    QGAN_CHEM=mgmno_stable python denoiser.py sample runs/denoiser_st20.pt <ckpt> <out_dir> --n 300
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
from chemistry import data_path
from crystal_physics import fractional, lattice_matrix
from crystal_mic import BOX_OFFSET, BOX_SCALE
from models.QINR_Crystal import PeriodicRefiner

SCHEDULE = (0.4, 0.3, 0.2, 0.1, 0.05)   # A; the generator's errors are ~0.2-0.4 A


class Denoiser(nn.Module):
    def __init__(self, d_tok=32, rounds=4):
        super().__init__()
        self.sig = nn.Sequential(nn.Linear(1, d_tok), nn.SiLU(), nn.Linear(d_tok, d_tok))
        self.net = PeriodicRefiner(d_tok=d_tok, rounds=rounds)

    def forward(self, frac, cell6, label, sigma):
        tok = self.sig(torch.log(sigma).view(-1, 1, 1)).expand(-1, frac.shape[1], -1)
        return self.net(frac, cell6, label, tok)


def mic_cart(a, b, lat):
    """Minimum-image Cartesian difference a - b, (B, n, 3)."""
    d = a - b
    d = d - torch.round(d)
    return torch.matmul(d, lat)


def load_rows(chem):
    raw = pickle.load(open(data_path('mgmno_100_aug.pickle'), 'rb'))
    c = torch.tensor(np.array([np.array(x).flatten() for x, _ in raw]), dtype=torch.float32)
    l = torch.tensor(np.array([np.array(y).flatten() for _, y in raw]), dtype=torch.float32)
    return c, l


def train(out, epochs=40, device='cuda'):
    c, l = load_rows(os.environ.get('QGAN_CHEM'))
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(c)
    for ep in range(epochs):
        perm, tot = torch.randperm(n), 0.0
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            x, lab = c[idx].to(device), l[idx].to(device)
            frac, lat = fractional(x), lattice_matrix(x)
            sigma = torch.exp(torch.empty(len(idx), device=device).uniform_(np.log(0.02), np.log(0.5)))
            noise = torch.randn_like(frac) * sigma.view(-1, 1, 1)
            noisy = frac + torch.matmul(noise, torch.linalg.inv(lat))
            noisy = noisy - torch.floor(noisy)
            pred = model(noisy, x[:, :6], lab, sigma)
            occ = lab.unsqueeze(-1)
            err = (mic_cart(pred, frac, lat) ** 2 * occ).sum((1, 2)) / occ.sum((1, 2)).clamp(min=1)
            loss = (err / sigma ** 2).mean()          # each noise level weighted equally
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        print(f'epoch {ep} loss {tot / n:.4f} (3.0 = no better than leaving the noise in)', flush=True)
    torch.save(model.state_dict(), out)


@torch.no_grad()
def refine(model, fake, labels, schedule=SCHEDULE):
    """fake (B, 90) generator output -> same with denoised atom positions."""
    frac = fractional(fake)
    for s in schedule:
        frac = model(frac, fake[:, :6], labels, torch.full((len(fake),), s, device=fake.device))
    atoms = (BOX_OFFSET + BOX_SCALE * frac).reshape(len(fake), 84) * labels.repeat_interleave(3, 1)
    return torch.cat([fake[:, :6], atoms], 1)


def sample_to_cifs(den_path, ckpt, out_dir, n, schedule=SCHEDULE):
    from eval_wave import load_generator, one_structure, sample
    model = Denoiser(); model.load_state_dict(torch.load(den_path, map_location='cpu')); model.eval()
    raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
    rl = np.array([np.array(y).flatten() for _, y in raw], dtype=np.float32)
    np.random.seed(0); torch.manual_seed(0)
    coords, labels, _ = sample(load_generator(ckpt)[0], rl, n)
    fake = refine(model, torch.tensor(coords), torch.tensor(labels), schedule).numpy()
    os.makedirs(out_dir, exist_ok=True)
    k = 0
    for c, l in zip(fake, labels):
        st = one_structure(c, l)
        if st is not None:
            st.to(filename=os.path.join(out_dir, f'{os.path.basename(out_dir)}_{k:04d}.cif')); k += 1
    print(f'{k}/{n} denoised samples -> {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'sample'])
    ap.add_argument('model')
    ap.add_argument('ckpt', nargs='?')
    ap.add_argument('out_dir', nargs='?')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--schedule', default=','.join(map(str, SCHEDULE)), help='noise levels in A, high to low')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a.model, a.epochs)
    else:
        sample_to_cifs(a.model, a.ckpt, a.out_dir, a.n, tuple(float(x) for x in a.schedule.split(',')))
