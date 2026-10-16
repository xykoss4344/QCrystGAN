"""Pretraining set: the most stable slot-fitting structure of N random A-B-O systems.

Reads datasets/oxide_pretrain/candidates.csv (oxide_pretrain_scan.py), keeps the lowest
E_hull entry per system (one per system, so no cross-source dedup is needed), samples
N_SYS systems, renames A->Mg, B->Mn (A = more electropositive), encodes, round-trip
checks, then runs rebuild_dataset (supercells, augmentation, labels, quantiles).

    QGAN_CHEM=oxide_pretrain python datasets/build_oxide_pretrain.py
"""
import csv
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from chemistry import CHEM, data_path  # noqa: E402

N_SYS = 1200
LEMAT = os.path.expanduser('~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/'
                           'snapshots/*/compatible_pbe/*.parquet')


def main():
    assert CHEM == 'oxide_pretrain'
    import pyarrow.parquet as pq
    from ase import Atoms
    from pymatgen.core import Lattice, Structure
    from build_chem_dataset import pair_floors
    from eval_wave import one_structure
    from make_representation import do_feature

    best = {}
    for r in csv.DictReader(open(data_path('candidates.csv'))):
        if r['system'] == 'Mg-Mn-O':
            continue           # the target system stays out of pretraining
        if r['system'] not in best or float(r['e_hull']) < float(best[r['system']]['e_hull']):
            best[r['system']] = r
    rng = np.random.default_rng(0)
    systems = sorted(best)
    pick = {best[s]['immutable_id']: best[s] for s in rng.choice(systems, min(N_SYS, len(systems)), replace=False)}
    rows, names, bad = [], [], 0
    for f in sorted(glob.glob(LEMAT)):
        t = pq.read_table(f, columns=['immutable_id', 'species_at_sites', 'cartesian_site_positions', 'lattice_vectors'])
        for r in t.to_pylist():
            m = pick.get(r['immutable_id'])
            if m is None:
                continue
            st = Structure(Lattice(r['lattice_vectors']), r['species_at_sites'],
                           r['cartesian_site_positions'], coords_are_cartesian=True)
            a, b = st.lattice.abc, st.lattice.angles
            if max(a) > 30 or min(b) < 30 or max(b) > 150:
                bad += 1
                continue
            ph = {m['A']: 'Mg', m['B']: 'Mn', 'O': 'O'}
            order = {'Mg': 0, 'Mn': 1, 'O': 2}
            sites = sorted(st.sites, key=lambda s: order[ph[s.specie.symbol]])
            atoms = Atoms([ph[s.specie.symbol] for s in sites],
                          scaled_positions=[s.frac_coords % 1.0 for s in sites], cell=st.lattice.matrix, pbc=True)
            row = do_feature(atoms).reshape(-1)
            lab = (np.abs(row.reshape(30, 3)[2:]).sum(-1) > 0.4).astype(np.float32)
            back = one_structure(row.astype(np.float32), lab)
            dm0 = st.distance_matrix
            if back is None or len(back) != len(st):
                bad += 1
                continue
            dm = back.distance_matrix
            if abs(dm[dm > 1e-8].min() - dm0[dm0 > 1e-8].min()) > 1e-3:
                bad += 1
                continue
            rows.append(row)
            names.append(f"{r['immutable_id']}:{m['system']}")
        print(f, len(rows), bad, flush=True)
    print(f'encoded {len(rows)} of {len(pick)} systems, skipped {bad}')
    np.save(data_path('unique_mgmno.npy'), np.array(rows).reshape(-1, 30, 3))
    pickle.dump(names, open(data_path('unique_mgmno_name_list'), 'wb'))
    json.dump(pair_floors(rows), open(data_path('floors.json'), 'w'), indent=1)
    import rebuild_dataset
    rebuild_dataset.main()


if __name__ == '__main__':
    main()
