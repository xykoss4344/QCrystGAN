"""Build datasets/<chem>/ for another A-B-O chemistry from the local LeMat-Bulk cache.

    QGAN_CHEM=limno python datasets/build_chem_dataset.py

1. Every ternary A-B-O entry in LeMat-Bulk (compatible_pbe) whose cell fits the
   8 A / 8 B / 12 O slots.
2. Deduplicated with StructureMatcher per reduced formula (LeMat merges MP,
   Alexandria and OQMD, so the same crystal appears more than once).
3. Renamed A->Mg, B->Mn, sorted Mg..Mn..O (do_feature slices by count -- the
   supercell bug was exactly this order going wrong) and encoded.
4. Round-trip checked: decoding each row with the real species must give the
   crystal's own minimum distance.
   With QGAN_EHULL_MAX set, only entries whose DFT E_above_hull (lemat_hull.py,
   lemat_ehull.csv in the data dir) is at most that value are kept -- the
   stability filter MatterGen applies to Alex-MP-20 (0.1 eV/atom).
5. rebuild_dataset.py then makes supercells, augmentation, labels, quantiles and
   the novelty reference; this writes the contact floors (floors.json).
"""
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
from chemistry import A, B, CHEM, DATA_DIR, data_path  # noqa: E402

LEMAT = os.path.expanduser('~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/'
                           'snapshots/*/compatible_pbe/*.parquet')


def load_lemat():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from pymatgen.core import Lattice, Structure
    want = sorted([A, B, 'O'])
    out = []
    ehull = None
    if os.environ.get('QGAN_EHULL_MAX'):
        import csv
        emax = float(os.environ['QGAN_EHULL_MAX'])
        ehull = {r['immutable_id']: float(r['e_hull'])
                 for r in csv.DictReader(open(data_path('lemat_ehull.csv')))}
    cols = ['immutable_id', 'elements', 'nelements', 'species_at_sites',
            'cartesian_site_positions', 'lattice_vectors']
    for f in sorted(glob.glob(LEMAT)):
        t = pq.read_table(f, columns=cols)
        t = t.filter(pc.equal(t['nelements'], 3))
        for r in t.to_pylist():
            sp = r['species_at_sites']
            if sorted(r['elements']) != want:
                continue
            if sp.count(A) > 8 or sp.count(B) > 8 or sp.count('O') > 12:
                continue
            if ehull is not None and ehull.get(r['immutable_id'], 1e9) > emax:
                continue
            st = Structure(Lattice(r['lattice_vectors']), sp, r['cartesian_site_positions'],
                           coords_are_cartesian=True)
            out.append((r['immutable_id'], st))
    assert out, f'no {A}-{B}-O entries found under {LEMAT}'
    return out


def dedup(entries):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    by_f = {}
    for name, st in entries:
        by_f.setdefault(st.composition.reduced_formula, []).append((name, st))
    keep = []
    for group in by_f.values():
        structs = [s for _, s in group]
        ids = {id(s): n for n, s in group}
        keep += [(ids[id(g[0])], g[0]) for g in sm.group_structures(structs)]
    return keep


def encode(st):
    """pymatgen Structure (real species) -> 90-dim row with Mg/Mn/O placeholders."""
    from ase import Atoms
    from make_representation import do_feature
    order = {A: 0, B: 1, 'O': 2}
    sites = sorted(st.sites, key=lambda s: order[s.specie.symbol])
    placeholder = {A: 'Mg', B: 'Mn', 'O': 'O'}
    atoms = Atoms([placeholder[s.specie.symbol] for s in sites],
                  scaled_positions=[s.frac_coords % 1.0 for s in sites],
                  cell=st.lattice.matrix, pbc=True)
    return do_feature(atoms).reshape(-1)


def pair_floors(rows):
    """0.5th percentile over structures of each one's minimum contact per real
    species pair, floored to 0.05 A -- the recipe behind crystal_mic._MIN_SEP_DATA."""
    from eval_wave import one_structure
    per = {}
    for row in rows:
        lab = (np.abs(row.reshape(30, 3)[2:]).sum(-1) > 0.4).astype(np.float32)
        st = one_structure(row.astype(np.float32), lab)
        d = st.distance_matrix
        sp = [s.specie.symbol for s in st.sites]
        best = {}
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                k = '-'.join(sorted([sp[i], sp[j]]))
                best[k] = min(best.get(k, 1e9), d[i, j])
        for k, v in best.items():
            per.setdefault(k, []).append(v)
    pairs = ['-'.join(sorted(p)) for p in [(A, A), (A, B), (A, 'O'), (B, B), (B, 'O'), ('O', 'O')]]
    # A pair that never occurs (e.g. A-A in every structure has one A) gets the
    # smallest measured floor -- permissive, never tighter than the data.
    fl = {k: np.floor(np.percentile(per[k], 0.5) / 0.05) * 0.05 for k in pairs if k in per}
    fallback = min(fl.values())
    return {k: round(float(fl.get(k, fallback)), 2) for k in pairs}


def main():
    assert CHEM != 'mgmno', 'set QGAN_CHEM to the new chemistry (chemistry.CHEMS)'
    os.makedirs(DATA_DIR, exist_ok=True)
    entries = load_lemat()
    uniq = dedup(entries)
    print(f'{A}-{B}-O: {len(entries)} LeMat entries -> {len(uniq)} distinct')

    from eval_wave import one_structure
    rows, names, bad = [], [], 0
    for name, st in uniq:
        lengths, angles = st.lattice.abc, st.lattice.angles
        if max(lengths) > 30 or min(angles) < 30 or max(angles) > 150:
            bad += 1
            continue
        row = encode(st)
        lab = (np.abs(row.reshape(30, 3)[2:]).sum(-1) > 0.4).astype(np.float32)
        back = one_structure(row.astype(np.float32), lab)
        dm, dm0 = back.distance_matrix, st.distance_matrix
        if (back.composition != st.composition
                or abs(dm[dm > 1e-8].min() - dm0[dm0 > 1e-8].min()) > 1e-3):
            bad += 1
            continue
        rows.append(row)
        names.append(name)
    print(f'encoded {len(rows)}, skipped {bad} (cell outside the representation or round-trip mismatch)')
    assert bad <= 0.1 * len(uniq), 'more than 10% failed the round trip: encoder is wrong'

    np.save(data_path('unique_mgmno.npy'), np.array(rows).reshape(-1, 30, 3))  # layout of datasets/unique_mgmno.npy
    pickle.dump(names, open(data_path('unique_mgmno_name_list'), 'wb'))
    floors = pair_floors(rows)
    json.dump(floors, open(data_path('floors.json'), 'w'), indent=1)
    print('floors', floors)

    import rebuild_dataset
    rebuild_dataset.main()


if __name__ == '__main__':
    main()
