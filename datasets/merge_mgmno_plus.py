"""datasets/mgmno_plus/: KAIST Mg-Mn-O primitives + LeMat-Bulk's (build_chem_dataset.py).

The LeMat-only control (153 distinct, DFT-relaxed) scored 19% relaxed M.S.U.N.
against 15% for the KAIST-trained lead with 8x the data: the LeMat crystals are
the more stable set. This trains on both. Duplicates across the two sources are
dropped (StructureMatcher, per formula), keeping the LeMat copy.

    QGAN_CHEM=mgmno_plus python datasets/merge_mgmno_plus.py
"""
import json, os, pickle, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from chemistry import CHEM, DATA_DIR, data_path
assert CHEM == 'mgmno_plus'


def main():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from eval_wave import one_structure
    from build_chem_dataset import pair_floors
    os.makedirs(DATA_DIR, exist_ok=True)
    kaist = np.load(os.path.join(HERE, 'unique_mgmno.npy')).reshape(-1, 90)
    kn = pickle.load(open(os.path.join(HERE, 'unique_mgmno_name_list'), 'rb'))
    lem = np.load(os.path.join(HERE, 'mgmno_lemat', 'unique_mgmno.npy')).reshape(-1, 90)
    ln = pickle.load(open(os.path.join(HERE, 'mgmno_lemat', 'unique_mgmno_name_list'), 'rb'))
    st = lambda r: one_structure(r.astype(np.float32), (np.abs(r.reshape(30, 3)[2:]).sum(-1) > 0.4).astype(np.float32))
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    lem_s = [st(r) for r in lem]
    by_f = {}
    for s in lem_s:
        by_f.setdefault(s.composition.reduced_formula, []).append(s)
    keep, dup = [], 0
    for r in kaist:
        s = st(r)
        if any(sm.fit(s, t) for t in by_f.get(s.composition.reduced_formula, [])):
            dup += 1
        else:
            keep.append(r)
    rows = np.vstack([lem, np.array(keep)])
    names = list(ln) + [n for n, r in zip(kn, kaist) if any((r == k).all() for k in keep)]
    names = [f'{i}_{n}' for i, n in enumerate(names)]          # rebuild asserts unique names
    print(f'LeMat {len(lem)} + KAIST {len(kaist)} ({dup} duplicates of LeMat dropped) -> {len(rows)}')
    np.save(data_path('unique_mgmno.npy'), rows.reshape(-1, 30, 3))
    pickle.dump(names, open(data_path('unique_mgmno_name_list'), 'wb'))
    json.dump(pair_floors(rows), open(data_path('floors.json'), 'w'), indent=1)
    import rebuild_dataset
    rebuild_dataset.main()


if __name__ == '__main__':
    main()
