"""Find every DFT-metastable ternary oxide A-B-O in LeMat-Bulk that fits the 8/8/12 slots.

Pass 1 keeps composition + energy of every <=3-element entry; for each A-B-O system with a
slot-fitting ternary, a PhaseDiagram over its sub-systems gives E_above_hull. Writes
datasets/oxide_pretrain/candidates.csv (immutable_id, system, formula, A, B, e_hull) for
entries with e_hull <= EMAX. A is the more electropositive cation (Mg-like slot).

    python datasets/oxide_pretrain_scan.py
"""
import csv
import glob
import os
from collections import defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element

EMAX = 0.1
LEMAT = os.path.expanduser('~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/'
                           'snapshots/*/compatible_pbe/*.parquet')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oxide_pretrain')
NOBLE = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}


def main():
    os.makedirs(OUT, exist_ok=True)
    by_set = defaultdict(list)   # frozenset(elements) -> [(id, comp, energy)]
    for f in sorted(glob.glob(LEMAT)):
        t = pq.read_table(f, columns=['immutable_id', 'elements', 'nelements', 'species_at_sites', 'energy'])
        t = t.filter(pc.less_equal(t['nelements'], 3))
        for r in t.to_pylist():
            if r['energy'] is None or NOBLE & set(r['elements']):
                continue
            sp = r['species_at_sites']
            comp = Composition({e: sp.count(e) for e in set(sp)})
            by_set[frozenset(r['elements'])].append((r['immutable_id'], comp, r['energy']))
        print(f, sum(map(len, by_set.values())), flush=True)
    rows = []
    for es, ents in by_set.items():
        if len(es) != 3 or 'O' not in es:
            continue
        a, b = sorted((e for e in es if e != 'O'), key=lambda e: Element(e).X or 9)
        fit = [(i, c, en) for i, c, en in ents
               if c[a] <= 8 and c[b] <= 8 and c['O'] <= 12]
        if not fit:
            continue
        sub = [x for s in by_set if s <= es for x in by_set[s]]
        try:
            pd = PhaseDiagram([PDEntry(c, en) for _, c, en in sub])
        except Exception:
            continue
        for i, c, en in fit:
            eh = pd.get_e_above_hull(PDEntry(c, en), allow_negative=True)
            if eh <= EMAX:
                rows.append([i, '-'.join(sorted(es)), c.reduced_formula, a, b, round(eh, 4)])
    with open(os.path.join(OUT, 'candidates.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['immutable_id', 'system', 'formula', 'A', 'B', 'e_hull'])
        w.writerows(rows)
    print(f'{len(rows)} metastable slot-fitting ternary oxides in '
          f'{len(set(r[1] for r in rows))} systems -> {OUT}/candidates.csv')


if __name__ == '__main__':
    main()
