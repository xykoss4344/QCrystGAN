"""DFT E_above_hull for every Mg-Mn-O (or A-B-O) entry in LeMat-Bulk compatible_pbe.

The hull is built from all LeMat entries in the A-B-O space (unaries, binaries,
ternaries), so it uses the same functional and corrections as the energies.
Writes datasets/<chem>/lemat_ehull.csv: immutable_id, formula, nsites, fits_slots, e_hull.

    QGAN_CHEM=mgmno python datasets/lemat_hull.py
"""
import csv
import glob
import os
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from chemistry import A, B, data_path  # noqa: E402

LEMAT = os.path.expanduser('~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/'
                           'snapshots/*/compatible_pbe/*.parquet')


def main():
    space = {A, B, 'O'}
    entries, meta = [], []
    cols = ['immutable_id', 'elements', 'species_at_sites', 'energy', 'nsites']
    for f in sorted(glob.glob(LEMAT)):
        t = pq.read_table(f, columns=cols)
        t = t.filter(pc.less_equal(t['nelements'] if 'nelements' in t.column_names else
                                   pc.list_value_length(t['elements']), 3))
        for r in t.to_pylist():
            if not set(r['elements']) <= space or r['energy'] is None:
                continue
            sp = r['species_at_sites']
            comp = Composition({e: sp.count(e) for e in set(sp)})
            entries.append(PDEntry(comp, r['energy']))
            fits = (len(set(r['elements'])) == 3 and sp.count(A) <= 8
                    and sp.count(B) <= 8 and sp.count('O') <= 12)
            meta.append((r['immutable_id'], comp.reduced_formula, r['nsites'], fits))
    pd = PhaseDiagram(entries)
    out = data_path('lemat_ehull.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['immutable_id', 'formula', 'nsites', 'fits_slots', 'e_hull'])
        for e, m in zip(entries, meta):
            w.writerow([*m, round(pd.get_e_above_hull(e, allow_negative=True), 4)])
    print(f'{len(entries)} entries in {sorted(space)} -> {out}')


if __name__ == '__main__':
    main()
