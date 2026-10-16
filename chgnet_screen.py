"""Generate-then-screen: rank relaxed CIFs by CHGNet energy above a CHGNet hull, keep the best k.

The hull is built by CHGNet-relaxing the DFT ground states of the A-B-O space
(LeMat-Bulk entries on the DFT hull, from datasets/<chem>/lemat_ehull.csv), so the
screen never touches LeMat-GenBench's MLIPs (ORB/MACE/UMA) -- it cannot see the
evaluator. Apply the same screen to every method compared.

    python chgnet_screen.py bench300/<m>_relaxed --k 100      # -> bench300/<m>_screen100/
"""
import argparse
import csv
import glob
import os
import shutil
import warnings

warnings.filterwarnings('ignore')
from chemistry import A, B, data_path

LEMAT = os.path.expanduser('~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/'
                           'snapshots/*/compatible_pbe/*.parquet')
HULL = data_path('chgnet_hull.csv')


def hull_entries():
    """CHGNet-relaxed energies of the DFT ground states, cached in chgnet_hull.csv."""
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition
    if not os.path.exists(HULL):
        import pyarrow.parquet as pq
        from chgnet.model import StructOptimizer
        from pymatgen.core import Lattice, Structure
        gs = {r['immutable_id'] for r in csv.DictReader(open(data_path('lemat_ehull.csv')))
              if float(r['e_hull']) <= 1e-6}
        opt, rows = StructOptimizer(), []
        for f in sorted(glob.glob(LEMAT)):
            t = pq.read_table(f, columns=['immutable_id', 'species_at_sites',
                                          'cartesian_site_positions', 'lattice_vectors'])
            for r in t.to_pylist():
                if r['immutable_id'] in gs:
                    st = Structure(Lattice(r['lattice_vectors']), r['species_at_sites'],
                                   r['cartesian_site_positions'], coords_are_cartesian=True)
                    e = opt.relax(st, fmax=0.05, steps=500, verbose=False)['trajectory'].energies[-1]
                    rows.append([r['immutable_id'], st.composition.formula.replace(' ', ''), float(e)])
        with open(HULL, 'w', newline='') as fh:
            csv.writer(fh).writerows([['id', 'formula', 'energy'], *rows])
    return [PDEntry(Composition(r['formula']), float(r['energy'])) for r in csv.DictReader(open(HULL))]


def main():
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Structure
    ap = argparse.ArgumentParser()
    ap.add_argument('relaxed_dir')
    ap.add_argument('--k', type=int, default=100)
    a = ap.parse_args()
    pd = PhaseDiagram(hull_entries())
    scored = []
    for r in csv.DictReader(open(os.path.join(a.relaxed_dir, 'relax_log.csv'))):
        if not r['E']:
            continue
        comp = Structure.from_file(os.path.join(a.relaxed_dir, r['file'])).composition
        if not set(e.symbol for e in comp) <= {A, B, 'O'}:
            continue
        e = pd.get_e_above_hull(PDEntry(comp, float(r['E']) * comp.num_atoms), allow_negative=True)
        scored.append((e, r['file']))
    scored.sort()
    out = a.relaxed_dir.rstrip('/\\').replace('_relaxed', f'_screen{a.k}')
    os.makedirs(out, exist_ok=True)
    for _, f in scored[:a.k]:
        shutil.copy(os.path.join(a.relaxed_dir, f), out)
    with open(os.path.join(out, 'screen.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows([['file', 'chgnet_ehull'], *[(f, round(e, 4)) for e, f in scored]])
    n = len(scored)
    print(f'{a.relaxed_dir}: {n} scored, CHGNet E_hull<=0.1: {sum(e <= 0.1 for e, _ in scored)}/{n}; '
          f'kept best {min(a.k, n)} -> {out} (cutoff {scored[min(a.k, n) - 1][0]:.3f} eV/atom)')


if __name__ == '__main__':
    main()
