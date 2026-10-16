"""Where does each method's M.S.U.N. come from? Per-structure LeMat E_hull joined with the
relaxed CIFs: metastable rate by composition and by cell size, space groups, relax drop.

    python diagnose.py <set_relaxed> [<set_relaxed> ...]      (bench300 dir names)
"""
import glob
import os
import sys
import warnings
from collections import Counter

import pandas as pd

warnings.filterwarnings('ignore')
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

R = 'C:/Users/Adminb/lemat-genbench/results_final/'
rows = []
for name in sys.argv[1:]:
    d = f'bench300/{name}'
    per = pd.read_csv(R + f'{name}_per_structure.csv')
    files = sorted(glob.glob(d + '/*.cif'))
    log = pd.read_csv(d + '/relax_log.csv').set_index('file') if os.path.exists(d + '/relax_log.csv') else None
    for _, r in per.iterrows():
        f = files[int(r['index'])]
        st = Structure.from_file(f)
        try:
            sg = SpacegroupAnalyzer(st, symprec=0.1).get_space_group_number()
        except Exception:
            sg = 1
        drop = (log.loc[os.path.basename(f), 'E0'] - log.loc[os.path.basename(f), 'E']) if log is not None else float('nan')
        rows.append(dict(set=name, formula=st.composition.reduced_formula, n=len(st), sg=sg,
                         ehull=r['e_above_hull_mean'], meta=r['e_above_hull_mean'] <= 0.1, drop=drop))
df = pd.DataFrame(rows)
df.to_csv('diagnose_rows.csv', index=False)
for name, g in df.groupby('set', sort=False):
    print(f'\n=== {name}: n={len(g)} metastable {g.meta.mean():.0%}, median relax drop {g['drop'].median():.2f} eV/at, '
          f'P1 {(g.sg == 1).mean():.0%}, distinct formulas {g.formula.nunique()}')
    by_n = g.groupby(pd.cut(g.n, [0, 8, 12, 16, 20, 28])).meta.agg(['mean', 'size'])
    print('  metastable by atoms/cell:', ', '.join(f'{i}: {m:.0%} (n={s})' for i, (m, s) in by_n.iterrows()))
    top = g.groupby('formula').meta.agg(['mean', 'size']).sort_values('size', ascending=False).head(8)
    print('  top formulas:', ', '.join(f'{f} {m:.0%}/{s}' for f, (m, s) in top.iterrows()))
    print('  top space groups:', Counter(g.sg).most_common(6))
