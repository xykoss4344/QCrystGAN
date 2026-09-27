"""Export the first N ternary Mg-Mn-O crystals MatterGen generated as CIFs.

MatterGen's chemical_system model (conditioned on 'Mg-Mn-O') also emits binaries
(Mg-O, Mn-O); those are outside the system we generate and are dropped, as is
anything over our 8 Mg / 8 Mn / 12 O slot limit. Taken in file order, so
the first N is not cherry-picked.

    python baselines/mattergen_export.py <mattergen results dir>... --out bench_out/mattergen_mgmno --n 100
"""
import argparse
import os
import warnings
import zipfile

warnings.filterwarnings('ignore')


def main():
    from pymatgen.core import Structure
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kept = seen = 0
    for d in a.dirs:
        z = zipfile.ZipFile(os.path.join(d, 'generated_crystals_cif.zip'))
        for name in sorted(z.namelist()):
            if not name.endswith('.cif') or kept >= a.n:
                continue
            seen += 1
            st = Structure.from_str(z.read(name).decode(), fmt='cif')
            c = st.composition
            if {e.symbol for e in c.elements} != {'Mg', 'Mn', 'O'}:
                continue
            if c['Mg'] > 8 or c['Mn'] > 8 or c['O'] > 12:
                continue
            st.to(filename=os.path.join(a.out, f'mattergen_mgmno_{kept:04d}.cif'))
            kept += 1
    print(f'kept {kept} ternary Mg-Mn-O of the first {seen} generated')
    assert kept == a.n, f'only {kept} usable crystals, wanted {a.n}'


if __name__ == '__main__':
    main()
