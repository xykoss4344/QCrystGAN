"""Relax a directory of CIFs with CHGNet: the generate-then-relax pipeline.

MatterGen and CDVAE-style pipelines relax generated structures before judging
stability; LeMat-GenBench itself relaxes only 50 steps. This relaxes fully and
records how far each structure moved, so the relaxer's share of the result
stays visible ("the relaxer is doing the work" -- Metric Problems.md). Apply it
to the baselines too, or the comparison is not fair.

    python relax_cifs.py bench_out/<method> bench_out/<method>_relaxed

Writes relaxed CIFs and relax_log.csv (file, n, E0, E, RMSD, steps).
"""
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main(src, dst, steps=500, fmax=0.05):
    from chgnet.model import StructOptimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    os.makedirs(dst, exist_ok=True)
    opt = StructOptimizer()
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    # Resumable: rows are appended to _progress.csv as they finish. A structure that was being
    # relaxed when the process was killed (_current.txt) is a failure -- some Kim et al. samples
    # stall CHGNet for hours (10-14), so the chain kills a relax that stops making progress.
    prog, cur = os.path.join(dst, '_progress.csv'), os.path.join(dst, '_current.txt')
    rows = list(csv.reader(open(prog))) if os.path.exists(prog) else []
    done = {r[0] for r in rows}
    if os.path.exists(cur):
        name = open(cur).read().strip()
        if name and name not in done:
            rows.append([name, '', '', '', '', 'fail: stalled, killed'])
            done.add(name)
    with open(prog, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    log = lambda row: (rows.append(row), csv.writer(open(prog, 'a', newline='')).writerow(row))
    for f in sorted(glob.glob(os.path.join(src, '*.cif'))):
        if os.path.basename(f) in done:
            continue
        open(cur, 'w').write(os.path.basename(f))
        try:  # unparseable CIF (e.g. overlapping sites) = failed sample, not a crash
            st = Structure.from_file(f)
        except Exception as e:
            log([os.path.basename(f), '', '', '', '', f'parse fail: {e}'[:60]])
            continue
        # < 2.5 A^3/atom is ~4x denser than any oxide (MgO ~9.4); CHGNet's neighbour graph explodes
        # on these (40 GB RAM on CPU, 10-13). Only Kim et al. samples hit this; counted as failures.
        if st.volume / len(st) < 2.5:
            log([os.path.basename(f), len(st), '', '', '', 'fail: V/atom < 2.5'])
            continue
        try:
            r = opt.relax(st, fmax=fmax, steps=steps, verbose=False)
        except Exception as e:
            log([os.path.basename(f), len(st), '', '', '', f'fail: {e}'[:60]])
            continue
        fin, tr = r['final_structure'], r['trajectory']
        rm = sm.get_rms_dist(st, fin)
        fin.to(filename=os.path.join(dst, os.path.basename(f)))
        log([os.path.basename(f), len(st), float(tr.energies[0]) / len(st),
                     float(tr.energies[-1]) / len(st), rm[0] if rm else 'unmatched',
                     len(tr.energies)])
    with open(os.path.join(dst, 'relax_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'n', 'E0', 'E', 'rmsd', 'steps'])
        w.writerows(rows)
    if os.path.exists(cur):
        os.remove(cur)
    ok = [r for r in rows if r[2] != ""]
    drop = sorted(float(r[2]) - float(r[3]) for r in ok)
    matched = sum(1 for r in ok if r[4] != 'unmatched')
    print(f'{src}: {len(ok)}/{len(rows)} relaxed, median energy drop '
          f'{drop[len(drop) // 2]:.3f} eV/atom, {matched}/{len(ok)} still match their start')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
