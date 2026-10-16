"""Quick raw-geometry check of a checkpoint: per-pair minimum distances of 150 samples vs real."""
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
from chemistry import data_path
from eval_wave import load_generator, one_structure, sample
import pickle
raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
rl = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
np.random.seed(0)
for ck in sys.argv[1:]:
    coords, labels, _ = sample(load_generator(ck)[0], rl, 150)
    acc = {}
    for c, l in zip(coords, labels):
        st = one_structure(c, l)
        if st is None: continue
        sp = [s.specie.symbol for s in st]; d = st.distance_matrix
        best = {}
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                k = '-'.join(sorted([sp[i], sp[j]])); best[k] = min(best.get(k, 9), d[i, j])
        for k, v in best.items(): acc.setdefault(k, []).append(v)
    vpa = np.median([one_structure(c, l).volume / len(one_structure(c, l)) for c, l in zip(coords[:60], labels[:60]) if one_structure(c, l)])
    print(os.path.basename(os.path.dirname(ck)), os.path.basename(ck), f'vpa {vpa:.1f}',
          '  '.join(f'{k} {np.median(v):.2f}' for k, v in sorted(acc.items()) if k in ('Mg-O', 'Mn-O', 'O-O')))
