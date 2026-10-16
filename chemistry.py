"""Which ternary oxide A-B-O the 28-slot layout (A 0:8, B 8:16, O 16:28) holds.

Choose with the QGAN_CHEM environment variable (default 'mgmno'). Set it for
every process that touches a run: training, sun_track, eval, benchmark.

The dataset pipeline (datasets/1-7, rebuild_dataset.py) keeps the names Mg/Mn/O
as slot placeholders; build_chem_dataset.py renames A->Mg, B->Mn on the way in.
Only code that needs real elements reads this module: decoding to structures
(CHGNet, CIFs), atomic numbers, formal charges, contact floors.
"""
import json
import os

# name: (A cation, B cation, formal charge of A). O is always the anion.
CHEMS = {
    'mgmno': ('Mg', 'Mn', 2),
    'mgmno_lemat': ('Mg', 'Mn', 2),   # data-size control: LeMat-Bulk's Mg-Mn-O only
    'mgmno_plus': ('Mg', 'Mn', 2),    # KAIST + LeMat-Bulk Mg-Mn-O (datasets/merge_mgmno_plus.py)
    'mgmno_stable': ('Mg', 'Mn', 2),  # LeMat Mg-Mn-O with DFT E_hull <= 0.1 (MatterGen's Alex-MP-20 rule)
    'mgmno_stable20': ('Mg', 'Mn', 2),  # same, E_hull <= 0.2
    'oxide_pretrain': ('Mg', 'Mn', 2),  # ~1200 metastable A-B-O oxides from 2775 LeMat systems,
                                        # cations renamed to the Mg/Mn slots (pretraining only)
    'limno': ('Li', 'Mn', 1),
    'licoo': ('Li', 'Co', 1),
    'camno': ('Ca', 'Mn', 2),
    'limno_stable': ('Li', 'Mn', 1),   # E_hull <= 0.1 subsets (generalisation of the final recipe)
    'licoo_stable': ('Li', 'Co', 1),
    'camno_stable': ('Ca', 'Mn', 2),
}
CHEM = os.environ.get('QGAN_CHEM', 'mgmno')
A, B, Q_A = CHEMS[CHEM]
X = 'O'
SPECIES_MAP = [A] * 8 + [B] * 8 + [X] * 12

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'datasets') if CHEM == 'mgmno' else os.path.join(ROOT, 'datasets', CHEM)


def data_path(name):
    return os.path.join(DATA_DIR, name)


def data_floors():
    """Per-pair contact floors measured on this chemistry's data, or None for mgmno
    (whose floors live in crystal_mic)."""
    p = data_path('floors.json')
    if CHEM == 'mgmno' or not os.path.exists(p):
        return None
    return {tuple(k.split('-')): v for k, v in json.load(open(p)).items()}
