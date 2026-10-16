"""Fraction of unique structures in a CIF dir (StructureMatcher, LeMat-like tolerances)."""
import glob, sys, warnings
warnings.filterwarnings('ignore')
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
ss = [Structure.from_file(f) for f in sorted(glob.glob(sys.argv[1] + '/*.cif'))]
groups = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).group_structures(ss)
print(f'{sys.argv[1]}: unique {len(groups)}/{len(ss)} = {len(groups)/len(ss):.3f}')
