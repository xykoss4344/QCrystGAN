"""Symmetry templates: space group + Wyckoff sites -> the 28-slot layout.

Every SH-QGAN crystal so far is P1; 83% of the real training crystals are not.
A template fixes the space group and which Wyckoff site each atom sits on. The
generator then only chooses one free point per site (plus the cell); `expand`
copies it onto every symmetric position with fixed affine maps, so the output
is symmetric by construction and stays differentiable.

Cells are stored primitive (the representation holds <= 8 A / 8 B / 12 O), while
pyxtal's Wyckoff operations live in the conventional cell. Centred groups are
converted with the standard primitive vectors (rows, in conventional fractional
coordinates); centring translations then collapse onto the same primitive
position and are dropped.

    python crystal_symmetry.py        # self-check: real crystals round-trip exactly
    python crystal_symmetry.py bank   # datasets/<chem>/sym_bank.pkl (QGAN_CHEM)
"""
import numpy as np
import torch

# Rows: primitive lattice vectors in conventional fractional coordinates
# (spglib / ITA choice). R is the obverse hexagonal setting pyxtal uses.
_T = {
    'P': np.eye(3),
    'C': np.array([[.5, .5, 0], [-.5, .5, 0], [0, 0, 1]]),
    'A': np.array([[1, 0, 0], [0, .5, -.5], [0, .5, .5]]),
    'I': np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]),
    'F': np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
    'R': np.array([[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]]),
}
CENTRING_MULT = {'P': 1, 'C': 2, 'A': 2, 'I': 2, 'F': 4, 'R': 3}
# Conventional-cell centring translations (zero first), padded to 4.
_CENT = {'P': [], 'C': [[.5, .5, 0]], 'A': [[0, .5, .5]], 'I': [[.5, .5, .5]],
         'F': [[0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
         'R': [[2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]]}
_CENT = {k: np.array([[0, 0, 0]] + v + [[0, 0, 0]] * (3 - len(v)), dtype=float) for k, v in _CENT.items()}
N_SLOTS, SLOT_START = 28, (0, 8, 16)   # A 0:8, B 8:16, O 16:28


SYSTEMS = ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'hexagonal', 'cubic')
CENTRINGS = ('P', 'C', 'A', 'I', 'F', 'R')
N_GROUP_FEAT = len(SYSTEMS) + len(CENTRINGS) + 1                 # + space group number / 230
N_SITE_FEAT = 3 + 9 + 3                                          # mult, dof, first-slot; R0; t0


def crystal_system(sg):
    return ('triclinic' if sg <= 2 else 'monoclinic' if sg <= 15 else 'orthorhombic' if sg <= 74
            else 'tetragonal' if sg <= 142 else 'hexagonal' if sg <= 194 else 'cubic')


def _orbit_ops(wp, T):
    """Conventional-cell ops (R, t) of Wyckoff position `wp` whose images of the
    site's representative are distinct in the primitive cell."""
    Tinv = np.linalg.inv(T)
    R0, t0 = wp.ops[0].rotation_matrix, wp.ops[0].translation_vector
    probe = R0 @ np.array([0.1234, 0.3571, 0.6789]) + t0
    seen, ops = [], []
    for op in wp.ops:
        R, t = op.rotation_matrix, op.translation_vector
        p = ((R @ probe + t) @ Tinv) % 1.0
        if any(np.abs(((p - q + 0.5) % 1.0) - 0.5).max() < 1e-6 for q in seen):
            continue
        seen.append(p)
        ops.append((R, t))
    return (R0, t0), ops


class Template:
    """One (space group, Wyckoff layout) for a composition, as slot tensors.

    rep[j]: the slot whose free point generates slot j; R[j], t[j]: its map.
    """

    def __init__(self, sg, layout):
        from pyxtal.symmetry import Group
        g = Group(sg)
        self.sg, self.layout = sg, layout
        self.centring = g.symbol[0]
        self.T = _T[self.centring]
        self.cent = _CENT[self.centring]
        self.system = crystal_system(sg)
        rep = np.arange(N_SLOTS)
        R = np.tile(np.eye(3), (N_SLOTS, 1, 1))
        t = np.zeros((N_SLOTS, 3))
        R0 = np.tile(np.eye(3), (N_SLOTS, 1, 1))     # site projection (Wyckoff constraint)
        t0 = np.zeros((N_SLOTS, 3))
        occ = np.zeros(N_SLOTS)
        self.site_slots = [[], [], []]     # first slot of each Wyckoff site, per species
        for s, labels in enumerate(layout):
            j = SLOT_START[s]
            for lab in labels:
                (r0, s0), ops = _orbit_ops(g[lab], self.T)
                assert j + len(ops) <= SLOT_START[s] + (8, 8, 12)[s], \
                    f'{sg} {layout}: species {s} overflows its slots'
                self.site_slots[s].append(j)
                for k, (R_, t_) in enumerate(ops):
                    rep[j + k], R[j + k], t[j + k], occ[j + k] = j, R_, t_, 1
                    R0[j + k], t0[j + k] = r0, s0
                j += len(ops)
        self.rep, self.R, self.t, self.R0, self.t0, self.occ = rep, R, t, R0, t0, occ
        # What each slot is, for the heads: its orbit size, the site's free
        # coordinates, whether it is the site's first slot, and the Wyckoff
        # constraint itself (R0, t0). Zero on empty slots.
        mult = np.array([(rep == rep[j]).sum() if occ[j] else 0 for j in range(N_SLOTS)])
        dof = np.array([np.linalg.matrix_rank(R0[j]) if occ[j] else 0 for j in range(N_SLOTS)])
        first = (rep == np.arange(N_SLOTS)) & (occ > 0)
        self.site_feat = np.concatenate([np.stack([mult / 8.0, dof / 3.0, first], 1),
                                         R0.reshape(N_SLOTS, 9) * occ[:, None],
                                         t0 * occ[:, None]], 1)
        self.group_feat = np.concatenate([np.eye(len(SYSTEMS))[SYSTEMS.index(self.system)],
                                          np.eye(len(CENTRINGS))[CENTRINGS.index(self.centring)],
                                          [sg / 230.0]])

    def counts(self):
        return tuple(int(self.occ[a:b].sum()) for a, b in ((0, 8), (8, 16), (16, 28)))

    def placed(self, mask):
        """This template moved onto the occupied slots of a label.

        Templates fill each species block from its first slot, but augmented
        labels scatter occupancy (only 26% of Mg-Mn-O rows are packed first).
        The k-th template slot of a species goes to the k-th occupied slot.
        """
        where = np.arange(N_SLOTS)
        for (a, b) in ((0, 8), (8, 16), (16, 28)):
            occ_slots = a + np.nonzero(mask[a:b])[0]
            n = len(occ_slots)
            assert n == int(self.occ[a:b].sum()), 'label composition does not match template'
            free = [j for j in range(a, b) if j not in set(occ_slots)]
            where[a:a + n] = occ_slots                     # template slot a+k -> occupied slot
            where[a + n:b] = free
        out = Template.__new__(Template)
        out.__dict__.update(self.__dict__)
        inv = np.empty(N_SLOTS, dtype=int); inv[where] = np.arange(N_SLOTS)
        for k in ('R', 't', 'R0', 't0', 'occ', 'site_feat'):
            v = getattr(self, k); nv = np.empty_like(v); nv[where] = v; setattr(out, k, nv)
        nrep = np.empty_like(self.rep); nrep[where] = where[self.rep]; out.rep = nrep
        return out


def stack(templates, device=None, dtype=torch.float32):
    """Batch a list of Templates into tensors for `expand`."""
    f = lambda k: torch.tensor(np.stack([getattr(tp, k) for tp in templates]), dtype=dtype, device=device)
    return dict(rep=torch.tensor(np.stack([tp.rep for tp in templates]), device=device),
                R=f('R'), t=f('t'), R0=f('R0'), t0=f('t0'), occ=f('occ'),
                T=torch.tensor(np.stack([tp.T for tp in templates]), dtype=dtype, device=device),
                mult=torch.tensor([CENTRING_MULT[tp.centring] for tp in templates], dtype=dtype, device=device),
                cent=f('cent'), site_feat=f('site_feat'), group_feat=f('group_feat'),
                system=[tp.system for tp in templates])


def expand(free, tb):
    """(B, 28, 3) conventional-cell points -> symmetric primitive fractional coords.

    Each site reads the point at its first slot, projects it onto its Wyckoff
    position (R0 x + t0), and slot j becomes that point under the site's j-th op,
    converted to the primitive cell. Unoccupied slots pass through.
    """
    src = torch.gather(free, 1, tb['rep'].unsqueeze(-1).expand(-1, -1, 3))
    site = torch.einsum('bjkl,bjl->bjk', tb['R0'], src) + tb['t0']
    conv = torch.einsum('bjkl,bjl->bjk', tb['R'], site) + tb['t']
    prim = torch.remainder(conv @ torch.linalg.inv(tb['T']), 1.0)
    occ = tb['occ'].unsqueeze(-1)
    return occ * prim + (1 - occ) * free


def project(prim, tb):
    """Symmetric again after an unconstrained update (the refiner): read each
    site's primitive position back into the conventional cell and re-expand.
    Idempotent on already-symmetric input.

    A primitive position is one conventional point only up to a centring
    translation, and the Wyckoff projection is not invariant to it (x,0,0 sends
    (1/2,1/2,0) to (1/2,0,0)). Try each and keep the one closest to its own
    projection -- the copy that already sits on the site.
    """
    y = (prim @ tb['T']).unsqueeze(2) + tb['cent'].unsqueeze(1)          # (B,28,4,3)
    site = torch.einsum('bjkl,bjcl->bjck', tb['R0'], y) + tb['t0'].unsqueeze(2)
    miss = (torch.remainder(site - y + 0.5, 1.0) - 0.5).norm(dim=-1)      # (B,28,4)
    best = miss.argmin(-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 1, 3)
    return expand(torch.gather(y, 2, best).squeeze(2), tb)


def constrain_cell(lengths, angles, systems):
    """Project conventional cell parameters (B,3 A / degrees) onto each crystal
    system's constraints."""
    L, A = lengths.clone(), angles.clone()
    for i, s in enumerate(systems):
        if s == 'monoclinic':
            A[i, 0] = A[i, 2] = 90.0
        elif s == 'orthorhombic':
            A[i] = 90.0
        elif s == 'tetragonal':
            m = L[i, :2].mean(); L[i, 0] = L[i, 1] = m; A[i] = 90.0
        elif s == 'hexagonal':          # trigonal groups too (hexagonal setting)
            m = L[i, :2].mean(); L[i, 0] = L[i, 1] = m
            A[i, 0] = A[i, 1] = 90.0; A[i, 2] = 120.0
        elif s == 'cubic':
            L[i] = L[i].mean(); A[i] = 90.0
    return L, A


def cell_matrix(lengths, angles):
    """(B,3),(B,3) degrees -> (B,3,3) row-vector lattice (a along x)."""
    a, b, c = lengths.unbind(-1)
    al, be, ga = torch.deg2rad(angles).unbind(-1)
    ca, cb, cg, sg = torch.cos(al), torch.cos(be), torch.cos(ga), torch.sin(ga)
    z = torch.zeros_like(a)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = torch.sqrt((c ** 2 - cx ** 2 - cy ** 2).clamp(min=1e-6))
    return torch.stack([torch.stack([a, z, z], -1),
                        torch.stack([b * cg, b * sg, z], -1),
                        torch.stack([cx, cy, cz], -1)], 1)


def primitive_params(lengths, angles, tb):
    """Conventional cell -> primitive cell (lengths, angles) for the representation."""
    P = tb['T'] @ cell_matrix(lengths, angles)
    ln = P.norm(dim=-1)
    cosang = lambda i, j: (P[:, i] * P[:, j]).sum(-1) / (ln[:, i] * ln[:, j])
    ang = torch.rad2deg(torch.acos(torch.stack([cosang(1, 2), cosang(0, 2), cosang(0, 1)], -1).clamp(-1, 1)))
    return ln, ang


def _enumerate(job):
    """(composition, [(sg, weight)], max_layouts, seed) -> bank entries. Top level
    so multiprocessing can spawn it on Windows."""
    from pyxtal.symmetry import Group
    comp, groups, max_layouts, seed = job
    rng = np.random.default_rng(seed)
    entries = []
    for sg, w in groups:
        g = Group(sg)
        k = CENTRING_MULT[g.symbol[0]]
        res = g.list_wyckoff_combinations([k * x for x in comp], Nmax=2000)  # P-1 at 28 atoms explodes otherwise
        layouts = res[0] if res and res[0] else []
        if len(layouts) > max_layouts:
            layouts = [layouts[i] for i in rng.choice(len(layouts), max_layouts, replace=False)]
        if layouts:
            entries.append((sg, w, [[list(x) for x in lay] for lay in layouts]))
    return comp, entries


def build_bank(max_layouts=64, seed=0, workers=8):
    """Space-group prior + Wyckoff layouts per composition -> datasets/<chem>/sym_bank.pkl.

    The prior is the training data's space-group frequency (P1 included, 17% of
    Mg-Mn-O). Layouts are enumerated by pyxtal for each (composition, group), not
    copied from the data: 585 of 686 real crystals have a template of their own,
    so copying would regenerate the training set.
    ponytail: uniform over layouts within a group, capped at `max_layouts` per
    pair; weight by the data's Wyckoff statistics if special positions dominate.
    """
    import collections
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from chemistry import data_path
    from eval_wave import to_structures
    raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
    lab = np.array([np.array(y).flatten() for _, y in raw])
    comps = sorted({tuple(int(v) for v in (r[:8].sum(), r[8:16].sum(), r[16:].sum())) for r in lab})
    ref = np.load(data_path('novelty_ref_idx.npy'))
    c = np.array([np.array(x).flatten() for x, _ in raw], dtype=np.float32)[ref]
    prior = collections.Counter()
    for st in to_structures(c, lab[ref].astype(np.float32), cap=len(ref)):
        try:
            prior[SpacegroupAnalyzer(st, symprec=0.1).get_space_group_number()] += 1
        except Exception:
            pass
    # P-1 and C2/m take 20-36 s per large composition; ~2 h serially.
    import multiprocessing as mp
    jobs = [(comp, sorted(prior.items()), max_layouts, seed + i) for i, comp in enumerate(comps)]
    bank = {}
    with mp.Pool(workers) as pool:
        for i, (comp, entries) in enumerate(pool.imap_unordered(_enumerate, jobs)):
            bank[comp] = entries
            print(f'  {i + 1}/{len(jobs)} {comp}: {len(entries)} space groups', flush=True)
    out = data_path('sym_bank.pkl')
    pickle.dump(dict(prior=dict(prior), bank=bank), open(out, 'wb'))
    sym = sum(w for sg, w in prior.items() if sg != 1) / sum(prior.values())
    print(f'{len(comps)} compositions, {len(prior)} space groups ({sym:.0%} of data symmetric) -> {out}')
    return out


_DOF = {}


def site_dof(sg, label):
    """Free coordinates of a Wyckoff position (0 = fixed point, 3 = general)."""
    if (sg, label) not in _DOF:
        from pyxtal.symmetry import Group
        _DOF[(sg, label)] = Group(sg)[label].get_dof()
    return _DOF[(sg, label)]


def build_data_bank():
    """Templates of the real training crystals -> datasets/<chem>/sym_bank_data.pkl.

    Enumerated layouts are feasible but mostly unlike real crystals, and the
    generator could not fill them (w27d from scratch: 13% valid at epoch 20;
    hexagonal/tetragonal/I-centred templates 9-24% after fine-tuning). DiffCSP++
    and WyFormer learn their templates from data; this takes them from data.
    The generator still chooses every free coordinate and the cell. Keyed by
    the template's own primitive composition, which is the label composition
    except for supercell rows -- those fall back to the enumerated bank.
    """
    import collections
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    from pyxtal import pyxtal
    from chemistry import A, B, data_path
    from eval_wave import to_structures
    raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
    ref = np.load(data_path('novelty_ref_idx.npy'))
    c = np.array([np.array(x).flatten() for x, _ in raw], dtype=np.float32)[ref]
    lab = np.array([np.array(y).flatten() for _, y in raw], dtype=np.float32)[ref]
    counts = collections.Counter()
    for st in to_structures(c, lab, cap=len(ref)):
        x = pyxtal()
        try:
            x.from_seed(st, tol=0.1)
        except Exception:
            continue
        sg = x.group.number
        layout = [sorted(s.wp.get_label() for s in x.atom_sites if s.specie == e) for e in (A, B, 'O')]
        try:
            comp = Template(sg, layout).counts()
        except AssertionError:
            continue                        # conventional layout overflows the slots
        counts[(comp, sg, str(layout))] += 1
    bank = collections.defaultdict(dict)
    for (comp, sg, lay), n in counts.items():
        w, lays = bank[comp].get(sg, (0, []))
        bank[comp][sg] = (w + n, lays + [eval(lay)])
    bank = {comp: [(sg, w, lays) for sg, (w, lays) in d.items()] for comp, d in bank.items()}
    out = data_path('sym_bank_data.pkl')
    pickle.dump(dict(bank=bank), open(out, 'wb'))
    print(f'{sum(counts.values())} real crystals -> {len(counts)} templates over {len(bank)} compositions -> {out}')
    return out


class TemplateSampler:
    """Composition -> random Template: space group by data frequency, layout
    weighted by exp(beta * mean free coordinates per site).

    Uniform layouts average 1.20 free coordinates per site against 1.53 in real
    symmetric Mg-Mn-O: enumeration is dominated by special-position-heavy
    layouts, whose one-parameter sites the generator then has to keep apart
    (w27_sym_ft ep210: most invalid crystals were two such sites coinciding).
    beta = 7 brings the sampled mean to the data's (1.17 at beta 0, 1.58 at 8).
    """

    def __init__(self, path=None, p_sym=1.0, seed=0, beta=7.0, p_data=0.0):
        import os
        import pickle
        from chemistry import data_path
        d = pickle.load(open(path or data_path('sym_bank.pkl'), 'rb'))
        self.bank, self.p_sym, self.beta = d['bank'], p_sym, beta
        # p_data: chance of drawing a real crystal's template when one exists
        # for this composition (build_data_bank); otherwise an enumerated one.
        self.p_data = p_data
        dp = data_path('sym_bank_data.pkl')
        self.data_bank = pickle.load(open(dp, 'rb'))['bank'] if p_data > 0 and os.path.exists(dp) else {}
        self._lw = {}
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self._placed = {}   # ponytail: unbounded; a few thousand (template, mask) pairs

    def sample(self, comp):
        entries = self.bank.get(tuple(comp))
        real = self.data_bank.get(tuple(comp))
        if real and self.rng.random() < self.p_data:
            entries, beta = real, 0.0          # real templates: weight by frequency only
        else:
            beta = self.beta
        if not entries or self.rng.random() > self.p_sym:
            entries = [(1, 1, [[['1a'] * comp[0], ['1a'] * comp[1], ['1a'] * comp[2]]])]
        w = np.array([e[1] for e in entries], dtype=float)
        sg, _, layouts = entries[self.rng.choice(len(entries), p=w / w.sum())]
        key_w = (tuple(comp), sg, beta, len(layouts))
        if key_w not in self._lw:
            m = np.array([np.mean([site_dof(sg, x) for sp in lay for x in sp]) for lay in layouts])
            lw = np.exp(beta * (m - m.max()))
            self._lw[key_w] = lw / lw.sum()
        lay = layouts[self.rng.choice(len(layouts), p=self._lw[key_w])]
        key = (sg, str(lay))
        if key not in self._cache:
            self._cache[key] = Template(sg, lay)
        return self._cache[key]

    def batch(self, labels, device=None):
        """(B, 28) occupancy labels -> stacked templates matching each composition."""
        lab = labels.detach().cpu().numpy() > 0.5
        out = []
        for r in lab:
            tp = self.sample((int(r[:8].sum()), int(r[8:16].sum()), int(r[16:].sum())))
            key = (id(tp), r.tobytes())
            if key not in self._placed:
                self._placed[key] = tp.placed(r)
            out.append(self._placed[key])
        return stack(out, device=device)


def _selfcheck(n=60):
    """Real symmetric crystals: extract (space group, layout, free points), expand
    with this module, and match the original. Checks R/t/T and the slot layout."""
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    from pyxtal import pyxtal
    from chemistry import A, B, data_path
    from eval_wave import to_structures
    raw = pickle.load(open(data_path('mgmno_100.pickle'), 'rb'))
    ref = np.load(data_path('novelty_ref_idx.npy'))
    c = np.array([np.array(x).flatten() for x, _ in raw], dtype=np.float32)[ref]
    l = np.array([np.array(y).flatten() for _, y in raw], dtype=np.float32)[ref]
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    ok = tried = 0
    for st in to_structures(c, l, cap=len(ref)):
        if tried >= n:
            break
        x = pyxtal()
        try:
            x.from_seed(st, tol=0.1)
        except Exception:
            continue
        if x.group.number == 1:
            continue
        order = {A: 0, B: 1, 'O': 2}
        layout = [[], [], []]
        free = [[], [], []]
        for site in x.atom_sites:
            s = order[site.specie]
            layout[s].append(site.wp.get_label())
            free[s].append(site.position)
        try:
            tp = Template(x.group.number, layout)
        except AssertionError:
            continue          # conventional-cell layout exceeds the slots
        tried += 1
        tb = stack([tp])
        frac = torch.zeros(1, N_SLOTS, 3)
        for s in range(3):
            for j, p in zip(tp.site_slots[s], free[s]):
                frac[0, j] = torch.tensor(p)
        par = x.lattice.get_para(degree=True)
        L, Ang = constrain_cell(torch.tensor([par[:3]], dtype=torch.float32),
                                torch.tensor([par[3:]], dtype=torch.float32), tb['system'])
        pl, pa = primitive_params(L, Ang, tb)
        out = expand(frac, tb)[0].numpy()
        sp = [A] * 8 + [B] * 8 + ['O'] * 12
        occ = tp.occ.astype(bool)
        rebuilt = Structure(Lattice.from_parameters(*pl[0].tolist(), *pa[0].tolist()),
                            [sp[j] for j in range(N_SLOTS) if occ[j]], out[occ])
        ok += sm.fit(rebuilt, st)
        # project() must leave a symmetric crystal unchanged (refiner step)
        again = project(expand(frac, tb), tb)[0].numpy()
        d = np.abs(((again - out + 0.5) % 1.0) - 0.5)[occ].max()
        assert d < 1e-5, f'project() moved a symmetric crystal by {d}'
    print(f'{ok}/{tried} symmetric real crystals rebuilt exactly from template + free points')
    assert tried >= 20 and ok >= 0.9 * tried, 'template expansion does not reproduce real crystals'


if __name__ == '__main__':
    import sys
    build_data_bank() if 'databank' in sys.argv else build_bank() if 'bank' in sys.argv else _selfcheck()
