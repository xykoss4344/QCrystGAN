"""Relaunch a run's exact config with overrides, detached, and record the PID.

    python launch_like.py runs/g1_mgmno_lemat_w23_s1 runs/new_name chem=mgmno_stable seed=2

`chem=` sets QGAN_CHEM and points dataset_path at that chemistry's data.
Prints and appends "name,pid" to runs/launched_pids.csv.
"""
import json
import os
import subprocess
import sys

src, dst, *over = sys.argv[1:]
a = json.load(open(os.path.join(src, 'args.json')))
a.update(out_folder=dst, resume_checkpoint='')
for kv in over:
    k, v = kv.split('=', 1)
    a[k] = type(a[k])(v) if k in a and not isinstance(a[k], bool) else (v == '1' if k in a else v)
chem = a.pop('chem')
a['dataset_path'] = os.path.abspath(os.path.join('datasets', '' if chem == 'mgmno' else chem,
                                                 'mgmno_100_aug.pickle'))
cmd = [sys.executable, 'train_crystal.py']
for k, v in a.items():
    if isinstance(v, bool):
        cmd += [f'--{k}'] if v else []
    else:
        cmd += [f'--{k}', str(v)]
os.makedirs(dst, exist_ok=True)
env = dict(os.environ, QGAN_CHEM=chem, PYTHONUNBUFFERED='1')
p = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(dst, 'stdout.log'), 'a'),
                     stderr=open(os.path.join(dst, 'stderr.log'), 'a'),
                     creationflags=0x00000008 | 0x00000200)  # DETACHED_PROCESS | NEW_PROCESS_GROUP
open('runs/launched_pids.csv', 'a').write(f'{os.path.basename(dst)},{p.pid}\n')
print(os.path.basename(dst), p.pid)
