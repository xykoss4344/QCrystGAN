#!/bin/sh
# Full benchmark for one CIF set: CHGNet relax, then LeMat-GenBench on raw and
# relaxed, then the timeout check (any "Timeout processing" invalidates E_hull/MSUN).
#   sh bench_full.sh bench300/<name>            (dir of CIFs; name = basename)
set -e
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # LeMat logs emoji; a cp1252 console kills the run
P=/c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe
L=/c/Users/Adminb/lemat-genbench
src=$(cd "$1" && pwd -W); name=$(basename "$1")
[ -d "$src"_relaxed ] || $P relax_cifs.py "$src" "$src"_relaxed
for n in "$name" "$name"_relaxed; do
  ls "$(dirname "$src")/$n"/*.cif | sed 's#^/c/#C:/#' > $L/list_$n.txt
  (cd $L && .venv/Scripts/python.exe scripts/run_benchmarks.py --cifs list_$n.txt \
     --config orb_mace_multi_mlip_hull --name $n > run_$n.log 2> run_${n}_err.log)
  echo "$n timeouts: $(grep -c 'Timeout processing' $L/run_$n.log $L/run_${n}_err.log | paste -sd' ')"
done
