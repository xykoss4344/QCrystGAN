#!/bin/sh
# LeMat-GenBench on a CIF dir as-is (no relax), e.g. an already-relaxed screened set.
#   sh lemat_only.sh bench300/<name>
set -e
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # LeMat logs emoji; a cp1252 console kills the run
L=/c/Users/Adminb/lemat-genbench
src=$(cd "$1" && pwd -W); n=$(basename "$1")
ls "$src"/*.cif | sed 's#^/c/#C:/#' > $L/list_$n.txt
(cd $L && .venv/Scripts/python.exe scripts/run_benchmarks.py --cifs list_$n.txt \
   --config orb_mace_multi_mlip_hull --name $n > run_$n.log 2> run_${n}_err.log)
echo "$n timeouts: $(grep -c 'Timeout processing' $L/run_$n.log $L/run_${n}_err.log | paste -sd' ')"
