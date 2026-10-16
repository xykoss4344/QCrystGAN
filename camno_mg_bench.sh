#!/bin/sh
until grep -q "li hull done" li_hull_queue.log; do sleep 120; done   # one LeMat at a time
sh bench_full.sh bench300/camno_mattergen_n300 || echo FAILED mg
QGAN_CHEM=camno_stable /c/Users/Adminb/AppData/Local/Python/pythoncore-3.12-64/python.exe chgnet_screen.py bench300/camno_mattergen_n300_relaxed --k 100 | tail -1
sh lemat_only.sh bench300/camno_mattergen_n300_screen100 || echo FAILED mg screen
echo "$(date +%m-%d_%H:%M) mg done"
