#!/bin/sh
# Benchmark-hull (ORB+MACE) E_hull of every distinct Li-Mn-O / Li-Co-O training crystal, to stability-filter
# with the evaluator's own hull (the LeMat DFT hull is broken for Li). One LeMat at a time.
for c in limno licoo; do sh lemat_only.sh bench300/real_all_$c || echo FAILED $c; done
echo "$(date +%m-%d_%H:%M) li hull done"
