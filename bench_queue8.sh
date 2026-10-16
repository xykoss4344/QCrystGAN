#!/bin/sh
# Rerun of the LeMat steps that died on the cp1252 emoji crash (relaxations already done).
for d in s3_vaegan_ft_s1_550_n300 g1_mgmno_lemat_classical_s1_200_n300 substitution_n300; do
  echo "$(date +%m-%d_%H:%M) start $d"
  sh bench_full.sh bench300/$d || echo "FAILED $d"
  sh lemat_only.sh bench300/${d}_screen100 || echo "FAILED screen $d"
done
echo "$(date +%m-%d_%H:%M) queue done"
