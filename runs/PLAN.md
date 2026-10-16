# Wave 1 — launched 2026-08-29

8 parallel single-threaded runs, 500 epochs each. Every run has its own
out_folder: a shared output directory is what produced the untraceable epoch-10
std_z recorded in the Experiment Log.

| folder | floors | madelung | sf | force | seed | purpose |
|---|---|---|---|---|---|---|
| w1_base_s0   | literature | - | - | - | 0 | v10 reference, reproduces the epoch-20 regression |
| w1_base_s1   | literature | - | - | - | 1 | seed variance on the reference |
| w1_floors_s0 | data | - | - | - | 0 | **the hypothesis test** |
| w1_floors_s1 | data | - | - | - | 1 | |
| w1_floors_s2 | data | - | - | - | 2 | |
| w1_mad_s0    | data | 0.01 | - | - | 0 | + electrostatics |
| w1_sf_s0     | data | 0.01 | 1.0 | - | 0 | + quantum structure factor |
| w1_force_s0  | data | 0.01 | 1.0 | 0.01 | 0 | + DFT force distillation |

**Primary question:** does `--floors data` stop the epoch-20 peak from being a
peak? Compare w1_base_* against w1_floors_* on validity and std_z past epoch 20.
Three seeds on the floors row because a single seed cannot answer it.

Each row differs from the one above by exactly one flag, so any effect is
attributable.

**Read results with `checkpoint_best.pt`, not the last checkpoint**, and never
report validity without volume per atom beside it.

# Wave 9 — launched 2026-09-25

4 runs, 500 epochs, ~87 s/epoch -> ~12 h. PIDs in `wave9_pids.csv` (TrainPID
is the real interpreter; LauncherPID is the Python Manager shim -- stop the
TrainPID with `Stop-Process -Id <id> -Force`).

Base config (every run), reconstructed from the w7/w8 logs because nothing
recorded their flags: `--batch_size 64 --floors bond --lambda_dist 0
--lambda_vol 0 --lambda_nn 1.0 --lambda_vpa_dist 0.2 --lambda_mode_seek 2.0`.
bond floors, batch 64 and lambda_dist 0 are confirmed by the logs; nn/vpa_dist
follow wave 6, and mode_seek 2.0 is read from the name "ms20" by analogy with
"ldist02" = 0.2. Every run now writes `args.json`, so this cannot happen again.

| folder | ema | seed | purpose |
|---|---|---|---|
| w9_ema_s0 | 0.999 | 0 | EMA generator weights for eval |
| w9_ema_s1 | 0.999 | 1 | seed variance |
| w9_ema_s2 | 0.999 | 2 | seed variance |
| w9_noema_s0 | - | 0 | control: same config and seed, no EMA |

Also new in this wave, for every run: a step whose gradient is non-finite is
skipped instead of written into the weights (the w8_ms20_s4 death).

**Question:** does EMA hold validity past the epoch-70 peak that every w7/w8 run
decayed from, without costing std_z or energy? Compare w9_ema_s0 against
w9_noema_s0 on the *last* checkpoint, not the best.

## Wave 9b — added the same day

| folder | change vs w9_ema | seed |
|---|---|---|
| w9_msinv_ema_s0 | `--mode_seek_form inv --lambda_mode_seek 0.002` | 0 |
| w9_msinv_ema_s1 | same | 1 |

Why: the corrected eval shows the w7/w8 late decay is NOT z collapse -- std_z
*rises* 0.03 -> 0.10 as validity falls. In w7_ms20_s0 the -ratio mode-seeking
term grows -0.03 -> -0.08 after epoch 100 while the NN-distance loss doubles
(0.21 -> 0.47) and validity drops 79% -> 38%. -ratio is unbounded; the original
MSGAN 1/(ratio+eps) fades once z is alive. Compare against w9_ema_s0/s1 on the
LAST checkpoint.

# Waves 12-13 — 2026-09-26, clean data

w12: species-aware contact hinge back on (`--floors data`, re-measured on clean
data), lambda_dist 0.02 / 0.1. Epoch 60: 98-100% valid at 1.0 A but only 3-7%
LeMat-valid, vpa 14 (inflating). Violations: ~3.5 pairs/structure, median 0.29 A
short -- the squared hinge fades near the floor.

w13 (all: inv MS 0.002, EMA 0.999 + warm-up, floors data, vpa_dist 1.0, seed 0):

| folder | hinge | lambda_dist | isolates |
|---|---|---|---|
| w13_lin01_vd1_s0 | lin | 0.1 | hinge shape (vs sq01) |
| w13_lin03_vd1_s0 | lin | 0.3 | hinge weight |
| w13_sq01_vd1_s0 | sq | 0.1 | vpa_dist 1.0 (vs w12_dist10_msinv) |

Question: LeMat-valid (probe `lemat=`) up, vpa back to ~11.3, std_z > 0.01.

# Wave 26 — qubit-scaling sweep (2026-09-27)

Question: does the quantum trunk gain more per parameter as it grows? Lead
config (w20: refiner + species features + force 0.5), seed 1, quantum vs
classical_matched at equal width. 12 qubits already exists (w20_force05_s1 vs
w24_classical_matched_s1).

| width | quantum | classical |
|---|---|---|
| 4 | w26_q4_s1 (96 circuit weights) | w26_c4_s1 |
| 8 | w26_q8_s1 (192) | w26_c8_s1 |
| 16 | w26_q16_s1 (384, GPU: 4.1 s/batch; CPU would be 26 s) | w26_c16_s1 |

Motivation: the 3-seed ablation at 12 qubits was a tie (energy -0.008 +/- 0.014
plain, +0.003 +/- 0.040 structure-factor). Reviewer item #12 asks for the sweep.

Note: w26_q16_s1 at batch 64 reached 18.6 GB host RAM (PennyLane keeps every
intermediate 2^16-amplitude state for backprop) and was stopped. The 16-qubit
pair runs at --batch_size 32 for BOTH quantum and classical (w26_q16_b32_s1,
w26_c16_b32_s1): ~5 GB, stable.

## Wave 26 trim — 2026-09-27 15:30
Stopped w26_q4_s1/w26_c4_s1 at ep100/110 (9 matched readings: energy/validity tie,
S.U.N. +1.3 pt for quantum 9/9, single seed) and w26_q16_b32_s1/w26_c16_b32_s1
(quantum at ~70 min/epoch on CUDA default.qubit, never reached ep10; needs a faster
simulator e.g. lightning.gpu/CUDA-Q under WSL2). Added w26_q8_s2/w26_c8_s2 (seed 2).
Remaining: w23 lead + 8-qubit seeds 0/1/2.
Stopped w26_q8_s1/w26_c8_s1 at ep100 (quantum dE lower at all 10 readings, gap shrank 0.17 -> 0.03).
Stopped w26_q8_s0/w26_c8_s0 at ep100 (classical ahead early, tie ep30-50, quantum ep60-70, classical ep90-100).
Added 4-qubit seeds 0 and 2 (w26_q4_s0/c4_s0/q4_s2/c4_s2) for the paper's 3-seed requirement.
Stopped w23_relax3_s1 at ep~265 (tracker flat since ep200: SUN 14-15%, dE 0.64). Lead checkpoint = ep200.
Launched g1_* generalisation runs (w23 recipe, seed 1): limno, licoo, camno, mgmno_lemat (data-size control).
Stopped w26_q8_s2/w26_c8_s2 at ep100.
Stopped w26_q8_s2/w26_c8_s2 at ep100 (by hand; the queued stop's regex failed).
Paused g1_licoo_w23_s1 and g1_camno_w23_s1 (+ trackers) at ~ep15 to relieve RAM/CPU (57/62 GB, 100% CPU). Relaunch after the 4-qubit seeds finish.
Stopped g1 trackers (limno, mgmno_lemat) to relieve commit charge (112/117 GB). Score g1 checkpoints later: QGAN_CHEM=<chem> python sun_track.py --n 150 runs/g1_<chem>_w23_s1
Stopped w26_[qc]4_s0/s2 at ep70-80 (tie; enough trained readings) and the main sun_track. Priority now: MatterGen head-to-head + generalisation (limno, mgmno_lemat).
Wave 27 (symmetry templates, crystal_symmetry.py, --sym_p 1.0): w27_sym_ft_s1 = fine-tune from w23 ep200; w27_sym_s1 = same recipe from scratch.
w27 stopped at ep~215/15: templates were special-position heavy (1.20 free coords/site vs real 1.53); most invalids = two 1-DOF sites coinciding. w27b = same with layouts weighted exp(7*mean DOF) (matches 1.53).
w27b_sym50_ft_s1: hedge, sym_p 0.5 fine-tune from w23 ep200 (w27b at ep10/210: collisions between different sites dominate invalids).
w27b stopped: centred conventional cells were not volume-scaled (C/I 7, R 4.3, F 3.3 A^3/atom vs 11.3). Fix: conventional lengths x mult^(1/3). Same w27b ep220 checkpoint: valid 45% -> 65% with no retraining. w27c = w27b + fix (ft from w23 ep200, and scratch).
w27c stopped (scratch: contact loss stuck ~15, 12% valid at ep20). Cause: heads never saw the template. w27d = w27c + template conditioning (group feat -> cell head, site feat -> atom tokens), zero-init.
w27d stopped (ft ep220 58% valid; scratch 13% at ep20 -- conditioning alone did not fix it). w27e = w27d + REAL-DATA templates (--sym_data 1.0; 507 templates of the training crystals; enumerated bank only where a composition has none).
Stopped g1_mgmno_lemat_w23_s1 at ep~245 (flat since ep200; ep200 is the benchmarked checkpoint).
Stopped w27e_symdata_s1 (scratch): every from-scratch symmetric run stalls ~13% valid at ep20 whatever the templates/conditioning; fine-tuning from a P1-trained model works (w27e ft 66% at ep210). Recipe = two-stage: P1 then symmetry.
Relaunched Li-Co-O / Ca-Mn-O as g2_licoo_w23_s1 / g2_camno_w23_s1 (w23 recipe = stage 1 of the two-stage recipe); g1_licoo/g1_camno (paused at ep~15) left on disk.
g3_mgmno_plus_w23_s1: w23 recipe on KAIST+LeMat Mg-Mn-O (1212 primitives). Motivation: LeMat-only control 19% relaxed MSUN vs KAIST w23 15%.
Paused g2_licoo / g2_camno at ep~20 (RAM fell to 1 GB during the symmetry benchmark). Resume from their checkpoints with --resume_checkpoint.
Stopped g1_limno at ep~215 (plateau; ep200 benchmarked). Resumed g2_licoo / g2_camno from their last checkpoints.
Stopped w27e_symdata_ft_s1 at ep~285: flat since ep230 (valid ~70-75%, SUN ~1%, dE ~1.3-1.4); benchmarked at ep230: relaxed MSUN 11% vs w23 15%. Symmetry as built does not beat P1 yet.
Stopped g2_camno at ep~205 (ep200 benchmarked).
Stopped g2_licoo at ep~200 (ep200 benchmarked).
g3_mgmno_plus_w23_s2: second seed of the combined-data run (s1 at ep130 already SUN 10.7%).
Stopped g3_mgmno_plus s1 (ep~200) and s2 (ep~55): combined data relaxed MSUN 13% vs w23 15% vs LeMat control 19% -- differences within n=100 noise (~+-4 pts). Data-quality claim not supported.

# Wave S — stability-filtered training data (2026-10-03/04, overnight)

Hypothesis: a generator reproduces its training set's stability. LeMat Mg-Mn-O
(the g1_mgmno_lemat set, 100 distinct) is only 39% metastable by DFT
(datasets/lemat_hull.py: hull from all 954 LeMat compatible_pbe Mg-Mn-O entries;
median E_hull 0.139). MatterGen's Alex-MP-20 is filtered to E_hull < 0.1.
New chemistries (build_chem_dataset.py with QGAN_EHULL_MAX):
mgmno_stable (<=0.1, 40 primitives), mgmno_stable20 (<=0.2, 63 primitives).
All runs = g1_mgmno_lemat_w23_s1 config (w23 recipe), launched with launch_like.py;
PIDs in runs/launched_pids.csv.

| folder | data | start |
|---|---|---|
| s1_stable10_w23_s1 | mgmno_stable | scratch |
| s1_stable20_w23_s1 | mgmno_stable20 | scratch |
| s2_stable_ft_s1 | mgmno_stable | fine-tune from g1_mgmno_lemat ep200 |
| s2_stable20_ft_s1 | mgmno_stable20 | fine-tune from g1_mgmno_lemat ep200 |

Note: sun_track's stable/dE proxy compares to the same-label real row, so on the
filtered sets the reference is itself lower energy -- not comparable across
datasets. Only LeMat decides. Scratch runs at ep50: 48% valid vs g1's 77%.

Benchmarks are now n=300 (bench300/, bench_full.sh). chgnet_screen.py adds a
generate-then-screen row (CHGNet hull, never the evaluator's MLIPs), applied to
every method alike.

## Wave S — memory event (2026-10-04, overnight)
Free RAM fell to ~8 GB of 62 with 4 trainings + 4 sun_track trackers + CHGNet
relax + MatterGen generation. Claude Code reaped its background shells:
- MatterGen part3 generation (480 extra samples): KILLED, nothing written. Not restarted.
- bench_full.sh bench300/w23_relax3_s1_200: shell killed; the relax_cifs.py child
  kept running, but the LeMat steps after it will NOT run. Rerun
  `sh bench_full.sh bench300/w23_relax3_s1_200` (it skips the relax if _relaxed exists
  -- check relax_log.csv was written first).
- All 4 trainings (s1_stable10, s1_stable20, s2_stable_ft, s2_stable20_ft) and the
  4 trackers survived. Stopping the trackers to free ~8 GB was blocked by the
  permission classifier; left for the user.
Lesson: on this box, budget ~6 GB per training (commit) and ~5 GB per tracker;
4 trainings + 4 trackers + relax + MatterGen exceeds 62 GB.

## Wave S results (LeMat, n=100, 0 timeouts)
| run | relaxed metastable | relaxed M.S.U.N. | relaxed E_hull | raw E_hull |
|---|---|---|---|---|
| w23 (KAIST) ep200 | 19% | 15% | 0.178 | 0.99 |
| g1_mgmno_lemat ep200 (parent) | 23% | 19% (MSUN/all) | 0.309 | 1.32 |
| **s2_stable_ft_s1 ep490** | **30%** | **23%** | 0.165 | 1.13 |
| MatterGen Mg-Mn-O | 55% | 34% | 0.164 | 0.17 |
+4 pts over its parent, +8 over w23: suggestive, inside n=100 noise vs parent. Needs n=300.
| s2_stable20_ft_s1 ep490 | 40% | 22% | **0.139** (below MatterGen 0.164) | 1.15 |
stable20 ft: best metastability and E_hull yet, but novelty drops to 84% (vs 97%) -- it
re-finds known crystals, so M.S.U.N. ties stable10 ft.
| s1_stable10_w23_s1 ep440 (scratch) | 36% | **25%** | 0.163 | 1.15 |
All three stable-data runs 22-25% vs w23 15% (n=100 each): consistent direction.

## Wave S2 — full-power launch (2026-10-04 18:20)
- s1_stable10_w23_s2: seed 2 of the best recipe (scratch, mgmno_stable).
- s1_stable10_classical_s1: classical_matched twin of s1_stable10_w23_s1 (quantum ablation on the new recipe).
- bench_queue2.sh: n=300 LeMat for s1_stable10 ep440, s2_stable20_ft ep490, s2_stable_ft ep490 (bench300/*_n300).
- bench_queue.sh still running: s1_stable20 ep440 (n=100) then w23 n=300.
- MatterGen part3 (480 more) generating.
- MatterGen part3 done (480 samples, 9.5 h). bench300/mattergen_mgmno_n300 = first 300 ternaries of part1-3 (file order, first 100 = the earlier set); benchmarking (bench_queue3.log).
- 2026-10-04 evening: free RAM hit ~3.5 GB (2 trainings + relax + 2 LeMat runs + user's Chrome).
  Watcher shells reaped (jobs themselves fine). Suspended the MatterGen n=300 LeMat tree
  (PIDs in runs/suspended_pids.txt) -- resume after bench_queue.log finishes s1_stable20.
  Lesson: never run two LeMat multi-MLIP benchmarks at once next to 2 trainings.
- Config check: bench_full.sh used comprehensive_multi_mlip_hull (orb+mace+UMA) instead of the
  orb_mace_multi_mlip_hull used for every earlier result. UMA fails to load (no HF access;
  e_above_hull_uma = nan) and the ensemble mean equals mean(orb, mace) on spot checks, so the
  numbers are comparable. bench_full.sh now pins orb_mace_multi_mlip_hull.
- 10-04 22:42 bench_queue done -> resumed MatterGen n=300 LeMat (resume_when_free.ps1).

## Night 2026-10-04 -> 05
- n=300 w23 relaxed: metastable 22%, M.S.U.N. 17.0% (13-22). Stable-data scratch n=300: 35% / 22.3% (18-27).
  Metastable gain significant (z~3.5); M.S.U.N. gain z~1.6 (uniqueness 85% vs 94% eats it).
- CHGNet screen validated on s1_stable10 n=300 relaxed: Spearman 0.74 with LeMat E_hull;
  CHGNet top-100 of 300 -> 75% LeMat-metastable (all 300: 35%). Screened sets for ours, w23 and
  MatterGen queued (bench_queue4.sh, lemat_only.sh) to measure M.S.U.N. after screening.
- s1_stable10_w23_s0: third seed of the stable-data recipe (with s1, s2) for the paper's 3-seed rule.

## Overnight scheduler (2026-10-04 23:15)
job_scheduler.ps1 launches runs/job_queue.txt one at a time when free RAM > 13 GB
(log: runs/job_queue_done.txt). Queue, in priority order:
1. s1_stable10_classical_s1 resumed from ep70 (it died silently at ep73 in the RAM crunch).
2. s3_vaegan_ft_s1: NEW --lambda_rec 10 (VAE-GAN: encoder + reconstruction of real crystals,
   KL 0.01) fine-tuned from s1_stable10_w23_s1 ep440 to ep560. Targets the measured mode
   dropping (uniqueness 85%). Smoke-tested 1 epoch: rec 0.034, kl 0.14.
3. g1_mgmno_lemat_classical_s1: classical twin at 100 crystals (quantum = g1_mgmno_lemat_w23_s1).
   With the stable10 pair (40 crystals) and the KAIST w20 pair, this is the quantum-vs-classical
   DATA-SIZE sweep (Leo's suggestion): does the quantum trunk help more as data shrinks?
4. s1_stable10_nosplit_s1: --no_split_head ablation on the stable recipe (split-head claim).
Metric focus (user, 2026-10-04): M.S.U.N. and the modern LeMat-GenBench metrics.
- MORE DATA: datasets/oxide_pretrain_scan.py found 27,313 DFT-metastable (E_hull<=0.1, own-system
  LeMat hull) slot-fitting ternary oxides in 2,775 A-B-O systems. build_oxide_pretrain.py: lowest-E_hull
  structure of 1200 random systems (Mg-Mn-O excluded), cations renamed to the Mg/Mn slots
  (A = more electropositive) -> 1185 encoded, 14,800 aug rows (chem oxide_pretrain).
  p1_oxide_pretrain_s1: pretrain 60 ep, no CHGNet distillation (would be wrong species), vpa 10.8-35.3.
  chain_pretrain_ft.ps1 queues p2_pretrain_ft_stable_s1 (full recipe on mgmno_stable from p1 ep50).

## KEY RESULT 2026-10-05 ~01:00 (n=300 head-to-head, 0 timeouts)
| | metastable | M.S.U.N. (95% CI) | novel | unique | E_hull |
|---|---|---|---|---|---|
| MatterGen Mg-Mn-O, n=300 | 48% | 24.0% (20-29) | 80% | 75% | 0.167 |
| SH-QGAN stable-data, n=300 | 35% | 22.3% (18-27) | 95% | 85% | 0.156 |
| MatterGen + CHGNet screen (best 100/300) | 90% | 39.0% (30-49) | 61% | 54% | 0.071 |
| SH-QGAN stable + CHGNet screen | 75% | 44.0% (35-54) | 86% | 72% | 0.093 |
MatterGen's n=100 34% was its first 100 (55% metastable, reproduced here); samples 101-300 are 45%
metastable and its uniqueness falls to 75% at n=300 (72 M.S.U.N. of 300). At equal n we are tied
unscreened and ahead screened (CIs overlap). M.S.U.N. depends on n through uniqueness: always compare equal n.
- 10-05 05:33 pretrain ep50 reached -> queued p2_pretrain_ft_stable_s1 (chain_pretrain_ft.ps1).

## 2026-10-05 day
- BUG: LeMat runs launched from some detached shells died at the end with
  "'charmap' codec can't encode character '\U0001f389'" (rich logging an emoji to a cp1252
  console) -> partial/NaN JSONs. Hit: s3_vaegan, g1_mgmno_lemat_classical, substitution_n300 (raw +
  screen), and s1_stable10_w23_s2 (stopped mid-run). Fix: bench_full.sh / lemat_only.sh export
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8. Reruns: bench_queue8.sh, then bench_queue9.sh (seed 2).
  Check every new LeMat err log for "Benchmark run failed" as well as "Timeout processing".
- n=300 results: stable20 fine-tune 23.0% (19-28); stable10 fine-tune 17.3% (its n=100 23% was noise);
  w23 + CHGNet screen 45.0% (36-55), novel 96%, unique 86% -- screening lifts every model.
- Scheduler had died overnight (no error captured); relaunched with stderr -> runs/scheduler_err.log.
- VAE-GAN (s3, lambda_rec 10, ft ep550) n=300: 22.7% (18-28), unique 85%, E_hull 0.145 -- no change vs parent 22.3%/85%: reconstruction did NOT fix uniqueness. Report as a negative ablation.
  VAE-GAN screened: 38% (parent screened 44%).
- User decision 2026-10-05: VAE-GAN stays OUT of the paper (removed from paper_results tables).
- Quantum vs classical at 40 stable crystals (n=300 each): 22.3% vs 21.3% M.S.U.N. (+1.0 pt, within noise; metastable 35% vs 34%). Tie at low data so far.

## ALL n=300 RESULTS IN (2026-10-06)
- Seeds of the stable-data recipe: 22.3 / 19.0 / 18.0% -> mean 19.8% (sd 2.2). Seed 1 was the best.
- Classical twin (40 crystals): 21.3% -> inside the quantum seed range: TIE.
- No split head: 4.7% (metastable 7%, E_hull 0.292) vs ~20% -> the split head is essential (4x).
- Oxide pretraining + fine-tune: 20.0% -> no gain over training on the 40 stable crystals alone.
- Substitution at n=300: 32.3% (27-38) -- best unscreened method; screened 71%.
- MatterGen n=300: 24.0%. Ours (3-seed mean) 19.8%: slightly behind, overlapping CIs.
- 100-crystal pair: quantum 19% (n=100) vs classical 14.3% (n=300); quantum n=300 running (bench_queue11).

## Wave 4 — beat substitution (32%) and MatterGen (24%) at n=300 (2026-10-06)
Gap: relaxed metastable 33% (us) vs 42% (sub) / 48% (MG); unique 85-88% vs 91%.
All fine-tunes from s1_stable10_w23_s1 ep440 to ep560:
- s4_relax10_ft_s1: lambda_relax 10 (was 3), relax_steps 50 (20), relax_k 8 (4) -> stability.
- s4_ms3x_ft_s1: lambda_mode_seek 0.006 (3x) -> uniqueness.
- s4_symdata_ft_s1: sym_p 0.5, sym_data 1.0, templates of the 59 stable crystals (43 templates) -> real prototypes.
- trunc_sweep.sh: truncation trick (QGAN_ZSCALE 0.5 / 0.7, eval_wave.sample) on ep440, CHGNet proxy + uniqueness.
- Truncation sweep (proxy, n=300): z-scale 1.0 / 0.7 / 0.5 -> CHGNet-metastable 194 / 203 / 198, unique 85 / 85 / 81%. Within noise: truncation does not help; dropped.
- 100-crystal pair at n=300: quantum 14.3% vs classical 14.3% -> exact tie. Quantum = classical at 40 and 100 crystals (the n=100 19% was noise).
- Wave 4 results (n=300, relaxed M.S.U.N. / metastable): relax10 23.7% / 37% (best; = MatterGen 24.4%),
  mode-seek 3x 18.7% / 33%, symmetry templates 14.1% / 33% (novelty falls to 84%: re-finds known crystals).
  Screened: relax10 42%, ms3x 37%, sym 28%. Substitution still 32.3% / 42%.
## Wave 5: s5_relax20_ft_s1 (from relax10 ep550: lambda 20, 100 steps, to ep660); relax10 recipe on seeds 0, 2
  (s5_relax10_ft_s0/s2 from their ep440). bench_queue13.sh scores each at n=300.

## Diagnosis 2026-10-06 (best model s4_relax10 vs substitution / MatterGen, n=300)
- Distinct formulas generated: 19 vs 79 / 73 (we sample only the 40 stable crystals' labels).
- CHGNet relax energy drop (median): 0.92 vs 0.05 / 0.00 eV/atom. Volume grows 5% on relaxing (IQR to +17%).
- P1 after relaxing: 67% vs 37% / 9%.
- ROOT CAUSE of the raw-quality gap: every pair distance is 0.2-0.4 A too short. Raw median (p10):
  Mn-O 1.80 (1.59) vs real 1.96 (1.91); Mg-O 1.86 (1.61) vs 2.09 (1.98); O-O 2.26 (2.01) vs 2.67 (2.54).
  The floors are right (Mn-O 1.85, Mg-O 1.95, O-O 2.5) but lambda_dist 0.1 with a squared hinge barely enforces them.
- Loss check: relax term flat ~0.1-0.2 at every weight; wgan rises 1.9 -> 2.4 as lambda_relax goes 3 -> 20.
## Wave 6: contact floors enforced (from s4_relax10 ep550 to ep640)
- s6_dist1lin_ft_s1: dist_hinge lin, lambda_dist 1.0;  s6_dist3lin_ft_s1: lin, 3.0.
- Wave 6 early check (ep560, 150 samples, raw): floors are met by INFLATING the cell, not rearranging:
  vpa 10.9 -> 12.2 (dist1lin) -> 14.2 (dist3lin) vs real 10.7; Mn-O median 1.81 -> 1.82 -> 1.89.
  Added s6_dist3vol_ft_s1: dist3lin + lambda_vol 3 (hinge 9.8-11.8 A^3/atom) + lambda_vpa_dist 3.
- relax20 (s5, lambda 20, 100 steps) n=300: 19.3% (metast 33%) < relax10 23.7%: more relaxation weight hurts. Sweet spot = relax10.
- dist3vol at ep570: vpa 12.2 (hinge 11.8 not held), Mn-O 1.83: model still inflates rather than rearranging.
- label_pool_test.sh: s4_relax10 sampled with KAIST / LeMat-100 composition pools (proxy + uniqueness).
- 10-06: RAM fell to 3.8 GB; stopped s6_dist1lin / s6_dist3lin / s6_dist3vol (all inflate the cell instead of fixing bonds; early checks enough) and their bench queues 15/16. Watcher shells were reaped (jobs fine).
- Composition pool test (s4_relax10 ep550, proxy): LeMat-100 label pool -> CHGNet-metastable 126/300 (42%) vs ~195/300 with the stable pool; unique 83%. Wider pools HURT (they add compositions of unstable crystals).
- RELAX10 RECIPE IS THE FINAL MODEL (n=300 each): seeds 0/1/2 = 20.7 / 23.7 / 22.0% -> mean 22.1% (sd 1.5),
  paired gain over the base recipe +2.7/+1.4/+3.0 (every seed). MatterGen 24.4% (20-29): within noise.
  Substitution 32.3% remains ahead. Paper tables now list the final model's 3 seeds.
## Wave 7: relax10 on stable20 models: s7_st20ft_relax10_s1 (from s2_stable20_ft ep490 -> 600), s7_st20_relax10_s1 (from s1_stable20 ep440 -> 550). bench_queue17.
- Wave 7 (n=300, 0 failures/timeouts): relax10 on stable20 scratch 22.0% (screened 51%), on stable20 fine-tune 22.3%
  (screened 41%). Same as the final recipe (22.1% mean): PLATEAU at ~22% unscreened for this architecture.
  Every lever tried since (truncation, wider label pools, floor penalties, more relax weight, mode seeking,
  symmetry templates, VAE-GAN, oxide pretraining) is flat or worse.

## Wave 8: learned geometry denoiser (denoiser.py), 2026-10-07
Trained on REAL stable20 crystals to remove 0.02-0.5 A Gaussian position noise (PeriodicRefiner backbone,
noise-level conditioned; cell untouched -> cannot inflate). 40 ep loss 1.16, 120 ep 0.93 (3.0 = identity).
Applied to s4_relax10 ep550 output (schedule 0.4..0.05): Mn-O 1.80 -> 1.94 (real 1.96), Mg-O 1.86 -> 1.96;
CHGNet relax drop 0.92 -> 0.43 eV/atom; CHGNet-metastable 194 -> 212/300; BUT unique 85% -> 78%.
Testing gentler schedules (den_sched_test.sh) and LeMat on the first denoised set (bench_den.log).
- Denoiser LeMat result (s4 ep550 + 40-ep denoiser, full schedule): 21.7% (metast 40%, novel 89%, unique 79%)
  vs 23.7% without: stability up, novelty+uniqueness down -> net worse (63-crystal denoiser memorises).
  Schedule tests, 120-ep denoiser (proxy CHGNet-metastable / unique): 0.2..0.05 201 / 82%;
  0.1,0.05 208 / 84%; 0.4..0.05 210 / 80%. Gentle schedule keeps uniqueness.
  Next: denoiser on mgmno_plus (KAIST+LeMat, ~1200 primitives) = generic local geometry, less memorisation;
  LeMat on den120_0p1_0p05.
- Big-data denoiser (mgmno_plus, 12k rows, loss 1.22): gentle 196/300 proxy, unique 87%; full 211/300, unique 80%.
  The uniqueness loss is NOT memorisation: denoising merges nearby generated samples into one basin.
- LeMat, gentle 120-ep denoiser on s4 ep550: 23.3% (metast 37%, novel 93%, unique 85%) vs 23.7% without.
  VERDICT: the denoiser fixes bond lengths (Mn-O 1.80 -> 1.94 A) and halves the relax energy drop,
  but does not raise M.S.U.N. -- after CHGNet relaxation, geometry errors were already being repaired.
  Final model stays relax10 recipe: 22.1% (3 seeds).

## RESULTS LOCKED 2026-10-08 (user decision)
Final model = relax10 recipe, 3 seeds 22.1 +/- 1.5%. Paper package in qgan/paper_results/:
table1-4 + all_runs.csv + fig1-4 (make_paper_results.py), best_crystals/ (12 novel, E_hull <= 0.037;
best_crystals.py), METHODS_AND_PROTOCOL.md, RESULTS_DRAFT.md. Params: generator 0.49M (288 circuit),
critic 0.21M; MatterGen chemical_system checkpoint 58.0M (counted locally).
## Generalisation (2026-10-08): final recipe on Ca-Mn-O
- Li-Mn-O / Li-Co-O DROPPED: LeMat compatible_pbe hull is broken for Li (LiCoO2 0.45, LiMn2O4 0.69 eV/atom above
  hull; only 2-3 'stable' crystals). Ca-Mn-O hull sane (CaMn2O4 0.0, Ca2MnO4 0.007, CaMnO3 0.035, ...);
  Mg-Mn-O re-checked (MgMn2O4, Mg6MnO8, MgMnO3 all 0.0).
- camno_stable: 66 entries E_hull<=0.1 -> 37 distinct primitives; vpa p50 12.76 -> prior 11.3-17.7.
- c1_camno_stable_s1 (base, 450 ep) -> chain_camno.sh: c2_camno_final_s1 (relax10, 441-560) -> n=300 bench;
  MatterGen 'Ca-Mn-O' (400 samples, results/chemsys_CaMnO) and substitution on camno_stable, n=300 each.
## Component ablations (2026-10-08): a0 full control, a1 no refiner, a2 no relax-distill, a3 no force-distill, a4 no critic geometry; single-stage 560 ep from scratch, final hyperparameters, seed 1; scheduler + bench_queue18 (n=300 at ep550). ARCHITECTURE.md written for the diagram.

## QUANTUM CLAIM TEST (2026-10-08, user decision: option 1, else fall back to "quantum parity")
Residual bypass removed (--no_trunk_residual): the signal must pass through the 2 x 12-qubit circuit
(288 params) or its classical_matched twin (288 params). Final recipe, mgmno_stable (40 crystals),
single-stage 560 ep from scratch, seeds 1/2/0, n=300 at ep550 (bench_queue19). Queued AHEAD of a1-a4.
Decision rule: quantum claim only if quantum > classical on M.S.U.N. in the 3-seed mean by more than
the seed spread (sd ~1.5-2.3 pts) AND in a majority of paired seeds. Otherwise -> quantum-parity framing.
- 10-08 15:29 RAM 7.4 GB: stopped a1_norefiner_s1 early (lower priority), re-queued at the end so the quantum-claim runs can start.
- 10-08 15:40 user: use 100% of the PC. Scheduler RAM threshold 13 -> 9 GB. LeMat MLIPs kept on CPU (GPU would change energies slightly vs locked results).

- 2026-10-08 gaming pause: low mode; suspended scheduler 8336, a0_full_s1 (22380), qn_quantum_nores_s2 (62512). Resume with NtResumeProcess + compute_mode full.
- 2026-10-08 resumed 8336/22380/62512, compute_mode full.

## 2026-10-08 night: Ca-Mn-O result + Li launch
- Ca-Mn-O n=300 relaxed: sub 26.3, MatterGen 23.0, ours c2 15.0 (metast 35, novel 90, uniq 78). Screened: sub 42, MG 26, ours 31.
  -> paper_results/table5_generalisation.md, RESULTS_DRAFT 4b.
- Li stable sets from benchmark-MLIP E_hull (LeMat DFT hull broken for Li; LiCoO2/LiMn2O4/Li2MnO3 ~0 on MLIP hull):
  datasets/{limno,licoo}_stable/lemat_ehull.csv; 103 / 97 crystals; vpa prior 8.9-14.0 / 7.9-12.3 (camno ratios).
- job_queue: l1_limno_stable_s1, l1_licoo_stable_s1 (base 450 ep, from c1 args) after a2.
- chain_li.sh (limno, licoo): substitution + MatterGen bench now, then l2 relax10 stage 441-560, n=300 bench. mkdir lock = 1 LeMat.
- MatterGen limno+licoo launched together on GPU -> VRAM overflow, ~15-20 s/it vs 2 it/s alone. Killing one was denied by
  classifier; left for user.

## 2026-10-09: crash recovery + paper figures
- 10-09 01:09 NVIDIA driver crash (nvlddmkm, CUDA unknown error) killed a0, qn_*_nores_s1/s2 and both MatterGen Li runs.
  Likely cause: VRAM full (MatterGen limno+licoo together + trainings). User saw a black overlay on wake and rebooted 11:20.
- All 11 interrupted runs re-queued from their last checkpoint (epoch_log.csv trimmed to the checkpoint; originals in
  epoch_log.precrash.csv). Resume restores weights, not Adam moments.
- job_scheduler.ps1: obeys runs/compute_mode.txt (low 5 / full 8 runs) and needs >= 5 GB free VRAM before launching.
- bench_queue18 (a0-a4) and bench_queue19 (qn) restarted 23:18. MatterGen Li NOT relaunched (VRAM); Li generalisation
  (chain_li.sh) paused until the GPU is free -- run the two MatterGen jobs one at a time.
- make_paper_results.py: figures now vector PDF + PNG, one style, all error bars 95% Wilson (multi-seed: pooled n,
  seed dots); new Fig. 5 novelty vs metastable; new tables 3b (component removals) and 4b (quantum claim test with the
  pre-registered rule) fill in automatically as LeMat results land.
- 10-10 02:25 LeMat raw run for qn_classical_nores_s0 logged 7 timeouts (8 trainings + 2 LeMat in parallel). Raw (unrelaxed)
  numbers are not used in any table, but it is INVALID. Trainings set to Idle priority so LeMat (Normal) wins the CPU.
  Every relaxed/screen run must be grep-checked for 'Timeout processing' before its row is trusted.
- 10-10 09:10 gaming: low mode; stopped qn_*_nores_s2 + a3 to free VRAM (re-queued from checkpoints); scheduler suspended.
10-10 12:56 gaming: stopped qn_quantum_nores_s2(ckpt160)/a1(ckpt20)/a4(ckpt10) to free VRAM, re-queued; other 4 trainings suspended; scheduler had exited (queue empty) - restart job_scheduler.ps1 on resume.
10-11 16:07 gaming, VRAM 1.4 GB free: stopped qn_quantum_nores_s2(ckpt530)/a1(490)/a4(390), re-queued; restart job_scheduler.ps1 on resume.

## 2026-10-12 data-size sweep (adviser: quantum had 63/107 runs, classical didn't)
ds{40,63,107}_{q,c}_s{0,1,2}: a0_full_s1 recipe (one 560-epoch stage), quantum vs classical_matched at each training-set size. ds40_q_s1 = a0_full_s1 (exists). Queue via job_scheduler.ps1, low mode. Then n=300 bench + LeMat at epoch 550 for all 18.
- 10-12 22:10 Kim et al. 2020 GAN baseline (adviser: compare different GANs): baselines/kim2020, seeds 0-2 on mgmno_stable aug, 501 ep CPU, sample gen_500 n=300 with training-label compositions; chain_kim.sh relaxes + LeMat.
- 10-12 22:48 queue reordered longest-first (ds107 s0/s2 next): ds107 runs ~5.5 min/epoch vs ds40 ~1.65 (3x data per epoch); LPT cuts makespan ~1 day.
- 2026-10-13 10:15 scheduler restarted as PID 27208: minVramMB 6500, 900 s after launch (180 s let 4 runs launch in 10 min and VRAM fell to 2.3 GB free). Relax+LeMat in both chains on CPU.
- 2026-10-13 14:10 ds107_c_s2 stopped at VRAM 0.4 GB free (user OK'd); resume from checkpoint_30 with RESUME_CMD.txt + --resume_checkpoint when a run finishes.
- 2026-10-13 22:05 scheduler restarted PID 36300 with minVramMB 3500 (6500 blocked replacement launches).
- 10-14 18:00: dropped ds63_q_s2/ds63_c_s2 from job_queue (user: finish sooner). Sweep = 40 & 107 x3 seeds; 63 = 2 seeds, no verdict.
