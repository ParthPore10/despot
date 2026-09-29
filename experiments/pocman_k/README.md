# Pocman: how many scenarios does DESPOT need?

Vary scenario count **K** and measure game return, planning latency, search/policy
size, and repeatability of the chosen action. This experiment uses FullPocman
and the existing DESPOT search implementation without changing the solver library.

## Run

From the repository directory (`despot/`):

```sh
python3 experiments/pocman_k/run.py --preset pilot
```

The runner builds with `make` and a C++11 compiler, runs timing jobs **serially**
in reproducibly shuffled order, saves raw data, and generates summary CSVs and
a report. Python's standard library suffices for data analysis. Use a Python
environment with Matplotlib to also produce `overview.png` and `overview.svg`;
analysis reports when plotting is unavailable.

After inspecting the pilot, use `--preset study` for a much longer experiment
(search alone can take hours), or override settings:

```sh
python3 experiments/pocman_k/run.py --preset pilot \
  --ks 16,32,64,128,256,512,1024 --seeds 11,22,33,44,55 \
  --steps 100 --time 0.1 --depth 60 --prune 0.01 \
  --belief-particles 50000 --repeats 10 --probe-every 10 \
  --output experiments/pocman_k/results/my-sweep

python3 experiments/pocman_k/analyze.py experiments/pocman_k/results/my-sweep
python3 -m unittest discover -s experiments/pocman_k -p 'test_*.py'
```

Output directories must be new, preventing accidental replacement of results.
Failed jobs retain logs and a failed manifest; partial data must not be treated
as a completed sweep. Avoid competing computational workloads during timing.

## Two complementary experiments

**Closed-loop episodes:** For each `(seed, K)`, initialize Pocman and its belief,
then plan, act, and update until terminal or the step limit. Reuse the seed
schedule across K. Different actions produce different later states and
observations. Compare episode returns across seeds, not independent samples of
step rewards.

**Fixed-belief probes:** A separate reference controller (default: largest K)
generates a trajectory. Before it acts at every `probe_every` step, replan the
*same particle belief and action/observation history* at every K with repeated
planner seeds. Only the independent controller advances the world. Actions on
different closed-loop trajectories are not used to measure stability.

The reference trajectory samples beliefs visited by one controller, not every
Pocman situation. Use more seeds and, if needed, different `--reference-k`
values. The initial belief follows the stock model's prior without conditioning
on an initial observation.

## Controls

| Setting | Quick pilot | Longer study |
|---|---:|---:|
| K | 16, 32, 64, 128, 256, 512 | pilot values plus 1024 |
| Environment seeds | 11, 22, 33 | 100 through 129 |
| Episode/reference step cap | 40 | 200 |
| Search CPU budget per decision | 0.05 s | 0.2 s |
| Maximum search/rollout depth | 30 | 90 |
| Belief particles | 5,000 | 50,000 |
| Planner repeats per probe/K | 5 | 20 |
| Probe interval | 10 steps | 20 steps |
| Discount | 0.95 | 0.95 |
| Regularization penalty | 0.01 | 0.01 |
| Bounds | SMART lower / SMART upper | same |

The positive penalty is a fixed experimental choice, not a tuned optimum; the
stock global default is zero. Study regularization by repeating complete sweeps
with `--prune 0`, `0.01`, and other prespecified values. Likewise compare time
budgets in separate sweeps. Pilot and study settings differ, so do not combine
them into one K curve.

Belief-particle count approximates the current belief; K supplies planning
scenarios. Keep the former fixed. This wrapper requires at least 2,500 belief
particles to avoid the stock constructor's automatic splitting path. Record
actual surviving belief size; an empty filtered belief fails the job explicitly.

## Metrics

| Output | Meaning |
|---|---|
| `episodes.csv`: `return` | Sum of rewards through termination or the shared step cap: the headline average reward, in reward units per episode. |
| `discounted_return` | Sum of `discount^t * reward_t`, also reported because planning uses a discounted objective. |
| `planning_wall_s` | Full planning latency: sampling, stream generation, bound initialization, search, action selection, and cleanup. Excludes belief update and CSV output. |
| `planning_cpu_s` | CPU time for the same full planning call. |
| `search_cpu_s` | Stock DESPOT trial/backup time only, controlled by the nominal search budget. |
| `tree_nodes` | Explored belief nodes (VNodes), excluding action nodes. |
| `policy_nodes` | Stock greedy policy's expanded action nodes; implicit default-policy rollout is uncounted. Not memory in bytes. |
| `chosen_default` | Whether the root default policy beat all expanded action lower bounds. Stock policy size can then overstate the explicitly selected policy. |
| `trials`, `expanded_nodes`, `longest_trial_length`, gaps | Search effort diagnostics. Gaps concern sampled, regularized bounds, not certified true POMDP optimality. |

Tree and policy sizes use different node definitions. Online step metrics are
averaged within episodes, then across episodes, so longer episodes do not
dominate the mean.

At a fixed belief, let `n_a` be the number of repeats choosing action `a` and
`R` the repeat count. The main stability metric is:

```text
pairwise agreement = sum_a n_a (n_a - 1) / [R (R - 1)]
```

This estimates how often two independent planner seeds choose the same action.
Also report modal share, action entropy, and agreement with the largest-K mode.
Largest-K agreement is a finite comparison, not ground truth; self-comparison
is omitted and reference ties use the smallest action index. High stability
does not imply high reward, and near-equivalent actions can alternate.

Bootstrap intervals resample episode seeds. Stability intervals resample whole
reference-trajectory seeds, keeping dependent probes together. Three seeds give
weak uncertainty estimates: the pilot validates measurements and shows initial
trends; it does not establish a sufficient K.

## Randomness and timing

Separate seed domains control world initialization/transitions, belief
initialization/filtering, and planning. The wrapper resets `Random::RANDOM`,
`Seeds::root_seed`, and C randomness. Planning
does not consume future world/filter randomness. All K values use the same
repeat seed schedule; different repeats use different seeds. Stochastic SMART
rollout choices also contribute to measured planner variability.

Stock sampling is systematic resampling followed by a shuffle. On macOS,
libc++'s `random_shuffle` uses an internal random source that `srand` does not
reset. The experiment belief subclass restores a known initial ordering and
uses an explicitly seeded Fisher-Yates permutation, retaining the systematic
resampling rule and Pocman's belief-update behavior. This changes shuffle
implementation locally; it does not modify the library. Matching seeds
across K does **not** make sampled states nested prefixes. This tests the
implementation's response to K, not an IID nested-scenario estimator. Fresh
streams avoid normal `Search()`'s extra first-call static stream construction;
tree construction and action selection still call the existing DESPOT methods.

The CPU budget is approximate and applies only to the search loop. Root bound
initialization and whole trials can exceed it. Larger K can leave fewer trials
under the same budget. Inspect total latency, trial counts, and residual gaps
together with return. A plateau under one budget does not establish sampling
convergence. Time-limited runs can change actions despite identical seeds;
compiler/platform differences also matter. The manifest records settings,
commands, platform, binary hash, and source hashes.

## Choosing K

Before a larger study, specify a meaningful reward tolerance and stability
threshold. Seek the smallest K with acceptable reward relative to a well-resourced
comparison, sufficiently stable decisions at representative beliefs, and time
and size costs within budget. Inspect paired reward differences and uncertainty.
Check the result at a larger time budget and held-out seeds. Do not select a
universal K from a three-seed pilot or assume the largest tested K is optimal.

## Files

- `main.cpp`, `Makefile`: isolated, instrumented Pocman executable.
- `run.py`: build, serial sweep, manifest, CSV aggregation.
- `analyze.py`: summaries, uncertainty, stability, report, optional plots.
- `results/<run>/raw/`: per-job records and logs.
- `steps.csv`, `episodes.csv`, `probes.csv`: aggregate raw measurements.
- `summary.csv`, `stability.csv`, `report.md`: analyzed results.
- `analysis.json`: analysis source/input hashes, bootstrap settings, Python version.
- `overview.png`, `overview.svg`: four-panel figure with Matplotlib available.
