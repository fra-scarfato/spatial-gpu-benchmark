# Experiments

Each folder measures one part of the thesis. Its README says what the part shows and gives its
commands in order, ready to copy, with what they write and the tables that come out. The code they
use is outside: `measure/` runs on the GPU, `analysis/` reads the data, `probes/` holds the CUDA
programs.

| Folder | Thesis | Tables | GPU time |
|---|---|---|---|
| [`partitioning-mechanisms/`](partitioning-mechanisms/) | `results-section.tex`: green contexts against MPS; memory contention and its causes | `tab:iso-rates` ... `tab:mem-presence` | about 9 h |
| [`stressor/`](stressor/) | `sec:stress-stressor`: the choice of the stressor | `tab:stress-scale` ... `tab:stress-peak` | about 6 h |
| [`models/`](models/) | `sec:stress-models`, `sec:stress-bound`: three models, baselines, correction, maximum, margin | `tab:stress-temporal-stat`, `tab:counter-variants`, `tab:models-*`, `tab:bound-*`, `tab:stress-margin`, `tab:stress-pessimism` | about 5 h |
| [`composition/`](composition/) | `sec:stress-compose`: two co-runners together | `tab:stress-compose` | about 2 h |
| [`partition-sizes/`](partition-sizes/) | `sec:stress-sizes`, `sec:stress-triple`, `sec:stress-energy` | `tab:stress-sizes*`, `tab:stress-triple*`, `tab:stress-energy*` | about 12 h |
| [`periodic-loads/`](periodic-loads/) | not yet in the thesis | none yet | about 64 h |

The times leave out the profiled runs (traces, DRAM counters, Nsight Compute), which take minutes
per set; `measure/profile.sh <set> --list` prints a set's runs.

## The order

Run the parts in the order of the table, or one part after the parts it needs:

| Part | Needs | For |
|---|---|---|
| partitioning-mechanisms | nothing | |
| stressor | nothing | |
| composition | nothing | |
| models | stressor | the twelve pairs and the targets' smooth curves |
| partition-sizes | models (and so stressor), composition | the profiles and the coefficients fitted on the twelve pairs; the factor f |
| periodic-loads | partition-sizes | the networks alone on the whole GPU (energy-tenants), the budgets |

Two constants of the stress chapter come from `partitioning-mechanisms`: the cost of a kernel
boundary (1.05 us) and the kernels per inference of the four networks, both in `tab:l2-presence`
and written in `analysis/counter-model.py` (`PRESENCE_US`, `ENG`).

## Before any part

- Run every command from the repository root, with `DATA` exported, after the build and the clock
  lock of the top [`README.md`](../README.md).
- The measuring scripts check the clocks, the GPU's other users and the binaries' hashes, and refuse
  to run otherwise. They are resumable: a run already on disk is skipped, so an interrupted command
  is simply started again. To measure anew, start from a `DATA` that holds only `models/` and
  `dataset/`.
- The probe's calibration and the readings compared with it must come from one boot: the probe's
  latency with nothing running moves between boots. Run each part's probe commands
  (`reporter.sh`) without rebooting in between; after a reboot, delete that part's probe directory
  and run them again.
- Predictions are frozen before their pairs are measured: each part runs its `predict` step before
  the pairs' sweep, and the triple step refuses to run once the triples exist. `make check` compares
  the frozen files with the thesis's (`analysis/frozen.sha256`); on data measured anew they differ,
  as they should.
- A sweep configuration is five runs of at least 30 s and 5000 inferences, about 3 minutes; a
  probe configuration is five runs of 30 s, about 3.5 minutes.

Earlier stressors and checks of the setup that did not enter the thesis, and every script as it
was first run, are in the git history (commit `2d2a202`).
