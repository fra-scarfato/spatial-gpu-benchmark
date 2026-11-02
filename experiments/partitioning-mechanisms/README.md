# The partitioning mechanisms

Green contexts against MPS (static, ATP, default) and no partitioning: the rate and dispersion of
each network alone and in pairs, the submitting thread, the work queues, and the memory contention
that remains between disjoint partitions, with its causes. Thesis: `results-section.tex`
(`sec:experiments`), every table from `tab:iso-rates` to `tab:mem-presence`.

Most of it was measured before the queues were kept: the families are run by `pipeline.py`, not by
a queue. The three queues here are the L2 and presence experiments of `sec:mem-causes`.

## Running it

```sh
./pipeline.py section --dry-run       # every family of the section, as commands
./pipeline.py section                 # measure, reduce, tables (the families: SECTION_FAMILIES in measure/sweep.sh)
./pipeline.py appendix                # the appendix's families
sudo ./measure/profile.sh soc-budget  # the DRAM counters (root, the user's step); README.md lists every set
```

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-10 | `l2-presence-gemm-read.sh` | `profile.sh kernels`; l2-presence, l2-gemm-sets, l2-read (exp19-21) |
| 10-10 | `saxpy-4sm-matched-levels.sh` | Steady SAXPY on 4 SMs: the levels matched to 22.3 and 8.0 GB/s |
| 10-10 | `l2-against-dram.sh` | Probe l2-vs-dram; l2-vs-dram (exp22) |

## Outputs

- Families: `$DATA/bench/<family>/` (the list in `analysis/lib/families.py`); the L2 queues add
  l2-presence, l2-gemm-sets, l2-read and l2-vs-dram (exp19-22). Profiles: `$DATA/profiles/<set>/`,
  reduced by `analysis/profiles.py` to `$DATA/profiles/derived/`.
- Tables: `analysis/tables/results-tables.py` (`./pipeline.py all --stage reduce,tables`) writes
  `$DATA/bench/tables/results.tex`. `tab:l2-presence` and `tab:l2-evidence` have no generator yet:
  their numbers are the exp19-22 rows of `$DATA/bench/aggregate/`.
- Used later: the cost per kernel of `tab:l2-presence` (1.05 us) is the presence term of the counter
  model (`PRESENCE_US` in `analysis/counter-model.py`).

## On another board

The arms are in `measure/lib/matrix.sh`, one line each, with the SM shares as MPS percentages or
green-context SM counts of the Orin's 16 SMs.
