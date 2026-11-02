# Diagnostics

Checks of the setup rather than results, like the pitfalls of a benchmark suite: an MPS hang, the
GPU governor, the probe's floor after a boot and its two states, and the memory controller's state.
They decided how the other queues run (the governor stays `performance`, `GPU_GOVERNOR` in
`measure/lib/guards.sh`; the floor is never used as an anchor), and no thesis table reads them.

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-06 | `mps-hang-repro.sh` | Reproduction of an MPS hang (Inception + RetinaNet), with gdb backtraces; no data |
| 10-06 | `mps-hang-repro-b.sh` | The same, second variant; no data |
| 10-09 | `gpu-governor-ab.sh` | GPU governor A/B (nvhost_podgov against performance) on a probe set |
| 10-09 | `gpu-governor-ab-gate.sh` | Stopped the Burst-scrambled queue after its diagnostic, to run the governor A/B |
| 10-10 | `probe-floor-after-boot.sh` | The probe's floor right after a boot |
| 10-10 | `probe-floor-two-states.sh` | What flips the probe's floor between its two states |
| 10-10 | `memory-controller-snapshot.sh` | A read-only debugfs snapshot of the memory controller, for the two floor states |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

`memory-controller-snapshot.sh` reads debugfs (root, the user's step). The MPS-hang queues produce
backtraces, not data. `gpu-governor-ab-gate.sh` stopped another queue: it is a record, not a step.
