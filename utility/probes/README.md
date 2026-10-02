# utility/probes

Four small CUDA programs that answer questions the runner cannot answer about itself. They are
built by their own Makefile, so rebuilding the runner never invalidates a probe and the reverse.

| program | what it does | used by |
|---|---|---|
| `auxprobe` | Deserialises an engine and prints how many auxiliary streams it actually uses (`ICudaEngine::getNbAuxStreams()`), which can be fewer than the build allowed. No inference. | `make -f engines.mk census` |
| `wqprobe` | Prints the device's default work-queue configuration, then creates one green context per sharing scope and reads back what each was granted. | the work-queue subsection |
| `devprobe` | Prints the device facts the memory experiments are sized on (L2 size and others) and measures device-to-device copy bandwidth across working-set sizes. | `profile.sh calibration`: its copies are the GB/s calibration of the DRAM counters (see `MEASUREMENTS.md`) |
| `itfproc` | The synthetic co-runner (`mem_saxpy`, `compute_gemm`) as a separate process, for the MPS arms where a co-runner has to be its own client. | `lib/matrix.sh`, the `memory-mps-*` families |

## Build

    make -C utility/probes
    make -C utility/probes clean

The paths at the top of the Makefile are this board's (aarch64, sbsa libraries). Elsewhere,
override them, for example:

    make -C utility/probes TRT_INC=/usr/include/x86_64-linux-gnu TRT_LIB=/usr/local/cuda/lib64

## Use

    utility/probes/auxprobe /path/to/one.engine /path/to/another.engine
    utility/probes/wqprobe
    utility/probes/devprobe
    utility/probes/itfproc compute_gemm --reps 32      # normally started by sweep.sh

`auxprobe` and `wqprobe` run no kernel, but they allocate device memory, so none of the four
belongs inside another client's timed window.
