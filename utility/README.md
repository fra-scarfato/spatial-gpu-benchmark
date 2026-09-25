# utility/

Standalone programs that answer questions the runner cannot answer about itself.
They live inside the runner now, but they are built by their own Makefile, so the two
runner does not invalidate a measurement tool, and a tool can be built on a
machine where the runner cannot.

| file | what it answers |
|---|---|
| `auxprobe.cpp` | **How many auxiliary streams does a built engine use?** Deserializes an engine and prints `ICudaEngine::getNbAuxStreams()` — how many the engine *actually* uses, not the maximum it was allowed at build time, which is a different number and the one that matters for choosing a work queue concurrency. No inference, no timing: a few seconds per engine. |
| `wqprobe.cpp` | **What does the platform report for work queues, and what does a green context actually get granted?** Prints the device's default configuration, then creates one green context per sharing scope and reads back what each was provisioned with. This is what established that the default is 8 in a plain process and 2 under MPS, and that the two scopes read back distinguishably. |
| `streams.py` | **Do two streams actually overlap?** Reads an nsys SQLite export and prints per-stream kernel count, busy time and span, plus the cross-stream overlap as a percentage of the time they jointly occupy. A stream count alone proves nothing: high busy time with near-zero overlap is serialisation. |

## Build

    make
    make clean

Needs the CUDA headers, and for `auxprobe` the TensorRT headers, in the paths
named at the top of the Makefile. Those are this board's; override them
elsewhere, for example:

    make TRT_INC=/usr/include/x86_64-linux-gnu TRT_LIB=/usr/local/cuda/lib64

## Use

    ./auxprobe /path/to/one.engine /path/to/another.engine
    ./wqprobe
    nsys export --type sqlite --output /tmp/r.sqlite /tmp/r.nsys-rep
    python3 streams.py /tmp/r.sqlite

## Notes

- `auxprobe` and `wqprobe` touch the GPU only to deserialize engines and to
  create green contexts. Neither runs a kernel, so neither perturbs a
  measurement in the way a profiler does — but both allocate device memory, so
  they still do not belong inside somebody's timed window.
- `streams.py` reads an export rather than the board, so it runs anywhere Python
  runs, including on a workstation holding a copied report.
- These are diagnostic tools. No number any of them prints belongs in a results
  table: the runner's own CSVs are the measurements, and the profiler perturbs
  what it observes.
