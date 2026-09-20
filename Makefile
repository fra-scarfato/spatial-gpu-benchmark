# Makefile for the inference benchmark.
#
# Layout, all relative to this directory:
#
#   include/     headers
#   src/         sources
#   models/      TensorRT engines and the source ONNX files
#   dataset/     the ImageNet validation images
#   benchmark/   RESULTS: <PREFIX>_latencies.csv and <PREFIX>_stats.csv
#
# Usage:
#   make                 build ./runner
#   make run             development run (fast, but the numbers are meaningful)
#   make full            the protocol, fixed-input mode      [same as full-fixed]
#   make full-fixed      the protocol, fixed-input mode
#   make full-dataset    the protocol, end-to-end dataset mode
#   make clean           remove the binary
#
# Parametrized; override any of these on the command line:
#
#   make full N=5000 T=30 FIXED=32 PREFIX=baseline
#   make full ENGINE=$(MODELS)/resnet50-v2-7.engine PREFIX=resnet50
#   make full-dataset N=5000 T=30 PREFIX=endtoend
#
#   FIXED     size of the fixed preprocessed input set (default 32)
#   N, T      run-length bounds: at least N inferences AND at least T seconds
#   DEV_N     inferences for `make run` (default 1000)
#   PREFIX    output name -> benchmark/<PREFIX>_latencies.csv
#
# Output files:
#   make run           -> benchmark/dev_*
#   make full          -> benchmark/<PREFIX>_*
#   make full-dataset  -> benchmark/<PREFIX>-dataset_*
#
# Before measuring: `sudo jetson_clocks`, and check that the GPU reports
# min_freq == max_freq. Otherwise the power governor throttles the GPU and
# inflates the numbers about 3.4x. See README.md.

CXX      := g++
CXXFLAGS := -O2 -std=c++20 -Wall
INCDIR   := include
SRCDIR   := src
TARGET   := runner

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
HEADERS  := $(wildcard $(INCDIR)/*.hpp)

# System paths. These stay absolute because that is where the toolchain lives
# on this Jetson; only the project paths below are relative.
#
#   - TensorRT headers are in the cross-compile include dir
#     (/usr/include/aarch64-linux-gnu) — that is where NvInfer.h is.
#   - OpenCV 4 headers are in /usr/include/opencv4. (pkg-config claims
#     /usr/local/include/opencv4, which does not exist here.)
#   - CUDA libs are under /usr/local/cuda/targets/sbsa-linux/lib
#     (sbsa = ARM server base architecture).
#
# They are explicit rather than discovered on purpose: this is a shared lab
# machine, and the build must not depend on an environment variable that
# another user forgot to export.
#
# OpenCV is pulled in with -isystem rather than -I so that its headers count as
# system headers and their warnings are not printed. Under C++20, OpenCV 4.8
# emits a few hundred -Wdeprecated-enum-enum-conversion warnings of its own (it
# predates that rule), which would otherwise bury any warning from our code.
INCLUDES := -I$(INCDIR) \
            -I/usr/include/aarch64-linux-gnu \
            -isystem /usr/include/opencv4 \
            -I/usr/local/cuda/include
LIBS     := -L/usr/local/cuda/targets/sbsa-linux/lib \
            -lnvinfer -lcudart \
            -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

# Project paths, relative to this directory.
MODELS   := models
DATASET  := dataset/imagenet-1k
RESULTS  := benchmark
ENGINE   := $(MODELS)/mobilenetv2-12.engine

# Defaults, overridable on the command line.
FIXED    ?= 32
N        ?= 5000
T        ?= 30
DEV_N    ?= 1000
PREFIX   ?= baseline

.PHONY: all clean run full full-fixed full-dataset

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)

# Development run: fixed inputs, no time floor, DEV_N inferences.
# 1000 samples is already converged (median, std and p99 match a 20,000-sample
# run to within ~0.3%), and it takes under a second, so it can be run after
# every edit. Results go to benchmark/dev_* so they never clobber a real run.
run: $(TARGET)
	@mkdir -p $(RESULTS)
	./$(TARGET) $(ENGINE) $(DATASET) \
		--fixed $(FIXED) --min-inferences $(DEV_N) --min-duration 0 \
		--csv $(RESULTS)/dev

# The phase-1 protocol: fixed preprocessed inputs, so the timed loop is pure
# GPU work and the engine is measured in isolation.
full: full-fixed

full-fixed: $(TARGET)
	@mkdir -p $(RESULTS)
	./$(TARGET) $(ENGINE) $(DATASET) \
		--fixed $(FIXED) --min-inferences $(N) --min-duration $(T) \
		--csv $(RESULTS)/$(PREFIX)

# Same bounds, but decoding and preprocessing happen inside the loop: the
# end-to-end pipeline. Note that the bounds stop the run as soon as they are
# satisfied, so this is NOT a pass over the whole dataset unless N and T are
# large enough to exhaust the pool.
full-dataset: $(TARGET)
	@mkdir -p $(RESULTS)
	./$(TARGET) $(ENGINE) $(DATASET) \
		--end-to-end --min-inferences $(N) --min-duration $(T) \
		--csv $(RESULTS)/$(PREFIX)-dataset

clean:
	rm -f $(TARGET)
