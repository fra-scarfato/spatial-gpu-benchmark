# Makefile for the inference benchmark.
#
#   make          build ./runner
#   make clean    remove the binary
#
# That is the whole interface. The targets that used to run a single
# configuration here (run, full, full-fixed, full-dataset) have been removed:
# they duplicated what sweep.sh does, with different defaults; they pointed at
# models/ and dataset/ as if those lived inside this repository, which they do
# not; and they were a second place that had to be kept in step with the
# protocol. To run one configuration by hand, take the command from
#
#   ./sweep.sh --exp <family> --dry-run
#
# which prints exactly what it would execute.
#
# Layout: the sources are here (include/, src/, utility/) and so is the binary.
# The data and the results are NOT: models/, dataset/, labels/ and bench/ are on
# the NVMe, in /mnt/nvme/f.scarfato, because the root filesystem is small and shared.
# sweep.sh is what tells the runner where they are, and it takes that root from $DATA.

CXX      := g++
CXXFLAGS := -O2 -std=c++20 -Wall
INCDIR   := include
SRCDIR   := src
TARGET   := runner

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
# The one CUDA source, the spin participant's kernel, is compiled by nvcc for the Orin (sm_87)
# and linked like any other object.
NVCC     := /usr/local/cuda/bin/nvcc
CUOBJS   := $(patsubst %.cu,%.o,$(wildcard $(SRCDIR)/*.cu))
HEADERS  := $(wildcard $(INCDIR)/*.hpp)

# System paths. Absolute on purpose: this is a shared lab machine, and the build
# must not depend on an environment variable that somebody forgot to export.
#
#   - TensorRT headers are in /usr/include/aarch64-linux-gnu (NvInfer.h lives
#     there, not in the CUDA tree).
#   - OpenCV 4 headers are in /usr/include/opencv4 (pkg-config claims
#     /usr/local/include/opencv4, which does not exist here).
#   - CUDA libraries are under /usr/local/cuda/targets/sbsa-linux/lib, sbsa
#     being the ARM server base architecture.
#
# OpenCV is pulled in with -isystem rather than -I so that its headers count as
# system headers. Under C++20, OpenCV 4.8 emits a few hundred of its own
# -Wdeprecated-enum-enum-conversion warnings, which would otherwise bury any
# warning from our code.
INCLUDES := -I$(INCDIR) \
            -I/usr/include/aarch64-linux-gnu \
            -isystem /usr/include/opencv4 \
            -I/usr/local/cuda/include
LIBS     := -L/usr/local/cuda/targets/sbsa-linux/lib \
            -lnvinfer -lcudart -lcublas \
            -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS) $(CUOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(CUOBJS) -o $(TARGET) $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -O2 -arch=sm_87 -c $< -o $@

clean:
	rm -f $(TARGET) $(CUOBJS)

# The analysis self-tests: each script checks its own arithmetic on synthetic input (interpolation,
# fits, folds, parsing), with no GPU. A new script with a selftest goes in this list.
SELFTESTS = utility/stress.py utility/analysis/compose.py utility/analysis/counter-model.py \
            utility/analysis/ensemble.py utility/analysis/geometry.py utility/analysis/infwin-model.py \
            utility/analysis/l2-residual.py

check:
	@set -e; for t in $(SELFTESTS); do echo "== $$t"; python3 $$t selftest; done
