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

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
