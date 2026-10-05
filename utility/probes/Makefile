# utility/probes/ — tools that answer questions the runner cannot answer about itself.
#
# Standalone, and built independently of the runner, so that rebuilding the
# runner does not invalidate a measurement tool and vice versa. The paths below
# are for this board (aarch64 Linux, sbsa libraries); override TRT_INC and
# TRT_LIB on another machine.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall
CUDA     ?= /usr/local/cuda
TRT_INC  ?= /usr/include/aarch64-linux-gnu
TRT_LIB  ?= $(CUDA)/targets/sbsa-linux/lib

BINS = auxprobe wqprobe devprobe itfproc gpu-latency/gpu-latency

all: $(BINS)

# Links TensorRT: it deserializes engines and asks them about their streams.
auxprobe: auxprobe.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDA)/include -I$(TRT_INC) $< -o $@ \
		-L$(TRT_LIB) -lnvinfer -lcudart

# CUDA only: it queries device resources and creates green contexts.
wqprobe: wqprobe.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDA)/include $< -o $@ -L$(TRT_LIB) -lcudart

# CUDA runtime only: reads the device's own numbers and measures copy bandwidth
# across working-set sizes, which is how the memory interferer gets sized.
devprobe: devprobe.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDA)/include $< -o $@ -L$(TRT_LIB) -lcudart

# The interferer as a separate client process, for the MPS arms. nvcc, because the throttled SAXPY
# is our own kernel; the other kinds use cuBLAS.
itfproc: itfproc.cu
	$(CUDA)/bin/nvcc -O2 -arch=sm_87 $< -o $@ -lcublas

# The stress reporter: RRZE-HPC's pointer chase, vendored with its licence (see gpu-latency/UPSTREAM).
gpu-latency/gpu-latency: gpu-latency/main.cu gpu-latency/gpu-error.h gpu-latency/dtime.hpp
	$(CUDA)/bin/nvcc -O3 -arch=sm_87 $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
