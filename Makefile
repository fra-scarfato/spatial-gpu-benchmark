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
# Layout: the sources are here (include/, src/) and so is the binary.
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

.PHONY: all clean check tables

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS) $(CUOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(CUOBJS) -o $(TARGET) $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -O2 -arch=sm_87 -c $< -o $@

clean:
	rm -f $(TARGET) $(CUOBJS)

# The analysis self-tests: each script checks its own arithmetic on synthetic input (interpolation,
# fits, folds, parsing), with no GPU. A new script with a selftest goes in this list.
SELFTESTS = analysis/stress.py analysis/compose.py analysis/counter-model.py \
            analysis/ensemble.py analysis/geometry.py analysis/infwin-model.py \
            analysis/explore/l2-residual.py

check:
	@set -e; for t in $(SELFTESTS); do echo "== $$t"; python3 $$t selftest; done
	@echo "== the frozen files against analysis/frozen.sha256"; \
	    cd $(DATA)/bench/frozen && sha256sum --quiet -c $(CURDIR)/analysis/frozen.sha256 && echo "  unchanged"

# ------------------------------------------------------------------ the stress chapter's numbers
#
# `make tables` brings every derived CSV of the stress chapter up to date, in dependency order, and
# writes the thesis tables to $(TABLES)/{stress,ensemble,geometry}.tex; `make -n tables` lists
# what is stale without running it. Each rule names what its script reads: the raw runs, the
# profiles, the CSVs of earlier steps and the scripts, so a new run or a changed script re-runs
# exactly what depends on it.
#
# The frozen files, in $(FROZEN), are inputs, never targets: the predictions of stress.py predict and of the
# new-predict steps (stress-predictions-*.csv), the calibrations, curves and readings written with
# them (stress-calibration-*, stress-sensitivity-*, stress-corunners-*), and
# geometry-triple-predictions.csv. Re-freezing is a decision taken by hand
# (campaigns/README.md); a missing frozen file stops make with "No rule to make target".
#
# ponytail: every rule depends on all raw runs, not on its own families, so a run of any family
# re-runs the chain (about 4 minutes). Per-family prerequisites if that becomes slow.

DATA ?= /mnt/nvme/f.scarfato
export DATA
AGG := $(DATA)/bench/aggregate
FROZEN := $(DATA)/bench/frozen
TABLES := $(DATA)/bench/tables
RUNS := $(wildcard $(DATA)/bench/*/*/r*)
PROFILES := $(wildcard $(DATA)/profiles/derived/*.csv $(DATA)/profiles/ncu/*.csv)
AN := analysis
LOAD := analysis/aggregate.py analysis/lib/families.py analysis/stress.py $(RUNS)
COUNTER := $(AN)/counter-model.py $(LOAD) $(PROFILES)
GEOMS := onoff88 uniform88 smooth88 smoothx88 smoothnew
cmp = $(foreach n,$(1),$(AGG)/stress-comparison-$(n).csv)

tables: $(TABLES)/stress.tex $(TABLES)/ensemble.tex $(TABLES)/geometry.tex $(AGG)/summary.csv

$(AGG)/summary.csv: $(LOAD)
	python3 analysis/aggregate.py

$(call cmp,$(GEOMS)): $(AGG)/stress-comparison-%.csv: $(FROZEN)/stress-predictions-%.csv $(LOAD)
	python3 analysis/stress.py compare $*

$(call cmp,infwin88 infwin88p infwin88d infwin88pd) &: $(AN)/infwin-model.py $(AN)/avg-model.py $(call cmp,smooth88) $(LOAD)
	python3 $(AN)/infwin-model.py

$(call cmp,infwinnewp): $(FROZEN)/stress-predictions-infwinnewp.csv $(AN)/infwin-model.py $(AN)/avg-model.py $(LOAD)
	python3 $(AN)/infwin-model.py new-compare

$(AGG)/counter-model-coeffs.csv $(call cmp,counters-dram counters-ref counters-l2) &: $(COUNTER)
	python3 $(AN)/counter-model.py

$(call cmp,counters-new-dram counters-new-ref counters-new-l2) &: \
		$(foreach v,dram ref l2,$(FROZEN)/stress-predictions-counters-new-$(v).csv) $(COUNTER)
	python3 $(AN)/counter-model.py new-compare

$(call cmp,max-dram max-ref max-l2 maxc-dram maxc-ref maxc-l2) &: $(AN)/ensemble.py $(call cmp,smooth88 infwin88p) $(COUNTER)
	python3 $(AN)/ensemble.py

$(AGG)/compose-4sm.csv: $(AN)/compose.py $(LOAD)
	python3 $(AN)/compose.py

GEOMETRY := $(AN)/geometry.py $(AGG)/counter-model-coeffs.csv $(AGG)/compose-4sm.csv $(COUNTER)

$(AGG)/geometry-splits.csv: $(GEOMETRY)
	python3 $(AN)/geometry.py splits

$(AGG)/geometry-triple-compare-current.csv: $(FROZEN)/geometry-triple-predictions.csv $(GEOMETRY)
	python3 $(AN)/geometry.py triple-current

$(AGG)/geometry-power.csv: $(AN)/geometry.py $(LOAD)
	python3 $(AN)/geometry.py power

$(TABLES)/stress.tex: $(AN)/tables/stress-tables.py $(AN)/avg-model.py $(AGG)/compose-4sm.csv $(LOAD) \
		$(call cmp,onoff88 uniform88 smooth88 smoothx88 infwin88 infwin88p) \
		$(call cmp,counters-dram counters-ref counters-l2 counters-new-dram counters-new-ref counters-new-l2)
	@mkdir -p $(TABLES)
	python3 $(AN)/tables/stress-tables.py > $@.tmp && mv $@.tmp $@

$(TABLES)/ensemble.tex: $(AN)/tables/ensemble-tables.py $(AN)/ensemble.py \
		$(call cmp,smooth88 infwin88p smoothnew infwinnewp counters-new-l2) $(COUNTER)
	@mkdir -p $(TABLES)
	python3 $(AN)/tables/ensemble-tables.py > $@.tmp && mv $@.tmp $@

$(TABLES)/geometry.tex: $(AN)/tables/geometry-tables.py $(AGG)/geometry-splits.csv \
		$(AGG)/geometry-triple-compare-current.csv $(AGG)/geometry-power.csv $(GEOMETRY)
	@mkdir -p $(TABLES)
	python3 $(AN)/tables/geometry-tables.py > $@.tmp && mv $@.tmp $@
