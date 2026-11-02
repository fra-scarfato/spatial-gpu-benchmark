# engines.mk — every TensorRT engine this work measures, rebuilt from its ONNX.
#
#   make -f engines.mk              every engine
#   make -f engines.mk census       what each built engine reports
#   make -f engines.mk clean        remove the engines, keep the build logs
#
# Run it from the runner directory. The model tree is one level up, beside the
# repository, and is deliberately not in it.
#
# Two things that are not obvious, and are most of the reason this file exists:
#
#   * Every build passes --fp16. That is permission, not instruction: TensorRT uses
#     half precision where it judges it fastest and leaves the I/O bindings FP32.
#     Asking an engine layer by layer confirms it: FP16 tensors, weights and kernel
#     tactics throughout, with only the input binding FP32.
#
#   * A dynamic-shape ONNX has to be told the shape. The ViT exports with all four
#     dimensions symbolic, and TensorRT resolves a symbolic channel dimension to
#     ONE, so the patch-embedding convolution is given 1 input channel where its
#     weights have 3, and the build dies with a weight-count mismatch that reads
#     like a corrupt file. It is not corrupt: --shapes fixes it. RetinaNet's graph
#     is static, so the same trick does not apply to it.
#
# Engines are not portable across TensorRT versions, and builds are not
# reproducible: two builds with identical flags produced 75, 76 and 77 layer plans
# for ResNet. Rebuilding does not give byte-identical engines, which is why results
# are compared within a build and never across one.

TRTEXEC ?= /usr/src/tensorrt/bin/trtexec
# The graphs and the engines are data, so they are on the NVMe with the rest of it.
DATA     ?= /mnt/nvme/f.scarfato
MODELS   ?= $(DATA)/models
AUXPROBE ?= probes/auxprobe
FP16     = --fp16

MB  = $(MODELS)/mobilenetv2-12
RS  = $(MODELS)/resnet50-v2-7
INC = $(MODELS)/inception_v3-onnx-float
VIT = $(MODELS)/vit-base-patch16-224
RN  = $(MODELS)/retinanet-9
EFF = $(MODELS)/efficientnet_b0

.PHONY: all mobile resnet inception vit retinanet efficientnet int8 census clean

all: mobile resnet inception vit retinanet efficientnet int8

# MobileNet and ResNet: the two networks of the first campaign. Every cap produces
# a serial engine for both, which is itself the census result.
mobile:
	$(TRTEXEC) --onnx=$(MB)/mobilenetv2-12.onnx $(FP16) --saveEngine=$(MB)/engine/mobilenetv2-12.engine > $(MB)/engine/mobilenetv2-12.buildlog 2>&1
	for n in 0 1 2 4; do \
		$(TRTEXEC) --onnx=$(MB)/mobilenetv2-12.onnx $(FP16) --maxAuxStreams=$$n --saveEngine=$(MB)/engine/mobilenetv2-12-aux$$n.engine > $(MB)/engine/mobilenetv2-12-aux$$n.buildlog 2>&1; \
	done

resnet:
	$(TRTEXEC) --onnx=$(RS)/resnet50-v2-7.onnx $(FP16) --saveEngine=$(RS)/engine/resnet50-v2-7.engine > $(RS)/engine/resnet50-v2-7.buildlog 2>&1
	for n in 0 1 2 4; do \
		$(TRTEXEC) --onnx=$(RS)/resnet50-v2-7.onnx $(FP16) --maxAuxStreams=$$n --saveEngine=$(RS)/engine/resnet50-v2-7-aux$$n.engine > $(RS)/engine/resnet50-v2-7-aux$$n.buildlog 2>&1; \
	done

# Inception-v3: the network that uses streams. It adopts as many as it is allowed
# up to five, and one is the latency optimum, so the caps go to 16 to show where
# the allowance stops being used.
inception:
	$(TRTEXEC) --onnx=$(INC)/inception_v3.onnx $(FP16) --saveEngine=$(INC)/engine/inception_v3.engine > $(INC)/engine/inception_v3.buildlog 2>&1
	for n in 0 1 2 4 5 8 16; do \
		$(TRTEXEC) --onnx=$(INC)/inception_v3.onnx $(FP16) --maxAuxStreams=$$n --saveEngine=$(INC)/engine/inception_v3-aux$$n.engine > $(INC)/engine/inception_v3-aux$$n.buildlog 2>&1; \
	done

# The ViT is a serial chain — the control, not a subject — and it is the one graph
# whose shape must be supplied, because its export left every dimension symbolic.
vit:
	$(TRTEXEC) --onnx=$(VIT)/vit-base-patch16-224.onnx $(FP16) --shapes=pixel_values:1x3x224x224 --saveEngine=$(VIT)/engine/vit-base-patch16-224.engine > $(VIT)/engine/vit-base-patch16-224.buildlog 2>&1

# RetinaNet-9 is a detector: a 480x640 input, fixed by the graph, and ten output
# tensors. It adopts more streams than anything else here, and the runner honours
# both of those -- the input geometry comes from the engine, and every output is
# bound rather than only the last one.
retinanet:
	$(TRTEXEC) --onnx=$(RN)/retinanet-9.onnx $(FP16) --saveEngine=$(RN)/engine/retinanet-9.engine > $(RN)/engine/retinanet-9.buildlog 2>&1
	for n in 1 8; do \
		$(TRTEXEC) --onnx=$(RN)/retinanet-9.onnx $(FP16) --maxAuxStreams=$$n --saveEngine=$(RN)/engine/retinanet-9-aux$$n.engine > $(RN)/engine/retinanet-9-aux$$n.buildlog 2>&1; \
	done

# EfficientNet-B0 (2026-10-13): a validation network, a CNN like the four above (the
# easy test). timm's export from the ONNX model zoo; SOURCE.txt beside it has the commit
# and the checksum.
efficientnet:
	$(TRTEXEC) --onnx=$(EFF)/efficientnet_b0_Opset17.onnx $(FP16) --saveEngine=$(EFF)/engine/efficientnet_b0.engine > $(EFF)/engine/efficientnet_b0.buildlog 2>&1

# INT8 versions of the four networks (2026-10-13): new workloads for the validation, with
# less memory traffic and other kernels, and the start of the quantization work. INT8 is
# permission on top of --fp16, as FP16 is above: TensorRT picks per layer. The scales
# come from a real calibration (measure/int8-calib.py, 500 ImageNet images the runner
# never feeds); trtexec alone would use placeholder scales and meaningless outputs.
int8:
	for m in $(MB)/mobilenetv2-12 $(RS)/resnet50-v2-7 $(INC)/inception_v3 $(RN)/retinanet-9; do \
		d=$$(dirname $$m); n=$$(basename $$m); \
		python3 measure/int8-calib.py calib $$m.onnx $$d/engine/$$n-int8.calib > $$d/engine/$$n-int8.caliblog 2>&1 && \
		$(TRTEXEC) --onnx=$$m.onnx $(FP16) --int8 --calib=$$d/engine/$$n-int8.calib --saveEngine=$$d/engine/$$n-int8.engine > $$d/engine/$$n-int8.buildlog 2>&1 && \
		python3 measure/int8-calib.py check $$d/engine/$$n.engine $$d/engine/$$n-int8.engine >> $$d/engine/$$n-int8.caliblog 2>&1 || exit 1; \
	done

# What every built engine says about itself. This is the census, and it is the
# reason to keep the engines: it reads them, it does not rebuild them.
census:
	@for e in $(MB)/engine/*.engine $(RS)/engine/*.engine $(INC)/engine/*.engine $(VIT)/engine/*.engine $(RN)/engine/*.engine $(EFF)/engine/*.engine; do \
		[ -e "$$e" ] || continue; \
		printf '  %-46s %s\n' "$$(basename $$e)" "$$($(AUXPROBE) $$e 2>/dev/null | tail -1)"; \
	done

# The logs are the census evidence and stay; they are a few tens of kilobytes.
clean:
	rm -f $(MB)/engine/*.engine $(RS)/engine/*.engine $(INC)/engine/*.engine $(VIT)/engine/*.engine $(RN)/engine/*.engine $(EFF)/engine/*.engine
