#!/usr/bin/env python3
"""int8-calib.py - the INT8 calibration cache of a network, and a check of the engine built from it.

    python3 measure/int8-calib.py calib <model.onnx> <cache>          write the calibration cache
    python3 measure/int8-calib.py check <fp16.engine> <int8.engine>   INT8 against FP16 outputs

engines.mk builds the INT8 engines with trtexec --int8 --fp16 --calib=<cache>, the same tool and
flags as the FP16 ones plus INT8, so this script only produces the scales. TensorRT computes them
while building an engine with a calibrator attached; that engine is thrown away. trtexec alone
cannot calibrate: without a cache it uses placeholder scales, right for timing but not for the
outputs, and the quantization work that follows needs real ones (agreed with the user, 2026-10-13).

Calibration data: 500 ImageNet validation images, the sorted files 1000-1499. The runner's fixed set
is the first 32 sorted files, so no measured input is a calibration input. Each image goes through
the runner's preprocessing (src/inputs.cpp, reproduced below): resize, centre crop, BGR to RGB,
ImageNet mean and standard deviation, NCHW. RetinaNet, a COCO detector with a 480x640 input, gets
the same ImageNet images through the same steps; detection accuracy is not measured in this work.
The calibrator is TensorRT's entropy calibrator (IInt8EntropyCalibrator2), the one trtexec and the
TensorRT samples use for CNNs. TensorRT 10 deprecates implicit quantization in favour of Q/DQ nodes
inserted by ModelOpt, which is not installed on the board: the calibrator is the path available here.

check runs both engines on 200 other images (files 2000-2199) and prints the top-1 agreement of the
first output (meaningful for the classifiers) and the cosine similarity of all outputs.

Device memory goes through libcudart via ctypes: neither pycuda nor cuda-python is on the board, and
installing is not allowed there.
"""
import ctypes
import os
import sys

import cv2
import numpy as np
import tensorrt as trt

DATA = os.environ.get("DATA", "/home/fscarfato/data")
DS = os.path.join(DATA, "dataset", "imagenet-1k")
CALIB = slice(1000, 1500)
CHECK = slice(2000, 2200)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # include/config.hpp
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_SIZE, RESIZE_SHORT = 224, 256
LOG = trt.Logger(trt.Logger.WARNING)
CUDART = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
H2D, D2H = 1, 2


def dmalloc(nbytes):
    p = ctypes.c_void_p()
    assert CUDART.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0, "cudaMalloc"
    return p.value


def copy(dst, src, nbytes, kind):
    assert CUDART.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), kind) == 0


def images(sl):
    return [os.path.join(DS, f) for f in sorted(os.listdir(DS))[sl]]


def preprocess(path, height, width):
    """The runner's inputs::preprocess, step for step, in float32 like the C++ (1 x 3 x H x W)."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    target = (min(height, width) * RESIZE_SHORT + IMAGE_SIZE // 2) // IMAGE_SIZE
    need = max(np.float32(height) / np.float32(h), np.float32(width) / np.float32(w))
    scale = max(np.float32(target) / np.float32(min(h, w)), need)
    img = cv2.resize(img, (int(np.float32(w) * scale), int(np.float32(h) * scale)))
    y0, x0 = (img.shape[0] - height) // 2, (img.shape[1] - width) // 2
    img = cv2.cvtColor(img[y0:y0 + height, x0:x0 + width], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1)[None])


class Calibrator(trt.IInt8EntropyCalibrator2):
    """Feeds the calibration images one at a time; TensorRT asks until get_batch returns None."""

    def __init__(self, files, shape, cache):
        super().__init__()
        self.files, self.shape, self.cache, self.i = files, shape, cache, 0
        self.dev = dmalloc(int(np.prod(shape)) * 4)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i == len(self.files):
            return None
        x = preprocess(self.files[self.i], *self.shape[2:])
        copy(self.dev, x.ctypes.data, x.nbytes, H2D)
        self.i += 1
        return [self.dev]

    def read_calibration_cache(self):
        return None         # always calibrate: a stale cache would silently survive a new model

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as fh:
            fh.write(bytes(cache))


def calib(onnx, cache):
    builder = trt.Builder(LOG)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, LOG)
    # parse_from_file, not parse: Inception keeps its weights in an external .data file beside it
    assert parser.parse_from_file(onnx), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    inp = net.get_input(0)
    shape = tuple(1 if d < 0 else d for d in inp.shape)     # a symbolic batch is calibrated at 1
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if any(d < 0 for d in inp.shape):
        prof = builder.create_optimization_profile()
        prof.set_shape(inp.name, shape, shape, shape)
        cfg.add_optimization_profile(prof)
        cfg.set_calibration_profile(prof)
    cfg.int8_calibrator = Calibrator(images(CALIB), shape, cache)
    print("calibrating %s, input %s %s, %d images" % (os.path.basename(onnx), inp.name, shape, CALIB.stop - CALIB.start))
    assert builder.build_serialized_network(net, cfg) is not None, "build failed"
    print("wrote", cache)


class Engine:
    """One engine with every I/O tensor bound to its own device buffer, batch 1."""

    def __init__(self, path):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            inp = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if inp:
                self.ctx.set_input_shape(name, tuple(1 if d < 0 else d for d in self.engine.get_tensor_shape(name)))
            self.io.append((name, inp))
        self.buf = {}
        for name, inp in self.io:
            a = np.empty(tuple(self.ctx.get_tensor_shape(name)), trt.nptype(self.engine.get_tensor_dtype(name)))
            self.buf[name] = (a, dmalloc(max(a.nbytes, 1)))
            self.ctx.set_tensor_address(name, self.buf[name][1])

    def input_hw(self):
        return self.buf[next(n for n, i in self.io if i)][0].shape[2:]

    def __call__(self, x):
        a, d = self.buf[next(n for n, i in self.io if i)]
        copy(d, x.ctypes.data, x.nbytes, H2D)
        assert self.ctx.execute_async_v3(0)
        assert CUDART.cudaDeviceSynchronize() == 0
        out = []
        for name, inp in self.io:
            if not inp:
                a, d = self.buf[name]
                copy(a.ctypes.data, d, a.nbytes, D2H)
                out.append(a.astype(np.float64).ravel().copy())
        return out


def check(fp16, int8):
    a, b = Engine(fp16), Engine(int8)
    agree, cos = [], []
    for f in images(CHECK):
        x = preprocess(f, *a.input_hw())
        oa, ob = a(x), b(x)
        agree.append(oa[0].argmax() == ob[0].argmax())
        va, vb = np.concatenate(oa), np.concatenate(ob)
        cos.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
    print("%s against %s, %d images: top-1 agreement %.1f %%, cosine median %.4f, min %.4f"
          % (os.path.basename(int8), os.path.basename(fp16), len(cos), 100 * np.mean(agree), np.median(cos), min(cos)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calib" and len(sys.argv) == 4:
        calib(sys.argv[2], sys.argv[3])
    elif cmd == "check" and len(sys.argv) == 4:
        check(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__.split("\n\n")[1])
