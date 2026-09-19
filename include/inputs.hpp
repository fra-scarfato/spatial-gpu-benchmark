#pragma once
// =====================================================================
// inputs.hpp — where the input tensors come from.
//
// Two modes, one interface:
//
//   fixed    A small set of images is decoded and preprocessed ONCE, at
//            construction, into pinned host memory. The timed loop then
//            cycles those tensors. No disk, no decode, no preprocessing
//            inside the measurement — so the timed loop is pure GPU
//            work, and every configuration sees byte-identical inputs.
//            This is the mode for the performance characterization.
//
//   dataset  Each iteration decodes and preprocesses an image. This is
//            the end-to-end pipeline, kept for later phases (real
//            pipeline behaviour, accuracy).
//
// It depends on gpu.hpp only for PinnedHostBuffer: the tensors handed to
// the GPU must be page-locked, and pinned memory is a CUDA concept that
// lives in the GPU layer. Note that this file makes no CUDA call of its
// own -- every CUDA call in the program is in gpu.cpp.
// =====================================================================

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "gpu.hpp"

namespace inputs {

// ---------------------------------------------------------------------
// listImages(): the .JPEG files in a directory, SORTED.
//
// Sorting is not cosmetic. ImageNet validation ground truth is defined
// in filename order, so the i-th image processed must correspond to the
// i-th label. Sorting also makes runs deterministic and therefore
// directly comparable: the same sequence of images on every run.
// ---------------------------------------------------------------------
std::vector<std::string> listImages(const std::string& dir);

// ---------------------------------------------------------------------
// preprocess(): the ImageNet input contract, BGR cv::Mat -> NCHW float.
//
//   1. Resize so the shortest side is cfg::resizeShortSide (256).
//   2. Center-crop cfg::imageSize x cfg::imageSize (224).
//   3. Convert to float and scale to [0, 1] (divide by 255).
//   4. Normalize per channel: (v - mean) / std, with the PyTorch
//      training constants in config.hpp. Do not "improve" them.
//   5. Lay out in NCHW (channel first). TensorRT wants
//      [batch][channel][height][width]; give it NHWC and it will
//      happily produce garbage while looking completely healthy.
//
// Why 256 then 224? Because the training pipeline did
// resize-to-256-then-crop, so the model expects it. OpenCV loads BGR;
// the network wants RGB.
//
// `dst` must have room for cfg::channels * imageSize * imageSize floats.
// ---------------------------------------------------------------------
void preprocess(const cv::Mat& bgr, float* dst);

// ---------------------------------------------------------------------
// InputSet — the set of tensors the timed loop draws from.
//
// `at(i)` returns a pointer to the tensor for iteration i: fixed mode
// just indexes (no CPU work); dataset mode pays the full decode and
// preprocess cost on every call.
//
// Contract: the returned pointer is valid until the next call to at().
// ---------------------------------------------------------------------
class InputSet {
public:
    InputSet(const Options& opts,
             const std::vector<std::string>& images,
             std::size_t tensorSize);          // floats per input (= engine inputVolume)

    const float* at(std::size_t i);

    std::string idOf(std::size_t i) const;     // source filename, for the CSV
    std::size_t count() const noexcept;        // number of distinct inputs
    bool        isFixed() const noexcept { return fixed_; }

private:
    bool        fixed_;
    std::size_t tensorSize_;

    // fixed mode
    PinnedHostBuffer         buffers_;         // nFixed_ * tensorSize_ floats
    std::size_t              nFixed_ = 0;
    std::vector<std::string> names_;

    // dataset mode
    std::vector<std::string> images_;
    PinnedHostBuffer         scratch_;         // tensorSize_ floats, reused
};

} // namespace inputs
