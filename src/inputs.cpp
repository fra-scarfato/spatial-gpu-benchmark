// =====================================================================
// inputs.cpp — dataset listing, ImageNet preprocessing, and the
// fixed-input set.
// =====================================================================

#include "inputs.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------
// listImages
// ---------------------------------------------------------------------
std::vector<std::string> inputs::listImages(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".JPEG")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ---------------------------------------------------------------------
// preprocess — the ImageNet input contract
// ---------------------------------------------------------------------
void inputs::preprocess(const cv::Mat& bgrImage, float* dst) {
    const int plane = cfg::imageSize * cfg::imageSize;

    // 1. Resize: shortest side -> cfg::resizeShortSide.
    const int h = bgrImage.rows, w = bgrImage.cols;
    const float scale = float(cfg::resizeShortSide) / std::min(h, w);
    cv::Mat resized;
    cv::resize(bgrImage, resized, cv::Size(int(w * scale), int(h * scale)));

    // 2. Center-crop.
    const int x0 = (resized.cols - cfg::imageSize) / 2;
    const int y0 = (resized.rows - cfg::imageSize) / 2;
    cv::Mat cropped = resized(cv::Rect(x0, y0, cfg::imageSize, cfg::imageSize)).clone();

    // 3 + 4 + 5: BGR -> RGB, uint8 -> float [0,1], normalize, scatter
    // into NCHW. f is HWC-interleaved (3 channels per pixel); the
    // network wants CHW, so split, normalize, and copy each plane into
    // its slab of dst.
    cv::Mat rgb, f;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> ch(cfg::channels);
    cv::split(f, ch);
    for (std::size_t c = 0; c < cfg::channels; ++c) {
        ch[c] = (ch[c] - cfg::mean[c]) / cfg::stddev[c];
        std::memcpy(dst + c * plane, ch[c].data, plane * sizeof(float));
    }
}

// ---------------------------------------------------------------------
// InputSet
// ---------------------------------------------------------------------

// Fixed mode: decode and preprocess the whole set ONCE, here, at
// construction — outside the timed loop. The images are sampled evenly
// across the dataset (not "the first N", which happen to be nearly all
// 500x375) so the set spans the real spread of image sizes.
inputs::InputSet::InputSet(const Options& opts,
                           const std::vector<std::string>& images,
                           std::size_t tensorSize)
    : fixed_(opts.fixedInputs), tensorSize_(tensorSize)
{
    if (tensorSize_ == 0)
        throw std::runtime_error("input tensor size is zero");
    if (images.empty())
        throw std::runtime_error("no images available for the input set");

    if (fixed_) {
        const std::size_t wanted = std::min(opts.numFixedInputs, images.size());
        if (wanted == 0)
            throw std::runtime_error("fixed input set size must be at least 1");

        // Pinned, so the H2D copies from it stay genuinely asynchronous.
        buffers_ = PinnedHostBuffer(wanted * tensorSize_);
        names_.reserve(wanted);

        std::size_t stored = 0;
        for (std::size_t k = 0; k < wanted; ++k) {
            const std::size_t idx =
                (wanted <= 1) ? 0 : (k * (images.size() - 1)) / (wanted - 1);
            cv::Mat img = cv::imread(images[idx], cv::IMREAD_COLOR);
            if (img.empty()) continue;              // skip unreadable
            preprocess(img, buffers_.data() + stored * tensorSize_);
            names_.push_back(fs::path(images[idx]).filename().string());
            ++stored;
        }
        if (stored == 0)
            throw std::runtime_error("no readable images for the fixed input set");
        nFixed_ = stored;
    } else {
        images_  = images;
        scratch_ = PinnedHostBuffer(tensorSize_);
    }
}

const float* inputs::InputSet::at(std::size_t i) {
    if (fixed_) {
        // Pure pointer arithmetic: no disk, no decode, no preprocessing.
        return buffers_.data() + (i % nFixed_) * tensorSize_;
    }

    // Dataset mode: pay the full CPU cost on every call. This is the
    // end-to-end pipeline path, kept for later phases.
    const std::string& path = images_[i % images_.size()];
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read image: " + path);
    preprocess(img, scratch_.data());
    return scratch_.data();
}

std::string inputs::InputSet::idOf(std::size_t i) const {
    if (fixed_) return names_[i % nFixed_];
    return fs::path(images_[i % images_.size()]).filename().string();
}

std::size_t inputs::InputSet::count() const noexcept {
    return fixed_ ? nFixed_ : images_.size();
}
