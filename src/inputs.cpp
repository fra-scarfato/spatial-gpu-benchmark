//
// inputs.cpp — dataset listing, ImageNet preprocessing, and the fixed-input set.
//

#include "inputs.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::vector<std::string> inputs::listImages(const std::string& dir) {
    // Sorted on purpose. The directory order is arbitrary, and two runs that
    // pick up the same images in a different order are not comparable — so the
    // sort is what makes "the same 32 images" mean the same thing twice.
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".JPEG")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// The ImageNet input contract. Getting any single step of this wrong does not
// crash: it quietly degrades accuracy, and the latency is unaffected, so the
// benchmark would look perfect while the model became a random number
// generator. That is why it is spelled out step by step.
void inputs::preprocess(const cv::Mat& bgrImage, float* dst) {
    const int plane = cfg::imageSize * cfg::imageSize;

    // 1. Resize so the SHORTEST side becomes 256. The aspect ratio is kept,
    //    so one of the two sides ends up longer than the crop below.
    const int h = bgrImage.rows, w = bgrImage.cols;
    const float scale = float(cfg::resizeShortSide) / std::min(h, w);
    cv::Mat resized;
    cv::resize(bgrImage, resized, cv::Size(int(w * scale), int(h * scale)));

    // 2. Centre-crop 224x224 out of it. The crop can only succeed because step
    //    1 guaranteed both sides are at least 224.
    const int x0 = (resized.cols - cfg::imageSize) / 2;
    const int y0 = (resized.rows - cfg::imageSize) / 2;
    cv::Mat cropped = resized(cv::Rect(x0, y0, cfg::imageSize, cfg::imageSize)).clone();

    // 3. OpenCV loads in BGR; the networks were trained on RGB. Swapping the
    //    channels is the single easiest step to forget, and it costs accuracy
    //    without costing any time.
    cv::Mat rgb, f;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    // 4. uint8 -> float, scaled to [0,1].
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

    // 5. Normalise per channel and scatter into NCHW.
    //
    //    f is HWC (three interleaved channels per pixel) and the network wants
    //    CHW (three whole planes). So split into per-channel planes, normalise
    //    each with its own mean and standard deviation, and copy each plane
    //    into its slab of dst — which is what the c * plane offset is doing.
    std::vector<cv::Mat> ch(cfg::channels);
    cv::split(f, ch);
    for (std::size_t c = 0; c < cfg::channels; ++c) {
        ch[c] = (ch[c] - cfg::mean[c]) / cfg::stddev[c];
        std::memcpy(dst + c * plane, ch[c].data, plane * sizeof(float));
    }
}

// Fixed mode: decode and preprocess the whole set ONCE, here, at construction,
// which puts all of that work outside the timed loop.
//
// The images are sampled EVENLY across the dataset rather than taken from the
// front. The first N images of ILSVRC2012 are nearly all 500x375, so "the first
// N" would give a set that barely varies in shape while looking like a diverse
// sample.
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

        // Pinned, because the host-to-device copies read from here and pinned
        // memory is what keeps those copies genuinely asynchronous.
        buffers_ = PinnedHostBuffer(wanted * tensorSize_);
        names_.reserve(wanted);

        // `stored` rather than `k` as the write index: an unreadable image is
        // skipped, so the destination has to advance only on success or the
        // set would be left with a hole.
        std::size_t stored = 0;
        for (std::size_t k = 0; k < wanted; ++k) {
            // Even spacing across the whole list: k*(N-1)/(wanted-1) lands on
            // the first and last image for k=0 and k=wanted-1, and spreads the
            // rest in between. The guard covers wanted == 1, where the divisor
            // would be zero.
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
        // Dataset mode: one image per iteration, decoded on demand, so a single
        // pinned scratch buffer is all that is needed.
        images_  = images;
        scratch_ = PinnedHostBuffer(tensorSize_);
    }
}

const float* inputs::InputSet::at(std::size_t i) {
    if (fixed_) {
        // Pointer arithmetic only — no disk, no decode, no preprocessing — and
        // the modulo is what cycles the set, so the caller can keep counting
        // inferences past the number of distinct images.
        return buffers_.data() + (i % nFixed_) * tensorSize_;
    }

    // Dataset mode: pay the full CPU cost on every call. This is the
    // end-to-end pipeline path.
    const std::string& path = images_[i % images_.size()];
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read image: " + path);
    preprocess(img, scratch_.data());
    return scratch_.data();
}

std::string inputs::InputSet::idOf(std::size_t i) const {
    // Recorded so the CSV can say which image an inference used, even though in
    // fixed mode the input itself is only an address.
    if (fixed_) return names_[i % nFixed_];
    return fs::path(images_[i % images_.size()]).filename().string();
}

std::size_t inputs::InputSet::count() const noexcept {
    // The number of DISTINCT inputs, not the number of samples: this is the
    // length of one pass, which is what the "single pass" stop rule uses.
    return fixed_ ? nFixed_ : images_.size();
}
