// auxprobe — the auxiliary-stream census.
//
// Phase A of the work queue experiment settled the queue knobs. This settles the
// other half of the professor's question: IBuilderConfig::setMaxAuxStreams() caps
// how many auxiliary streams TensorRT MAY use, but
//
//     ICudaEngine::getNbAuxStreams()
//
// reports how many the built engine actually USES, which can be fewer "if the
// network does not contain enough parallelism or if TensorRT determines that
// using more auxiliary streams does not help improve the performance". That
// number is what a defensible wqConcurrencyLimit has to be consistent with:
// main stream + n aux = the concurrency one client can produce.
//
// The census is entirely in the engine, so this deserializes and prints — no
// inference, no timing, nothing that perturbs a measurement in flight.
//
// Build (matches the runner's toolchain):
//   g++ -O2 -std=c++17 -I/usr/local/cuda/include -I/usr/include/aarch64-linux-gnu \
//       auxprobe.cpp -o auxprobe \
//       -L/usr/local/cuda/targets/sbsa-linux/lib -lnvinfer -lcudart

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <NvInferRuntime.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: auxprobe <engine.engine> [engine.engine ...]\n");
        return 2;
    }

    // TensorRT's own logger messages are noise here; the census is one number
    // per engine.
    class Quiet final : public nvinfer1::ILogger {
        void log(Severity s, const char* msg) noexcept override {
            if (s == Severity::kERROR || s == Severity::kINTERNAL_ERROR)
                std::fprintf(stderr, "  [TRT] %s\n", msg);
        }
    } logger;

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) { std::fprintf(stderr, "createInferRuntime failed\n"); return 1; }

    int failures = 0;
    for (int a = 1; a < argc; ++a) {
        const std::string path = argv[a];

        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); ++failures; continue; }
        std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());

        nvinfer1::ICudaEngine* engine =
            runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!engine) {
            std::fprintf(stderr, "deserialize failed: %s\n", path.c_str());
            ++failures;
            continue;
        }

        const int32_t aux = engine->getNbAuxStreams();

        // Trim the path to the bare file name so the line stays readable.
        std::string name = path;
        const std::size_t slash = name.find_last_of('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);

        std::printf("  %-28s aux streams = %d   -> concurrency per client = %d"
                    "  (%d main + %d aux)\n",
                    name.c_str(), aux, aux + 1, 1, aux);

        // TensorRT 10 removed the destroy() methods from these interfaces; the
        // objects are owned by the caller and released with delete.
        delete engine;
    }

    delete runtime;
    return failures ? 1 : 0;
}
