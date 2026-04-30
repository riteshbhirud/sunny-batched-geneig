// Phase 3.5a — implementation of NvrtcGeneigSolver with BatchesPerBlock support.
//
// The constructor walks the cuSolverDx-NVRTC pipeline:
//   nvrtcCreateProgram → nvrtcCompileProgram (with -dlto) → nvrtcGetLTOIR
//   → nvJitLinkCreate → nvJitLinkAddFile(libcusolverdx.fatbin)
//   → nvJitLinkAddData(LTOIR) → nvJitLinkComplete → nvJitLinkGetLinkedCubin
//   → cuModuleLoadDataEx → cuModuleGetFunction.
//
// BatchesPerBlock auto-selection: try BPB candidates in {16, 8, 4, 2, 1}.
// For each candidate that hasn't been ruled out by a quick sanity check, we
// actually compile and read back `solver_shared_memory_size`. The first one
// whose reported size fits within (device_max_smem - 2 KB) is accepted.

#include "nvrtc_solver.hpp"
#include "kernel_source.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void throw_nvrtc(nvrtcResult r, const char* call) {
    std::ostringstream oss;
    oss << "NVRTC error at " << call << ": " << nvrtcGetErrorString(r);
    throw std::runtime_error(oss.str());
}

[[noreturn]] void throw_cu(CUresult r, const char* call) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    std::ostringstream oss;
    oss << "CUDA driver error at " << call << ": " << (msg ? msg : "<no msg>");
    throw std::runtime_error(oss.str());
}

void throw_nvjitlink(nvJitLinkHandle linker, nvJitLinkResult r, const char* call) {
    std::ostringstream oss;
    oss << "nvJitLink error at " << call << ": " << static_cast<int>(r);
    if (linker) {
        std::size_t lsize = 0;
        if (nvJitLinkGetErrorLogSize(linker, &lsize) == NVJITLINK_SUCCESS && lsize > 0) {
            std::vector<char> log(lsize);
            if (nvJitLinkGetErrorLog(linker, log.data()) == NVJITLINK_SUCCESS) {
                oss << "\n" << log.data();
            }
        }
    }
    throw std::runtime_error(oss.str());
}

#define NVRTC_CHECK(expr) do {                                           \
    nvrtcResult _r = (expr);                                             \
    if (_r != NVRTC_SUCCESS) throw_nvrtc(_r, #expr);                     \
} while (0)

#define CU_CHECK(expr) do {                                              \
    CUresult _r = (expr);                                                \
    if (_r != CUDA_SUCCESS) throw_cu(_r, #expr);                         \
} while (0)

#define NVJITLINK_CHECK(linker, expr) do {                               \
    nvJitLinkResult _r = (expr);                                         \
    if (_r != NVJITLINK_SUCCESS) throw_nvjitlink((linker), _r, #expr);   \
} while (0)

// No artificial safety margin — cuFuncSetAttribute is the real gate.
// Earlier we tried 2 KB margin which incorrectly rejected the N=54 BPB=1
// case (shared_mem=99712 vs device_max=101376) on a config that we'd
// already proven works in Phase 3b.
constexpr int kSafetyMarginBytes = 0;

}  // namespace

void NvrtcGeneigSolver::compile_for_(int bpb) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id_));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
    CU_CHECK(cuDeviceGetAttribute(&minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));
    const int arch = major * 10 + minor;

    // ---- NVRTC: kernel source → LTO IR ----
    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program, nvrtc_geneig::kKernelSource,
        "geneig_full_kernel.cu", 0, nullptr, nullptr));

    const std::string m_size_def    = "-DM_SIZE="            + std::to_string(n_);
    const std::string lda_def       = "-DSOLVER_LDA="        + std::to_string(n_);
    const std::string sm_def        = "-DSOLVER_SM=800";
    const std::string bpb_def       = "-DBATCHES_PER_BLOCK=" + std::to_string(bpb);
    const std::string arch_opt      = "--gpu-architecture=sm_" + std::to_string(arch);
    const std::string overlay_inc   = std::string("--include-path=") + CUSOLVERDX_OVERLAY_INCLUDE_DIR;
    const std::string cusolver_inc  = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    const std::string cutlass_inc   = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    const std::string cuda_inc      = std::string("--include-path=") + CUDA_INCLUDE_DIR;

    const std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-dlto",
        "--relocatable-device-code=true",
        m_size_def.c_str(),
        lda_def.c_str(),
        sm_def.c_str(),
        bpb_def.c_str(),
        arch_opt.c_str(),
        overlay_inc.c_str(),
        cusolver_inc.c_str(),
        cutlass_inc.c_str(),
        cuda_inc.c_str(),
    };

    auto t_before_compile = clk::now();
    const nvrtcResult compile_res =
        nvrtcCompileProgram(program, static_cast<int>(opts.size()), opts.data());
    auto t_after_compile = clk::now();
    if (compile_res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(program, &log[0]);
        nvrtcDestroyProgram(&program);
        std::ostringstream oss;
        oss << "NVRTC compile failed (n=" << n_ << ", bpb=" << bpb << "): "
            << nvrtcGetErrorString(compile_res) << "\n" << log;
        throw std::runtime_error(oss.str());
    }

    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
    std::vector<char> lto_ir(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(program, lto_ir.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    auto t_after_ltoir = clk::now();

    // ---- nvJitLink: LTO IR + cuSolverDx fatbin → cubin ----
    const std::string nvjit_arch = "-arch=sm_" + std::to_string(arch);
    const char* link_opts[] = { "-lto", nvjit_arch.c_str() };
    nvJitLinkHandle linker = nullptr;
    NVJITLINK_CHECK(nullptr, nvJitLinkCreate(&linker, 2, link_opts));
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker,
        NVJITLINK_INPUT_FATBIN, CUSOLVERDX_FATBIN_PATH));
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker,
        NVJITLINK_INPUT_LTOIR, lto_ir.data(), lto_size, "geneig_lto"));
    NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));
    auto t_after_link = clk::now();

    std::size_t cubin_size = 0;
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    nvJitLinkDestroy(&linker);

    // ---- Module load + symbol resolution ----
    if (module_) { cuModuleUnload(module_); module_ = nullptr; }
    CU_CHECK(cuModuleLoadDataEx(&module_, cubin.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&kernel_, module_, "geneig_full_kernel"));

    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, module_, "solver_block_dim"));
        if (size != sizeof(dim3)) throw std::runtime_error("solver_block_dim has unexpected size");
        CU_CHECK(cuMemcpyDtoH(&block_dim_, ptr, size));
    }
    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, module_, "solver_shared_memory_size"));
        if (size != sizeof(unsigned int)) throw std::runtime_error("solver_shared_memory_size has unexpected size");
        CU_CHECK(cuMemcpyDtoH(&shared_mem_bytes_, ptr, size));
    }
    auto t_done = clk::now();

    std::fprintf(stderr,
        "[NvrtcGeneigSolver] compile (n=%d bpb=%d): %.0f ms total "
        "(nvrtc %.0f, nvJitLink %.0f); shared_mem=%u, lto=%zu, cubin=%zu\n",
        n_, bpb,
        ms_since(t_start, t_done),
        ms_since(t_before_compile, t_after_compile),
        ms_since(t_after_ltoir, t_after_link),
        shared_mem_bytes_, lto_size, cubin_size);
}

NvrtcGeneigSolver::NvrtcGeneigSolver(int n,
                                     int device_id,
                                     int batches_per_block_request)
    : n_(n), device_id_(device_id) {
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id_));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context_, cu_device));
    CU_CHECK(cuCtxSetCurrent(context_));

    CU_CHECK(cuDeviceGetAttribute(&device_max_smem_,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cu_device));

    const int budget = device_max_smem_ - kSafetyMarginBytes;
    if (budget <= 0) {
        std::ostringstream oss;
        oss << "device max dynamic shared memory (" << device_max_smem_
            << " B) is below safety margin (" << kSafetyMarginBytes << " B)";
        throw std::runtime_error(oss.str());
    }

    if (batches_per_block_request > 0) {
        // Explicit override: compile and verify it fits.
        compile_for_(batches_per_block_request);
        if (static_cast<int>(shared_mem_bytes_) > budget) {
            std::ostringstream oss;
            oss << "explicit BPB=" << batches_per_block_request
                << " requires " << shared_mem_bytes_
                << " B shared memory, exceeds device budget "
                << budget << " B (max " << device_max_smem_
                << ", margin " << kSafetyMarginBytes << ")";
            throw std::runtime_error(oss.str());
        }
        batches_per_block_ = batches_per_block_request;
    } else {
        // Auto-select: try BPB candidates from largest to smallest.
        const int candidates[] = {16, 8, 4, 2, 1};
        bool ok = false;
        for (int cand : candidates) {
            try {
                compile_for_(cand);
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "[NvrtcGeneigSolver] BPB=%d compile rejected: %s\n",
                    cand, e.what());
                continue;
            }
            if (static_cast<int>(shared_mem_bytes_) <= budget) {
                batches_per_block_ = cand;
                ok = true;
                break;
            }
            std::fprintf(stderr,
                "[NvrtcGeneigSolver] BPB=%d shared_mem=%u > budget=%d, trying smaller\n",
                cand, shared_mem_bytes_, budget);
        }
        if (!ok) {
            std::ostringstream oss;
            oss << "no BPB candidate in {16,8,4,2,1} fits device budget "
                << budget << " B (max " << device_max_smem_ << " B). "
                << "Last attempted BPB shared_mem=" << shared_mem_bytes_;
            throw std::runtime_error(oss.str());
        }
    }

    CU_CHECK(cuFuncSetAttribute(kernel_,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_mem_bytes_)));
}

NvrtcGeneigSolver::~NvrtcGeneigSolver() {
    if (module_) {
        cuModuleUnload(module_);
        module_ = nullptr;
    }
    CUdevice dev;
    if (cuDeviceGet(&dev, device_id_) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(dev);
    }
    context_ = nullptr;
}

void NvrtcGeneigSolver::launch(CUdeviceptr d_H, CUdeviceptr d_S,
                                CUdeviceptr d_W, CUdeviceptr d_U,
                                CUdeviceptr d_info,
                                int batch_size, CUstream stream) {
    if (batch_size <= 0) return;
    int local_bs = batch_size;
    void* args[] = { &d_H, &d_S, &d_W, &d_U, &d_info, &local_bs };
    const unsigned grid =
        static_cast<unsigned>((batch_size + batches_per_block_ - 1) / batches_per_block_);
    CU_CHECK(cuLaunchKernel(kernel_,
                            /*grid:*/ grid, 1, 1,
                            /*block:*/ block_dim_.x, block_dim_.y, block_dim_.z,
                            shared_mem_bytes_, stream, args, nullptr));
}
