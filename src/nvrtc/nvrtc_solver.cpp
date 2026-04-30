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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>
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

namespace {

struct CompileOpts {
    std::string m_size_def;
    std::string lda_def;
    std::string sm_def;
    std::string bpb_def;
    std::string block_dim_def;
    std::string arch_opt;
    std::string overlay_inc;
    std::string cusolver_inc;
    std::string cutlass_inc;
    std::string cuda_inc;
    std::vector<const char*> ptrs;
};

CompileOpts make_opts(int n, int bpb, unsigned block_dim_x, int arch) {
    CompileOpts o;
    o.m_size_def    = "-DM_SIZE="            + std::to_string(n);
    o.lda_def       = "-DSOLVER_LDA="        + std::to_string(n);
    o.sm_def        = "-DSOLVER_SM=800";
    o.bpb_def       = "-DBATCHES_PER_BLOCK=" + std::to_string(bpb);
    o.block_dim_def = "-DBLOCK_DIM_X="       + std::to_string(block_dim_x);
    o.arch_opt      = "--gpu-architecture=sm_" + std::to_string(arch);
    o.overlay_inc   = std::string("--include-path=") + CUSOLVERDX_OVERLAY_INCLUDE_DIR;
    o.cusolver_inc  = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    o.cutlass_inc   = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    o.cuda_inc      = std::string("--include-path=") + CUDA_INCLUDE_DIR;
    o.ptrs = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-dlto",
        "--relocatable-device-code=true",
        o.m_size_def.c_str(),
        o.lda_def.c_str(),
        o.sm_def.c_str(),
        o.bpb_def.c_str(),
        o.block_dim_def.c_str(),
        o.arch_opt.c_str(),
        o.overlay_inc.c_str(),
        o.cusolver_inc.c_str(),
        o.cutlass_inc.c_str(),
        o.cuda_inc.c_str(),
    };
    return o;
}

// Compile NVRTC source → cubin (via -dlto + nvJitLink against cuSolverDx
// fatbin). Returns the linked cubin bytes. Throws on failure with NVRTC log.
std::vector<char> nvrtc_to_cubin(const char* source,
                                 const char* source_name,
                                 const std::vector<const char*>& opts,
                                 int arch,
                                 const char* fail_context) {
    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program, source, source_name,
                                   0, nullptr, nullptr));
    const nvrtcResult res =
        nvrtcCompileProgram(program, static_cast<int>(opts.size()), opts.data());
    if (res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(program, &log[0]);
        nvrtcDestroyProgram(&program);
        std::ostringstream oss;
        oss << "NVRTC compile failed (" << fail_context << "): "
            << nvrtcGetErrorString(res) << "\n" << log;
        throw std::runtime_error(oss.str());
    }
    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
    std::vector<char> lto(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(program, lto.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));

    const std::string nvjit_arch = "-arch=sm_" + std::to_string(arch);
    const char* link_opts[] = { "-lto", nvjit_arch.c_str() };
    nvJitLinkHandle linker = nullptr;
    NVJITLINK_CHECK(nullptr, nvJitLinkCreate(&linker, 2, link_opts));
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker,
        NVJITLINK_INPUT_FATBIN, CUSOLVERDX_FATBIN_PATH));
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker,
        NVJITLINK_INPUT_LTOIR, lto.data(), lto_size, "lto_module"));
    NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));
    std::size_t cubin_size = 0;
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    nvJitLinkDestroy(&linker);
    return cubin;
}

int device_arch(int device_id) {
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, device_id));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    return major * 10 + minor;
}

}  // namespace

unsigned int NvrtcGeneigSolver::probe_suggested_block_dim_x_(int bpb) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int arch = device_arch(device_id_);
    // Probe uses BlockDim<128> (irrelevant — any value works to read the
    // suggested constants). The probe source has no execute() calls, so
    // nvJitLink finishes in ~1 s rather than the 60–200 s of a full compile.
    const CompileOpts opts = make_opts(n_, bpb, /*block_dim_x=*/128, arch);
    const std::vector<char> cubin = nvrtc_to_cubin(
        nvrtc_geneig::kProbeSource, "probe.cu", opts.ptrs, arch,
        ("probe n=" + std::to_string(n_) + " bpb=" + std::to_string(bpb)).c_str());

    CUmodule mod = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&mod, cubin.data(), 0, nullptr, nullptr));
    auto read_dim_x = [&](const char* name) -> unsigned {
        CUdeviceptr ptr; std::size_t size; dim3 v{0,0,0};
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, mod, name));
        CU_CHECK(cuMemcpyDtoH(&v, ptr, size));
        return v.x;
    };
    const unsigned bd_chol  = read_dim_x("cholesky_suggested_block_dim");
    const unsigned bd_trsmL = read_dim_x("trsm_left_suggested_block_dim");
    const unsigned bd_trsmR = read_dim_x("trsm_right_suggested_block_dim");
    const unsigned bd_trsmC = read_dim_x("trsm_lc_suggested_block_dim");
    const unsigned bd_heev  = read_dim_x("heev_suggested_block_dim");
    cuModuleUnload(mod);

    const unsigned chosen = std::max({bd_chol, bd_trsmL, bd_trsmR, bd_trsmC, bd_heev});
    auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[NvrtcGeneigSolver] probe (n=%d bpb=%d): %.0f ms; "
        "suggested block_dim.x — chol=%u trsmL=%u trsmR=%u trsmLC=%u heev=%u → max=%u\n",
        n_, bpb, ms, bd_chol, bd_trsmL, bd_trsmR, bd_trsmC, bd_heev, chosen);
    return chosen;
}

void NvrtcGeneigSolver::compile_production_(int bpb, unsigned int block_dim_x) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int arch = device_arch(device_id_);
    const CompileOpts opts = make_opts(n_, bpb, block_dim_x, arch);
    const std::vector<char> cubin = nvrtc_to_cubin(
        nvrtc_geneig::kKernelSource, "geneig_full_kernel.cu",
        opts.ptrs, arch,
        ("production n=" + std::to_string(n_) + " bpb=" + std::to_string(bpb)
         + " bdx=" + std::to_string(block_dim_x)).c_str());

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
    auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[NvrtcGeneigSolver] production (n=%d bpb=%d bdx=%u): %.0f ms; "
        "shared_mem=%u, cubin=%zu\n",
        n_, bpb, block_dim_x, ms, shared_mem_bytes_, cubin.size());
}

NvrtcGeneigSolver::NvrtcGeneigSolver(int n,
                                     int device_id,
                                     int batches_per_block_request,
                                     int force_block_dim_x)
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

    auto try_one = [&](int bpb) -> bool {
        unsigned int chosen_bd_x = 0;
        if (force_block_dim_x > 0) {
            chosen_bd_x = static_cast<unsigned>(force_block_dim_x);
            std::fprintf(stderr,
                "[NvrtcGeneigSolver] BPB=%d forcing block_dim.x=%u (probe skipped)\n",
                bpb, chosen_bd_x);
        } else {
            try {
                chosen_bd_x = probe_suggested_block_dim_x_(bpb);
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "[NvrtcGeneigSolver] BPB=%d probe rejected: %s\n",
                    bpb, e.what());
                return false;
            }
        }
        try {
            compile_production_(bpb, chosen_bd_x);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[NvrtcGeneigSolver] BPB=%d production rejected: %s\n",
                bpb, e.what());
            return false;
        }
        if (static_cast<int>(shared_mem_bytes_) > budget) {
            std::fprintf(stderr,
                "[NvrtcGeneigSolver] BPB=%d shared_mem=%u > budget=%d\n",
                bpb, shared_mem_bytes_, budget);
            return false;
        }
        batches_per_block_ = bpb;
        return true;
    };

    if (batches_per_block_request > 0) {
        if (!try_one(batches_per_block_request)) {
            std::ostringstream oss;
            oss << "explicit BPB=" << batches_per_block_request
                << " could not be constructed (NVRTC reject or shared_mem "
                << shared_mem_bytes_ << " > budget " << budget << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        const int candidates[] = {16, 8, 4, 2, 1};
        bool ok = false;
        for (int cand : candidates) {
            if (try_one(cand)) { ok = true; break; }
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
