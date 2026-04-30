// Phase 3a — implementation of NvrtcGeneigSolver (reduce half).
//
// The constructor walks the standard cuSolverDx-NVRTC pipeline:
//   nvrtcCreateProgram → nvrtcCompileProgram (with -dlto) → nvrtcGetLTOIR
//   → nvJitLinkCreate → nvJitLinkAddFile(libcusolverdx.fatbin)
//   → nvJitLinkAddData(LTOIR) → nvJitLinkComplete → nvJitLinkGetLinkedCubin
//   → cuModuleLoadDataEx → cuModuleGetFunction.
//
// The primary device context is acquired via cuDevicePrimaryCtxRetain so this
// path coexists with runtime-API allocations (cudaMalloc et al.) on the same
// device.

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

}  // namespace

NvrtcGeneigSolver::NvrtcGeneigSolver(int device_id) : device_id_(device_id) {
    using clk = std::chrono::steady_clock;
    auto t_ctor = clk::now();
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id_));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context_, cu_device));
    CU_CHECK(cuCtxSetCurrent(context_));

    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
    CU_CHECK(cuDeviceGetAttribute(&minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));
    const int arch = major * 10 + minor;
    auto t_after_ctx = clk::now();

    // ---- NVRTC: kernel source → LTO IR ------------------------------------
    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program,
                                   nvrtc_geneig::kKernelSource,
                                   "geneig_full_kernel.cu",
                                   /*numHeaders=*/0,
                                   /*headers=*/nullptr,
                                   /*includeNames=*/nullptr));

    const std::string m_size_def    = "-DM_SIZE=54";
    const std::string lda_def       = "-DSOLVER_LDA=54";
    // SOLVER_SM is the tuning-database lookup tag for cuSolverDx templates,
    // not a runtime architecture target. We pin it at SM<800> (Ampere) because
    // (a) it has full template-database coverage in cuSolverDx 25.12 for
    // potrf, trsm, and heev across all parameter combinations we use; (b) the
    // Phase 2 static build used SM<800> and ran correctly on sm_120, confirming
    // that runtime architecture dispatch is independent of the template-time
    // tuning tag. The `--gpu-architecture=sm_NN` flag below selects the actual
    // runtime arch via nvJitLink + the cuSolverDx fatbin.
    const std::string sm_def        = "-DSOLVER_SM=700";
    const std::string arch_opt      = "--gpu-architecture=sm_" + std::to_string(arch);
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
        arch_opt.c_str(),
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
        cuDevicePrimaryCtxRelease(cu_device);
        std::ostringstream oss;
        oss << "NVRTC compile failed: " << nvrtcGetErrorString(compile_res) << "\n" << log;
        throw std::runtime_error(oss.str());
    }

    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
    std::vector<char> lto_ir(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(program, lto_ir.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    auto t_after_ltoir = clk::now();

    // ---- nvJitLink: LTO IR + cuSolverDx fatbin → cubin --------------------
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

    // ---- Module load + symbol resolution ---------------------------------
    CU_CHECK(cuModuleLoadDataEx(&module_, cubin.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&kernel_, module_, "geneig_reduce_kernel"));
    auto t_after_module_load = clk::now();

    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, module_, "solver_block_dim"));
        if (size != sizeof(dim3)) {
            throw std::runtime_error("solver_block_dim has unexpected size");
        }
        CU_CHECK(cuMemcpyDtoH(&block_dim_, ptr, size));
    }
    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, module_, "solver_shared_memory_size"));
        if (size != sizeof(unsigned int)) {
            throw std::runtime_error("solver_shared_memory_size has unexpected size");
        }
        CU_CHECK(cuMemcpyDtoH(&shared_mem_bytes_, ptr, size));
    }

    CU_CHECK(cuFuncSetAttribute(kernel_,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_mem_bytes_)));
    auto t_done = clk::now();

    std::fprintf(stderr,
        "[NvrtcGeneigSolver] phase timings (ms):\n"
        "  ctx-init           : %8.1f\n"
        "  nvrtcCompile       : %8.1f  (lto_size=%zu)\n"
        "  GetLTOIR           : %8.1f\n"
        "  nvJitLink (link)   : %8.1f\n"
        "  cubin load+symbols : %8.1f  (cubin_size=%zu)\n"
        "  introspection      : %8.1f\n"
        "  TOTAL              : %8.1f\n",
        ms_since(t_ctor,             t_after_ctx),
        ms_since(t_before_compile,   t_after_compile),     lto_size,
        ms_since(t_after_compile,    t_after_ltoir),
        ms_since(t_after_ltoir,      t_after_link),
        ms_since(t_after_link,       t_after_module_load), cubin_size,
        ms_since(t_after_module_load,t_done),
        ms_since(t_ctor,             t_done));
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

void NvrtcGeneigSolver::launch_reduce(CUdeviceptr d_H, CUdeviceptr d_S,
                                       CUdeviceptr d_L, CUdeviceptr d_M,
                                       CUdeviceptr d_info,
                                       int batch_size, CUstream stream) {
    if (batch_size <= 0) return;
    int local_bs = batch_size;
    void* args[] = { &d_H, &d_S, &d_L, &d_M, &d_info, &local_bs };
    CU_CHECK(cuLaunchKernel(kernel_,
                            /*grid:*/ static_cast<unsigned>(batch_size), 1, 1,
                            /*block:*/ block_dim_.x, block_dim_.y, block_dim_.z,
                            shared_mem_bytes_, stream, args, nullptr));
}
