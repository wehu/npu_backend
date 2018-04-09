//
// Created by wehu on 18-4-8.
//

#include "npu_compiler.h"
#include "npu_executable.h"
#include "npu_platform_id.h"
#include "tensorflow/core/platform/tracing.h"
#include "llvm/IR/DataLayout.h"

namespace se = ::perftools::gputools;

namespace npu {

    /* static */ const char *NpuCompiler::kDataLayout =
            "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

    using namespace xla;

    NpuCompiler::NpuCompiler()
            : pointer_size_(llvm::DataLayout(kDataLayout)
                                    .getPointerSize(0 /* default address space */)) {}

    StatusOr<std::unique_ptr<HloModule>> NpuCompiler::RunHloPasses(
            std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
            DeviceMemoryAllocator* device_allocator) {
        XLA_SCOPED_LOGGING_TIMER("NpuCompiler::RunHloPasses");
        return std::move(module);
    }

    StatusOr<std::unique_ptr<Executable>> NpuCompiler::RunBackend(
            std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
            DeviceMemoryAllocator* device_allocator) {
        XLA_SCOPED_LOGGING_TIMER("NpuCompiler::RunBackend");

        TF_RET_CHECK(stream_exec != nullptr);

        std::unique_ptr<HloProfileIndexMap> profile_index_map;
        std::unique_ptr<HloProfilePrinterData> profile_printer;

        auto * executable = new NpuExecutable(std::move(module),
                                              std::move(profile_printer),
                                              std::move(profile_index_map));

        return std::unique_ptr<Executable>(executable);
    }

    StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
    NpuCompiler::CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                                    const AotCompilationOptions& options) {
        return Unimplemented("not yet implemented: NpuCompiler::CompileAheadOfTime");
    }

    se::Platform::Id NpuCompiler::PlatformId() const {
        return npuPlatformId;
    }

}

static bool InitModule() {
    xla::Compiler::RegisterCompilerFactory(npu::npuPlatformId, []() {
        return xla::MakeUnique<npu::NpuCompiler>();
    });
    return true;
}
static bool module_initialized = InitModule();