//
// Created by wehu on 18-4-8.
//

#include <llvm/Support/raw_ostream.h>
#include "llvm/IR/DiagnosticInfo.h"
#include <llvm/IR/DiagnosticPrinter.h>
#include "npu_compiler.h"
#include "npu_executable.h"
#include "npu_ir_emitter.h"
#include "npu_ir_emitter_context.h"
#include "npu_platform_id.h"
#include "tensorflow/core/platform/tracing.h"
#include "llvm/IR/DataLayout.h"

#include "npu_stream_assignment.h"
#include "npu_thunk_schedule.h"
#include "npu_hlo_schedule.h"

namespace se = ::perftools::gputools;

namespace npu {

    /* static */ const char *NpuCompiler::kTargetTriple =
            "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
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

        llvm::LLVMContext llvm_context;
        std::string buffer;
        llvm::raw_string_ostream error(buffer);
        llvm::DiagnosticPrinterRawOStream printer(error);
        auto DiagnosticHandler = [](const llvm::DiagnosticInfo& diag_info,
                                    void* Context) {
            auto printer = static_cast<llvm::DiagnosticPrinterRawOStream*>(Context);
            diag_info.print(*printer);
        };
        llvm_context.setDiagnosticHandlerCallBack(DiagnosticHandler, &printer);

        llvm::Module llvm_module(module->name().c_str(), llvm_context);
        // Set the target triple and the data layout.
        llvm_module.setTargetTriple(kTargetTriple);
        llvm_module.setDataLayout(kDataLayout);

        // Determine the HLO schedule, which is an ordering of HLO instructions.  This
        // is used by buffer assignment to enable buffer reuse, and the same ordering
        // must also be used to determine the thunk launch schedule.
        std::unique_ptr<NpuStreamAssignment> stream_assignment = AssignStreams(*module);
        TF_ASSIGN_OR_RETURN(
                std::unique_ptr<NpuHloSchedule> hlo_schedule,
                NpuHloSchedule::Build(*module, *stream_assignment, pointer_size_));

        // Run buffer analysis on the HLO graph. This analysis figures out which
        // temporary buffers are required to run the computation.
        TF_ASSIGN_OR_RETURN(
                std::unique_ptr<BufferAssignment> buffer_assignment,
                BufferAssigner::Run(module.get(), hlo_schedule->ConsumeHloOrdering(),
                                    BufferSizeBytesFunction(),
                        /*color_alignment=*/[](LogicalBuffer::Color) {
                            return npuAlignBytes;
                        }));
        // BufferAssignment::Stats::ToString() and BufferAssignment::ToString()
        // include headers, so no need for us to print them ourselves.
        XLA_VLOG_LINES(1, buffer_assignment->GetStats().ToString());
        XLA_VLOG_LINES(2, buffer_assignment->ToString());
        XLA_VLOG_LINES(2, module->ToString());
        const string xla_dump_optimized_hlo_proto_to =
                module->config().debug_options().xla_dump_optimized_hlo_proto_to();

        IrEmitterContext ir_emitter_context(module.get(), buffer_assignment.get(),
                                            &stream_exec->GetDeviceDescription(),
                                            &llvm_module);

        HloComputation* entry_computation = module->entry_computation();
        IrEmitter ir_emitter(module->config(), entry_computation,
                                     &ir_emitter_context);
        {
            XLA_SCOPED_LOGGING_TIMER("NpuCompiler::RunBackend - IR emission");
            TF_RETURN_IF_ERROR(
                    entry_computation->root_instruction()->Accept(&ir_emitter));
        }

        auto thunk_schedule = MakeUnique<NpuThunkSchedule>(
                ir_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
                hlo_schedule->ThunkLaunchOrder());
        VLOG(2) << "Printing the thunk schedule...";
        XLA_VLOG_LINES(2, thunk_schedule->ToString());

        std::unique_ptr<HloProfileIndexMap> profile_index_map;
        std::unique_ptr<HloProfilePrinterData> profile_printer;

        auto* npu_executable = new NpuExecutable(
                std::move(thunk_schedule),
                std::move(module), std::move(buffer_assignment),
                std::move(profile_printer), std::move(profile_index_map));
        return std::unique_ptr<Executable>(npu_executable);
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