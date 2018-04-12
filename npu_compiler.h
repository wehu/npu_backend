//
// Created by wehu on 18-4-8.
//

#ifndef TENSORFLOW_NPU_COMPILER_H
#define TENSORFLOW_NPU_COMPILER_H

#include "tensorflow/compiler/xla/service/llvm_compiler.h"

namespace xla {
    namespace npu {

        // The NPU compiler generates efficient NPU executables.
        class NpuCompiler : public LLVMCompiler {
        public:
            NpuCompiler();

            ~NpuCompiler() override {};

            // Bring in
            // StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
            //     std::vector<std::unique_ptr<HloModule>> modules,
            //     std::vector<std::vector<perftools::gputools::StreamExecutor*>>
            //        stream_execs)

            using LLVMCompiler::Compile;

            StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
                    std::unique_ptr<HloModule> module,
                    perftools::gputools::StreamExecutor *stream_exec,
                    DeviceMemoryAllocator *device_allocator) override;

            StatusOr<std::unique_ptr<Executable>> RunBackend(
                    std::unique_ptr<HloModule> module,
                    perftools::gputools::StreamExecutor *stream_exec,
                    DeviceMemoryAllocator *device_allocator) override;

            StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
            CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                               AotCompilationOptions const &options) override;

            perftools::gputools::Platform::Id PlatformId() const override;

            HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
                // Capture just the pointer size, not the entire GpuCompiler object.
                int64 pointer_size = pointer_size_;
                return [pointer_size](const Shape &shape) {
                    return ShapeUtil::ByteSizeOf(shape, pointer_size);
                };
            }

            static const char *kTargetTriple;
            static const char *kDataLayout;

        private:
            // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
            const int64 pointer_size_;

            TF_DISALLOW_COPY_AND_ASSIGN(NpuCompiler);
        };

    } // namespace npu
} // namespace xla

#endif //TENSORFLOW_NPU_COMPILER_H
