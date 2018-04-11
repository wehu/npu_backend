//
// Created by wehu on 18-4-8.
//

#ifndef TENSORFLOW_NPU_EXECUTABLE_H
#define TENSORFLOW_NPU_EXECUTABLE_H

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"

#include "npu_thunk_schedule.h"
#include "npu_buffer_allocations.h"

namespace npu {

    using namespace xla;

    class NpuExecutable : public Executable {
    public:
        NpuExecutable(std::unique_ptr<xla::cpu::SimpleOrcJIT> jit,
                      std::unique_ptr<const NpuThunkSchedule> thunk_schedule,
                      std::unique_ptr<const HloModule> hlo_module,
                      std::unique_ptr<const BufferAssignment> assignment,
                      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);

        // ExecuteOnStream will fail if the compute capability of the stream doesn't
        // match the compute capability passed to this object's constructor.
        StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteOnStream(
                const ServiceExecutableRunOptions* run_options,
                tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
                HloExecutionProfile* hlo_execution_profile) override;

        StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteAsyncOnStream(
                const ServiceExecutableRunOptions* run_options,
                tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) override;

        const Status EqualOrFail(const Executable& executable) {
            // TODO(b/62952745) Implement equality test on NPU executable.
            return Unimplemented("Equality test on NPU executable is not implemented.");
        }

    private:

        Status ExecuteThunks(const ServiceExecutableRunOptions* run_options,
                             const NpuBufferAllocations& buffer_allocations,
                             bool block_host_until_done,
                             HloExecutionProfile* hlo_execution_profile);

        const PointsToSet& GetRootPointsToSet() const;

        const std::unique_ptr<xla::cpu::SimpleOrcJIT> jit_;

        const std::unique_ptr<const NpuThunkSchedule> thunk_schedule_;

        const std::unique_ptr<const BufferAssignment> assignment_;

        TF_DISALLOW_COPY_AND_ASSIGN(NpuExecutable);
    };

}

#endif //TENSORFLOW_NPU_EXECUTABLE_H
