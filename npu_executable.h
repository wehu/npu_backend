//
// Created by wehu on 18-4-8.
//

#ifndef TENSORFLOW_NPU_EXECUTABLE_H
#define TENSORFLOW_NPU_EXECUTABLE_H

#include "tensorflow/compiler/xla/service/executable.h"

namespace npu {

    using namespace xla;

    class NpuExecutable : public Executable {
    public:
        NpuExecutable(std::unique_ptr<const HloModule> hlo_module,
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

        TF_DISALLOW_COPY_AND_ASSIGN(NpuExecutable);
    };

}

#endif //TENSORFLOW_NPU_EXECUTABLE_H
