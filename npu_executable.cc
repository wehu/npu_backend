//
// Created by wehu on 18-4-8.
//

#include "npu_executable.h"

namespace npu {

    using namespace xla;

    NpuExecutable::NpuExecutable(std::unique_ptr<const HloModule> hlo_module,
                                 std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                                 std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)) {

    };

    StatusOr<std::unique_ptr<ShapedBuffer>> NpuExecutable::ExecuteOnStream(
            const ServiceExecutableRunOptions* run_options,
            tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
            HloExecutionProfile* hlo_execution_profile) {
        return Unimplemented(
                "Synchronous execution on stream is not yet supported on NPU.");
    }

    StatusOr<std::unique_ptr<ShapedBuffer>> NpuExecutable::ExecuteAsyncOnStream(
            const ServiceExecutableRunOptions* run_options,
            tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
        return Unimplemented(
                "Asynchronous execution on stream is not yet supported on NPU.");

    }


}

