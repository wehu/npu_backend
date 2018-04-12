#ifndef TENSORFLOW_NPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_NPU_TRANSFER_MANAGER_H_

#include "npu_infeed_manager.h"

#include <vector>

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace npu {

    using namespace xla;

    class NpuTransferManager : public GenericTransferManager {
    public:
        NpuTransferManager();
        ~NpuTransferManager() override {}

        Status TransferLiteralToInfeed(perftools::gputools::StreamExecutor* executor,
                                       const Literal& literal) override;
        Status TransferBufferToInfeed(perftools::gputools::StreamExecutor* executor,
                                      int64 size, const void* source) override;
    private:

        StatusOr<NpuInfeedBuffer*> TransferBufferToInfeedInternal(
                perftools::gputools::StreamExecutor* executor, int64 size,
                const void* source);

        Status EnqueueBuffersToInfeed(perftools::gputools::StreamExecutor* executor,
                                      std::vector<NpuInfeedBuffer*> buffers);

        TF_DISALLOW_COPY_AND_ASSIGN(NpuTransferManager);
    };

}  // namespace npu

#endif  // TENSORFLOW_NPU_TRANSFER_MANAGER_H_
