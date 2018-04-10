#include "npu_transfer_manager.h"
#include "npu_platform_id.h"
#include "npu_compiler.h"

#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/DataLayout.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace npu {

    NpuTransferManager::NpuTransferManager()
            : GenericTransferManager(
            npuPlatformId,
            /*pointer_size=*/llvm::DataLayout(npu::NpuCompiler::kDataLayout)
                    .getPointerSize(0 /* default address space */)) {}

    Status NpuTransferManager::TransferLiteralToInfeed(se::StreamExecutor* executor,
                                                       const Literal& literal) {

      return Status::OK();
    }

    Status NpuTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                                      int64 size,
                                                      const void* source) {

      return Status::OK();
    }

}  // namespace npu

static std::unique_ptr<xla::TransferManager> CreateNpuTransferManager() {
    return xla::MakeUnique<npu::NpuTransferManager>();
}

static bool InitModule() {
    xla::TransferManager::RegisterTransferManager(npu::npuPlatformId,
                                                  &CreateNpuTransferManager);
    return true;
}

static bool module_initialized = InitModule();
