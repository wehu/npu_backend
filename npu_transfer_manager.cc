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

namespace xla {
    namespace npu {

        NpuTransferManager::NpuTransferManager()
                : GenericTransferManager(
                npuPlatformId,
                /*pointer_size=*/llvm::DataLayout(npu::NpuCompiler::kDataLayout)
                        .getPointerSize(0 /* default address space */)) {}

        Status NpuTransferManager::TransferLiteralToInfeed(se::StreamExecutor *executor,
                                                           const Literal &literal) {
            const Shape &shape = literal.shape();
            VLOG(2) << "Transferring literal to infeed with shape: "
                    << ShapeUtil::HumanString(shape);

            if (!ShapeUtil::IsTuple(shape)) {
                int64 size = GetByteSizeRequirement(shape);
                return TransferBufferToInfeed(executor, size, literal.untyped_data());
            }

            if (ShapeUtil::IsNestedTuple(shape)) {
                return Unimplemented(
                        "Infeed with a nested tuple shape is not supported: %s",
                        ShapeUtil::HumanString(literal.shape()).c_str());
            }

            std::vector<NpuInfeedBuffer *> buffers;
            buffers.reserve(ShapeUtil::TupleElementCount(shape));
            auto cleanup = tensorflow::gtl::MakeCleanup([buffers]() {
                for (NpuInfeedBuffer *b : buffers) {
                    b->Done();
                }
            });

            for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
                const Shape &tuple_element_shape =
                        ShapeUtil::GetTupleElementShape(shape, i);
                int64 tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
                TF_ASSIGN_OR_RETURN(
                        NpuInfeedBuffer *buffer,
                        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                                       literal.untyped_data({i})));
                buffers.push_back(buffer);
            }

            cleanup.release();
            return EnqueueBuffersToInfeed(executor, buffers);
        }

        Status NpuTransferManager::TransferBufferToInfeed(se::StreamExecutor *executor,
                                                          int64 size,
                                                          const void *source) {
            TF_ASSIGN_OR_RETURN(NpuInfeedBuffer *buffer,
                                TransferBufferToInfeedInternal(executor, size, source));
            return EnqueueBuffersToInfeed(executor, {buffer});
        }

        Status NpuTransferManager::EnqueueBuffersToInfeed(
                se::StreamExecutor *executor, std::vector<NpuInfeedBuffer *> buffers) {
            NpuInfeedManager *infeed_manager = GetOrCreateNpuInfeedManager();
            se::Stream *stream = infeed_manager->GetStream(executor);

            Status block_status = stream->BlockHostUntilDone();
            if (!block_status.ok()) {
                for (NpuInfeedBuffer *b : buffers) {
                    b->Done();
                }
                return InternalError("Failed to complete data transfer on stream %p: %s",
                                     stream, block_status.error_message().c_str());
            }

            infeed_manager->EnqueueBuffers(buffers);

            VLOG(2) << "Infeed data transferred";

            return Status::OK();
        }

        StatusOr<NpuInfeedBuffer *> NpuTransferManager::TransferBufferToInfeedInternal(
                se::StreamExecutor *executor, int64 size, const void *source) {
            if (size > std::numeric_limits<int32>::max()) {
                return InvalidArgument("Infeed shape is too large: needs %lld bytes", size);
            }

            if (size == 0) {
                return InvalidArgument("Infeed shape needs 0 bytes");
            }

            NpuInfeedManager *infeed_manager = GetOrCreateNpuInfeedManager();
            se::Stream *stream = infeed_manager->GetStream(executor);
            if (stream == nullptr) {
                return InternalError("Failed to obtain a stream");
            }

            NpuInfeedBuffer *buffer = new NpuInfeedBuffer(executor, size);
            stream->ThenMemcpy(buffer->device_memory(), source, size);

            VLOG(2) << "Queued infeed data on stream " << stream;

            return buffer;
        }

    }  // namespace npu
} // namespace xla

static std::unique_ptr<xla::TransferManager> CreateNpuTransferManager() {
    return xla::MakeUnique<xla::npu::NpuTransferManager>();
}

static bool InitModule() {
    xla::TransferManager::RegisterTransferManager(xla::npu::npuPlatformId,
                                                  &CreateNpuTransferManager);
    return true;
}

static bool module_initialized = InitModule();
