//
// Created by wehu on 18-4-12.
//

#include "npu_while_thunk.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
    namespace npu {

        NpuWhileThunk::NpuWhileThunk(
                const BufferAllocation::Slice& condition_result_buffer_index,
                std::unique_ptr<NpuThunkSequence> condition_thunk_sequence,
                std::unique_ptr<NpuThunkSequence> body_thunk_sequence,
                const HloInstruction* hlo)
                : NpuThunk(Kind::kWhile, hlo),
                  condition_result_buffer_index_(condition_result_buffer_index),
                  condition_thunk_sequence_(MakeUnique<NpuSequentialThunk>(
                          std::move(*condition_thunk_sequence), hlo)),
                  body_thunk_sequence_(
                          MakeUnique<NpuSequentialThunk>(std::move(*body_thunk_sequence), hlo)) {}

        Status NpuWhileThunk::Initialize(const NpuExecutable& executable) {
            TF_RETURN_IF_ERROR(condition_thunk_sequence_->Initialize(executable));
            TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(executable));
            return Status::OK();
        }

        Status NpuWhileThunk::ExecuteOnStream(const NpuBufferAllocations& buffer_allocations,
                                           perftools::gputools::Stream* stream) {
            perftools::gputools::DeviceMemoryBase condition_result_data =
                    buffer_allocations.GetDeviceAddress(condition_result_buffer_index_);

            while (true) {
                // Invoke thunk sequence for while 'condition' computation.
                TF_RETURN_IF_ERROR(
                        condition_thunk_sequence_->ExecuteOnStream(buffer_allocations, stream));

                // Copy the result of condition computation and break the loop if 'false'.
                bool condition_result;
                stream->ThenMemcpy(&condition_result, condition_result_data, sizeof(bool));
                Status block_status = stream->BlockHostUntilDone();
                if (!block_status.ok()) {
                    return InternalError(
                            "Failed to complete all kernels launched on stream %p: %s", stream,
                            block_status.error_message().c_str());
                }

                if (!condition_result) {
                    break;
                }

                // Invoke thunk sequence for while 'body' computation.
                TF_RETURN_IF_ERROR(
                        body_thunk_sequence_->ExecuteOnStream(buffer_allocations, stream));
            }
            return Status::OK();
        }

    }  // namespace npu
}  // namespace xla