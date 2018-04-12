//
// Created by wehu on 18-4-12.
//

#include "npu_conditional_thunk.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
    namespace npu {

        NpuConditionalThunk::NpuConditionalThunk(
                const BufferAllocation::Slice& predicate_buffer_index,
                const BufferAllocation::Slice& true_operand_buffer_index,
                const BufferAllocation::Slice& false_operand_buffer_index,
                NpuThunkSequence true_thunk_sequence, NpuThunkSequence false_thunk_sequence,
                const HloInstruction* hlo)
                : NpuThunk(Kind::kConditional, hlo),
                  predicate_buffer_index_(predicate_buffer_index),
                  true_operand_buffer_index_(true_operand_buffer_index),
                  false_operand_buffer_index_(false_operand_buffer_index),
                  true_thunk_(std::move(true_thunk_sequence), hlo),
                  false_thunk_(std::move(false_thunk_sequence), hlo) {}

        Status NpuConditionalThunk::Initialize(const NpuExecutable& executable) {
            TF_RETURN_IF_ERROR(true_thunk_.Initialize(executable));
            TF_RETURN_IF_ERROR(false_thunk_.Initialize(executable));
            return Status::OK();
        }

        Status NpuConditionalThunk::ExecuteOnStream(
                const NpuBufferAllocations& buffer_allocations,
                perftools::gputools::Stream* stream) {
            // Copy the predicate value from device.
            bool predicate;
            perftools::gputools::DeviceMemoryBase predicate_address =
                    buffer_allocations.GetDeviceAddress(predicate_buffer_index_);
            stream->ThenMemcpy(&predicate, predicate_address, sizeof(bool));

            Status block_status = stream->BlockHostUntilDone();
            if (!block_status.ok()) {
                return InternalError("Failed to retrieve predicate value on stream %p: %s.",
                                     stream, block_status.error_message().c_str());
            }

            // Execute the true or the false computation depending on the value of the
            // predicate.
            if (predicate) {
                TF_RETURN_IF_ERROR(true_thunk_.ExecuteOnStream(buffer_allocations, stream));
            } else {
                TF_RETURN_IF_ERROR(
                        false_thunk_.ExecuteOnStream(buffer_allocations, stream));
            }

            return Status::OK();
        }

    }  // namespace gpu
}  // namespace xla