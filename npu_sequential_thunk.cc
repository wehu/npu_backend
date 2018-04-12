//
// Created by wehu on 18-4-12.
//

#include "npu_sequential_thunk.h"

#include "tensorflow/core/lib/core/errors.h"

namespace xla {
    namespace npu {

        NpuSequentialThunk::NpuSequentialThunk(std::vector<std::unique_ptr<NpuThunk>>&& thunks,
                                         const HloInstruction* hlo)
                : NpuThunk(Kind::kSequential, hlo), thunks_(std::move(thunks)) {}

        tensorflow::Status NpuSequentialThunk::Initialize(
                const NpuExecutable& executable) {
            for (auto& thunk : thunks_) {
                TF_RETURN_IF_ERROR(thunk->Initialize(executable));
            }
            return tensorflow::Status::OK();
        }

        tensorflow::Status NpuSequentialThunk::ExecuteOnStream(
                const NpuBufferAllocations& buffer_allocations,
                perftools::gputools::Stream* stream) {
            for (const auto& thunk : thunks_) {
                TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(buffer_allocations, stream));
            }
            return tensorflow::Status::OK();
        }

    }  // namespace npu
}  // namespace xla