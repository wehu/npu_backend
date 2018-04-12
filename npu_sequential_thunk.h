//
// Created by wehu on 18-4-12.
//

#ifndef TENSORFLOW_NPU_SEQUENTIAL_THUNK_H
#define TENSORFLOW_NPU_SEQUENTIAL_THUNK_H

#include "npu_buffer_allocations.h"
#include "npu_thunk.h"

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
    namespace npu {

        // A thunk that wraps a list of sub-thunks. Executing this thunk executes all
        // the sub-thunks sequentially. This is useful to implement instructions that
        // require multiple kernel launches or library calls.
        class NpuSequentialThunk : public NpuThunk {
        public:
            NpuSequentialThunk(std::vector<std::unique_ptr<NpuThunk>>&& thunks,
                            const HloInstruction* hlo);

            const std::vector<std::unique_ptr<NpuThunk>>& thunks() const { return thunks_; }

            tensorflow::Status Initialize(const NpuExecutable& executable) override;
            tensorflow::Status ExecuteOnStream(
                    const NpuBufferAllocations& buffer_allocations,
                    perftools::gputools::Stream* stream) override;

        private:
            // The list of sub-thunks.
            std::vector<std::unique_ptr<NpuThunk>> thunks_;

            SE_DISALLOW_COPY_AND_ASSIGN(NpuSequentialThunk);

        };

    }  // namespace npu
}  // namespace xla

#endif //TENSORFLOW_NPU_SEQUENTIAL_THUNK_H
