//
// Created by wehu on 18-4-12.
//

#ifndef TENSORFLOW_NPU_WHILE_THUNK_H
#define TENSORFLOW_NPU_WHILE_THUNK_H

#include "npu_buffer_allocations.h"
#include "npu_sequential_thunk.h"
#include "npu_thunk.h"

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
    namespace npu {

        // WhileThunk implements the while instruction on GPU by invoking a thunk
        // sequence for the while 'condition' computation, and (conditionally) another
        // thunk sequence for the while 'body' computation. WhileThunk assumes that
        // buffers for the following set of while-related instructions share the same
        // allocation:
        //   init, condition.parameter, body.parameter, body.root, while.result
        // WhileThunk synchronizes the stream to test the result of the 'condition'
        // computation.
        class NpuWhileThunk : public NpuThunk {
        public:
            // Constructs a WhileThunk to compute while instruction 'hlo'.
            NpuWhileThunk(const BufferAllocation::Slice& condition_result_buffer_index,
                       std::unique_ptr<NpuThunkSequence> condition_thunk_sequence,
                       std::unique_ptr<NpuThunkSequence> body_thunk_sequence,
                       const HloInstruction* hlo);

            Status Initialize(const NpuExecutable& executable) override;
            Status ExecuteOnStream(const NpuBufferAllocations& buffer_allocations,
                                   perftools::gputools::Stream* stream) override;

        private:
            const BufferAllocation::Slice condition_result_buffer_index_;
            std::unique_ptr<NpuSequentialThunk> condition_thunk_sequence_;
            std::unique_ptr<NpuSequentialThunk> body_thunk_sequence_;

            SE_DISALLOW_COPY_AND_ASSIGN(NpuWhileThunk);

        };

    }  // namespace npu
}  // namespace xla

#endif //TENSORFLOW_NPU_WHILE_THUNK_H
