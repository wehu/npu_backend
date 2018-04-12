//
// Created by wehu on 18-4-12.
//

#ifndef TENSORFLOW_NPU_CONDITIONAL_THUNK_H
#define TENSORFLOW_NPU_CONDITIONAL_THUNK_H


#include "npu_buffer_allocations.h"
#include "npu_sequential_thunk.h"
#include "npu_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
    namespace npu {

        // ConditionalThunk implements the conditional instruction on GPU by reading the
        // predicate of the conditional and executing the true or the false computation
        // depending on the value of the predicate.
        //
        // ConditionalThunk assumes that the buffers of the conditional result and the
        // result of the true and false computations share the same allocation. Also,
        // the buffers of the true operand of the conditional and that of the parameter
        // instruction of the true computation share the same allocation. Similarly, the
        // buffers of the false operand and that of the parameter instruction of the
        // false computation share the same allocation.
        class NpuConditionalThunk : public NpuThunk {
        public:
            NpuConditionalThunk(const BufferAllocation::Slice& predicate_buffer_index,
                             const BufferAllocation::Slice& true_operand_buffer_index,
                             const BufferAllocation::Slice& false_operand_buffer_index,
                             NpuThunkSequence true_thunk_sequence,
                             NpuThunkSequence false_thunk_sequence,
                             const HloInstruction* hlo);

            Status Initialize(const NpuExecutable& executable) override;
            Status ExecuteOnStream(const NpuBufferAllocations& buffer_allocations,
                                   perftools::gputools::Stream* stream) override;

        private:
            BufferAllocation::Slice predicate_buffer_index_;
            BufferAllocation::Slice true_operand_buffer_index_;
            BufferAllocation::Slice false_operand_buffer_index_;
            NpuSequentialThunk true_thunk_;
            NpuSequentialThunk false_thunk_;

            SE_DISALLOW_COPY_AND_ASSIGN(NpuConditionalThunk);
        };

    }  // namespace gpu
}  // namespace xla

#endif //TENSORFLOW_NPU_CONDITIONAL_THUNK_H
