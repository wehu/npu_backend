//
// Created by wehu on 18-4-11.
//

#ifndef TENSORFLOW_NPU_TUPLE_THUNK_H
#define TENSORFLOW_NPU_TUPLE_THUNK_H

#include "npu_executable.h"
#include "npu_thunk.h"

#include <vector>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
    namespace npu {

        class NpuTupleThunk : public NpuThunk {
        public:
            NpuTupleThunk(tensorflow::gtl::ArraySlice<BufferAllocation::Slice>
                          tuple_element_buffers,
                          const BufferAllocation::Slice &dest_buffer,
                          const HloInstruction *hlo_instruction)
                    : NpuThunk(Kind::kTuple, hlo_instruction),
                      tuple_element_buffers_(tuple_element_buffers.begin(),
                                             tuple_element_buffers.end()),
                      dest_buffer_(dest_buffer) {}

            tensorflow::Status ExecuteOnStream(
                    const NpuBufferAllocations &buffer_allocations,
                    perftools::gputools::Stream *stream) override;

        private:
            const std::vector<BufferAllocation::Slice> tuple_element_buffers_;
            const BufferAllocation::Slice dest_buffer_;

            SE_DISALLOW_COPY_AND_ASSIGN(NpuTupleThunk);
        };

    }  // namespace npu
} // namespace xla

#endif //TENSORFLOW_NPU_TUPLE_THUNK_H
