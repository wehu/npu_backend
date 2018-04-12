//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_THUNK_H
#define TENSORFLOW_NPU_THUNK_H

#include "npu_buffer_allocations.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
    namespace npu {

        class NpuExecutable;

        class NpuThunk {
        public:
            enum class Kind {
                kConditional,
                kConvolution,
                kCopy,
                kFft,
                kInfeed,
                kSequential,
                kTuple,
                kWhile,
                kKernel,
            };

            explicit NpuThunk(Kind kind, const HloInstruction *hlo_instruction)
                    : kind_(kind), hlo_instruction_(hlo_instruction) {}

            virtual ~NpuThunk() {}

            NpuThunk(const NpuThunk &) = delete;

            NpuThunk &operator=(const NpuThunk &) = delete;

            Kind kind() const { return kind_; }

            const HloInstruction *hlo_instruction() const { return hlo_instruction_; }

            virtual tensorflow::Status Initialize(const NpuExecutable &executable) {
                return tensorflow::Status::OK();
            }

            virtual bool ShouldHaltAllActivityBeforeRunning(
                    perftools::gputools::Stream * /*stream*/) {
                return false;
            }

            virtual bool ShouldBlockFutureThunks() { return false; }

            virtual tensorflow::Status ExecuteOnStream(
                    const NpuBufferAllocations &buffer_allocations,
                    perftools::gputools::Stream *stream) = 0;

        private:
            Kind kind_;
            const HloInstruction *hlo_instruction_;
        };

        using NpuThunkSequence = std::vector<std::unique_ptr<NpuThunk>>;

    }  // namespace npu
} // namespace xla


#endif //TENSORFLOW_NPU_THUNK_H
