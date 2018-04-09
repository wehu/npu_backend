#ifndef TENSORFLOW_NPU_STREAM_H_
#define TENSORFLOW_NPU_STREAM_H_

#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "npu_executor.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuStream : public internal::StreamInterface {
    public:
        explicit NpuStream(NpuExecutor *parent)
                : parent_(parent) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
        ~NpuStream() override {}

        bool Init();

        void Destroy();

        bool IsIdle() const;
  
        NpuExecutor *parent() const { return parent_; }

    private:
        NpuExecutor *parent_;  // Executor that spawned this stream.

    };

    NpuStream *AsNpuStream(Stream *stream);

}  // namespace npu

#endif  // TENSORFLOW_NPU_STREAM_H_
