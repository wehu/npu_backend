#ifndef TENSORFLOW_NPU_STREAM_H_
#define TENSORFLOW_NPU_STREAM_H_

#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#include "npu_executor.h"

namespace npu {

    class NpuStream : public perftools::gputools::internal::StreamInterface {
    public:
        NpuStream(NpuExecutor *parent);

        ~NpuStream() override {}

        bool Init();

        void Destroy();

        bool IsIdle() const;

        bool EnqueueTask(std::function<void()> task);

        void *CudaStreamHack() override { return nullptr; }
        void **CudaStreamMemberHack() override { return nullptr; }

        void BlockUntilDone();

        NpuExecutor *parent() const { return parent_; }

    private:
        NpuExecutor *parent_;

        // Use only one thread and own task queue to preserve FIFO ordering
        // for the operations enqueued by any given stream.
        static const int kExecutorThreads = 1;
        std::unique_ptr<port::ThreadPool> host_executor_;

        mutex mu_;
        int pending_tasks_ GUARDED_BY(mu_) = 0;
        condition_variable completion_condition_;

    };

    NpuStream *AsNpuStream(Stream *stream);

}  // namespace npu

#endif  // TENSORFLOW_NPU_STREAM_H_
