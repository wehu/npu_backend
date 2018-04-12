#ifndef TENSORFLOW_NPU_STREAM_H_
#define TENSORFLOW_NPU_STREAM_H_

#include "npu_executor.h"

#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = perftools::gputools;

namespace xla {
    namespace npu {

        class NpuStream : public se::internal::StreamInterface {
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
            std::unique_ptr<se::port::ThreadPool> host_executor_;

            se::mutex mu_;
            int pending_tasks_ GUARDED_BY(mu_) = 0;
            se::condition_variable completion_condition_;

        };

        NpuStream *AsNpuStream(se::Stream *stream);

    }  // namespace npu
} // namespace xal

#endif  // TENSORFLOW_NPU_STREAM_H_
