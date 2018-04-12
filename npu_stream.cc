#include "npu_stream.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {
    namespace npu {

        using namespace perftools::gputools;

        NpuStream::NpuStream(NpuExecutor *parent)
                : parent_(parent),
                  host_executor_(new port::ThreadPool(port::Env::Default(),
                                                      port::ThreadOptions(),
                                                      "host_executor", kExecutorThreads)) {}

        bool NpuStream::Init() {
            return true;
        }

        void NpuStream::Destroy() {
        }

        bool NpuStream::IsIdle() const {
            return pending_tasks_ == 0;
        }

        bool NpuStream::EnqueueTask(std::function<void()> task) {
            {
                mutex_lock lock(mu_);
                ++pending_tasks_;
            }
            host_executor_->Schedule([this, task]() {
                task();
                {
                    mutex_lock lock(mu_);
                    --pending_tasks_;
                }
                completion_condition_.notify_all();
            });
            return true;
        }

        void NpuStream::BlockUntilDone() {
            mutex_lock lock(mu_);
            while (pending_tasks_ != 0) {
                completion_condition_.wait(lock);
            }
        }

        NpuStream *AsNpuStream(Stream *stream) {
            DCHECK(stream != nullptr);
            return static_cast<NpuStream *>(stream->implementation());
        }

    }  // namespace npu
} // namespace xla
