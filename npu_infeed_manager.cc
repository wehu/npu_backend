//
// Created by wehu on 18-4-10.
//

#include "npu_infeed_manager.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/core/platform/logging.h"

namespace se = ::perftools::gputools;

namespace xla {
    namespace npu {

        NpuInfeedManager::NpuInfeedManager() : host_to_device_executor_(nullptr) {}

        void NpuInfeedManager::Reset() {
            tensorflow::mutex_lock l(mu_);
            CHECK(dequeued_buffer_.empty());
            for (auto buffer : enqueued_buffer_) {
                buffer->Done();
            }
            enqueued_buffer_.clear();
        }

        void NpuInfeedManager::EnqueueBuffers(const std::vector<NpuInfeedBuffer *> &buffers) {
            tensorflow::mutex_lock l(mu_);
            bool was_empty = enqueued_buffer_.empty();
            for (NpuInfeedBuffer *b : buffers) {
                enqueued_buffer_.push_back(b);
            }
            if (was_empty) {
                // This has the potential to suffer from the notified thread
                // immediately trying and failing to acquire mu_, but seems
                // preferable to the alternative of notifying outside the lock
                // on every enqueue.
                cv_.notify_one();
            }
        }

        NpuInfeedBuffer *NpuInfeedManager::BlockingDequeueBuffer() {
            tensorflow::mutex_lock l(mu_);
            while (enqueued_buffer_.empty()) {
                cv_.wait(l);
            }
            NpuInfeedBuffer *current_buffer = enqueued_buffer_.front();
            enqueued_buffer_.pop_front();
            dequeued_buffer_.insert(current_buffer);
            return current_buffer;
        }

        void NpuInfeedManager::ReleaseBuffers(const std::vector<NpuInfeedBuffer *> &buffers) {
            {
                tensorflow::mutex_lock l(mu_);
                for (NpuInfeedBuffer *b : buffers) {
                    CHECK(ContainsKey(dequeued_buffer_, b));
                    dequeued_buffer_.erase(b);
                }
            }
            for (NpuInfeedBuffer *b : buffers) {
                b->Done();
            }
        }

        se::Stream *NpuInfeedManager::GetStream(se::StreamExecutor *executor) {
            if (host_to_device_executor_ == nullptr) {
                host_to_device_executor_ = executor;
                host_to_device_stream_ = MakeUnique<se::Stream>(executor);
                host_to_device_stream_->Init();
            }

            if (executor != host_to_device_executor_) {
                // The requested executor must be the same as the one for which
                // the stream is cached.
                return nullptr;
            }

            return host_to_device_stream_.get();
        }

        NpuInfeedManager *GetOrCreateNpuInfeedManager() {
            static NpuInfeedManager *manager = new NpuInfeedManager;
            return manager;
        }

    }  // namespace npu
} // namespace xla