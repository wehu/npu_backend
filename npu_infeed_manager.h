//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_INFEED_MANAGER_H
#define TENSORFLOW_NPU_INFEED_MANAGER_H

#include <deque>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace npu {

    using namespace xla;

    class NpuInfeedBuffer {
    public:
        NpuInfeedBuffer(perftools::gputools::StreamExecutor* executor, int64 length)
                : executor_(executor), length_(length) {
            device_memory_ = executor_->AllocateArray<uint8>(length);
            CHECK(!device_memory_.is_null());
        }

        ~NpuInfeedBuffer() { executor_->Deallocate(&device_memory_); }

        int64 length() const { return length_; }

        void Done() { delete this; }

        perftools::gputools::DeviceMemoryBase* device_memory() {
            return &device_memory_;
        }

    private:
        perftools::gputools::StreamExecutor* executor_;  // Not owned.
        const int64 length_;
        perftools::gputools::DeviceMemoryBase device_memory_;
    };

    class NpuInfeedManager {
    public:
        NpuInfeedManager();

        void Reset();

        void EnqueueBuffers(const std::vector<NpuInfeedBuffer*>& buffers);

        NpuInfeedBuffer* BlockingDequeueBuffer();

        void ReleaseBuffers(const std::vector<NpuInfeedBuffer*>& buffers);

        perftools::gputools::Stream* GetStream(
                perftools::gputools::StreamExecutor* executor);

    private:
        tensorflow::mutex mu_;

        tensorflow::condition_variable cv_;

        std::deque<NpuInfeedBuffer*> enqueued_buffer_;

        tensorflow::gtl::FlatSet<const NpuInfeedBuffer*> dequeued_buffer_;

        std::unique_ptr<perftools::gputools::Stream> host_to_device_stream_;

        perftools::gputools::StreamExecutor* host_to_device_executor_;
    };


    NpuInfeedManager* GetOrCreateNpuInfeedManager();

}  // namespace npu

#endif //TENSORFLOW_NPU_INFEED_MANAGER_H
