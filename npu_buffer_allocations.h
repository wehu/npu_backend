//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_BUFFER_ALLOCATIONS_H
#define TENSORFLOW_NPU_BUFFER_ALLOCATIONS_H

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

#include "npu_constants.h"

namespace npu {

    using namespace xla;

    class NpuBufferAllocations {
    public:
        class Builder {
        public:

            void RegisterBuffer(BufferAllocation::Index index,
                                perftools::gputools::DeviceMemoryBase address);

            StatusOr <std::unique_ptr<NpuBufferAllocations>> Build(
                    const BufferAssignment &buffer_assignment, int device_ordinal,
                    DeviceMemoryAllocator *memory_allocator);

        private:
            std::map<BufferAllocation::Index, perftools::gputools::DeviceMemoryBase>
                    registered_buffers_;
        };

        NpuBufferAllocations(const NpuBufferAllocations &) = delete;

        NpuBufferAllocations &operator=(const NpuBufferAllocations &) = delete;

        DeviceMemoryAllocator *memory_allocator() const { return memory_allocator_; }

        int device_ordinal() const { return device_ordinal_; }

        perftools::gputools::DeviceMemoryBase GetDeviceAddress(
                BufferAllocation::Index buffer_index) const;

        perftools::gputools::DeviceMemoryBase GetDeviceAddress(
                const BufferAllocation::Slice &buffer_slice) const;

        perftools::gputools::DeviceMemoryBase GetTempBufferBase() const {
            return temp_buffer_base_;
        }

        tensorflow::Status TearDown(
                const std::set<perftools::gputools::DeviceMemoryBase> &live_addresses,
                const BufferAssignment &buffer_assignment);

    private:
        NpuBufferAllocations(BufferAllocation::Index buffer_count, int device_ordinal,
                          DeviceMemoryAllocator *memory_allocator)
                : buffers_(buffer_count),
                  device_ordinal_(device_ordinal),
                  memory_allocator_(memory_allocator) {}

        void SetBuffer(BufferAllocation::Index buffer_index,
                       perftools::gputools::DeviceMemoryBase buffer);

        std::vector<perftools::gputools::DeviceMemoryBase> buffers_;

        perftools::gputools::DeviceMemoryBase temp_buffer_base_;

        int device_ordinal_;

        DeviceMemoryAllocator *memory_allocator_;
    };

}  // namespace npu

#endif //TENSORFLOW_NPU_BUFFER_ALLOCATIONS_H
