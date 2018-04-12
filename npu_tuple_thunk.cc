//
// Created by wehu on 18-4-11.
//

#include "npu_tuple_thunk.h"

#include "tensorflow/compiler/xla/util.h"

namespace se = ::perftools::gputools;

namespace xla {
    namespace npu {

        tensorflow::Status NpuTupleThunk::ExecuteOnStream(
                const NpuBufferAllocations &buffer_allocations, se::Stream *stream) {
            std::vector<void *> tuple_element_buffer_addresses;
            for (BufferAllocation::Slice tuple_element_buffer : tuple_element_buffers_) {
                tuple_element_buffer_addresses.push_back(
                        buffer_allocations.GetDeviceAddress(tuple_element_buffer).opaque());
            }
            se::DeviceMemory<void *> dest_buffer_address(
                    buffer_allocations.GetDeviceAddress(dest_buffer_));

            auto host_size = tuple_element_buffer_addresses.size() * sizeof(void *);
            if (!stream
                    ->ThenMemcpy(&dest_buffer_address,
                                 tuple_element_buffer_addresses.data(), host_size)
                    .ok()) {
                return InternalError(
                        "Unable to launch MemcpyH2D from %p to %p with size %lu",
                        tuple_element_buffer_addresses.data(), dest_buffer_address.opaque(),
                        sizeof(void *) * tuple_element_buffer_addresses.size());
            }
            return tensorflow::Status::OK();
        }

    }  // namespace npu
} // namespace xla