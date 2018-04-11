//
// Created by wehu on 18-4-11.
//

#include "npu_kernel_thunk.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "npu_executable.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace npu {

    NpuKernelThunk::NpuKernelThunk(
            tensorflow::gtl::ArraySlice<const BufferAllocation *> args,
            const string &kernel_name, const HloInstruction *hlo_instruction)
            : NpuThunk(Kind::kKernel, hlo_instruction),
              args_(args.begin(), args.end()),
              kernel_name_(kernel_name) {}

    tensorflow::Status NpuKernelThunk::Initialize(const NpuExecutable &executable) {
        tensorflow::mutex_lock lock(mutex_);
        if (loader_spec_) {
            // Already initialized by another thread.
            return tensorflow::Status::OK();
        }

        return tensorflow::Status::OK();
    }

    tensorflow::Status NpuKernelThunk::ExecuteOnStream(
            const NpuBufferAllocations &buffer_allocations, se::Stream *stream) {
        // Load the kernel.

        se::StreamExecutor *executor = stream->parent();
        const se::KernelBase *kernel = nullptr;
        {
            tensorflow::mutex_lock lock(mutex_);
            auto it = kernel_cache_.find(executor);
            if (kernel_cache_.end() == it) {
                it = kernel_cache_.emplace(executor, se::KernelBase(executor)).first;
                /*if (!executor->GetKernel(*loader_spec_, &it->second)) {
                    return InternalError("Unable to load kernel %s", kernel_name_.c_str());
                }*/
            }
            kernel = &it->second;
        }

        VLOG(0) << "Launching " << kernel_name_;
        // Launch the kernel with potentially multiple blocks and threads.
        static constexpr int kKernelArgsLimit = 1024;
        auto kernel_args = MakeUnique<se::KernelArgsArray<kKernelArgsLimit>>();
        for (const BufferAllocation *arg : args_) {
            const auto &buf = buffer_allocations.GetDeviceAddress(arg->index());
            kernel_args->add_device_memory_argument(buf);
            VLOG(3) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << " ("
                    << buf.size() << "B)";
        }
        /*if (!stream->parent()->Launch(
                stream, se::ThreadDim(launch_dimensions.threads_per_block()),
                se::BlockDim(launch_dimensions.block_count()), *kernel,
                *kernel_args)) {
            return InternalError("Unable to launch kernel %s", kernel_name_.c_str());
        }*/
        return tensorflow::Status::OK();
    }

}  // namespace npu
