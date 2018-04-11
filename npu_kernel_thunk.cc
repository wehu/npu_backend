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
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"
//#include "npu_stream.h"

namespace se = ::perftools::gputools;

namespace npu {


    using ComputeFunctionType = void (*)(int8*, int8*);

    NpuKernelThunk::NpuKernelThunk(
            tensorflow::gtl::ArraySlice<const BufferAllocation *> args,
            const string &kernel_name, const HloInstruction *hlo_instruction,
            xla::cpu::SimpleOrcJIT* jit)
            : NpuThunk(Kind::kKernel, hlo_instruction),
              args_(args.begin(), args.end()),
              kernel_name_(kernel_name),
              jit_(jit) {}

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

        /*se::StreamExecutor *executor = stream->parent();
        const se::KernelBase *kernel = nullptr;
        {
            tensorflow::mutex_lock lock(mutex_);
            auto it = kernel_cache_.find(executor);
            if (kernel_cache_.end() == it) {
                it = kernel_cache_.emplace(executor, se::KernelBase(executor)).first;
                if (!executor->GetKernel(*loader_spec_, &it->second)) {
                    return InternalError("Unable to load kernel %s", kernel_name_.c_str());
                }
            }
            kernel = &it->second;
        }*/

        // Resolve symbols in the constructor rather than at execution time to avoid
        // races because FindSymbol is not thread safe.
        llvm::JITSymbol sym = jit_->FindCompiledSymbol(kernel_name_);
        // We expect to find the symbol provided with entry_function_name; otherwise
        // this is an internal error.
        CHECK(sym) << "Symbol " << kernel_name_ << " not found.";
        // getAddress can do work under the hood in the jit, so it needs to be
        // guarded by the mutex.
        auto compute_function =
                reinterpret_cast<ComputeFunctionType>(llvm::cantFail(sym.getAddress()));

        VLOG(0) << "Launching " << kernel_name_;
        // Launch the kernel with potentially multiple blocks and threads.
        static constexpr int kKernelArgsLimit = 1024;
        auto kernel_args = MakeUnique<se::KernelArgsArray<kKernelArgsLimit>>();
        for (const BufferAllocation *arg : args_) {
            const auto &buf = buffer_allocations.GetDeviceAddress(arg->index());
            kernel_args->add_device_memory_argument(buf);
            VLOG(0) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << " ("
                    << buf.size() << "B)";
        }
        auto data = kernel_args->argument_addresses().data();
        VLOG(0) << args_.size();
        compute_function((int8*)data[0], (int8*)data[1]);
        VLOG(0) << "bbbb";
        /*if (!stream->parent()->Launch(
                stream, se::ThreadDim(launch_dimensions.threads_per_block()),
                se::BlockDim(launch_dimensions.block_count()), *kernel,
                *kernel_args)) {
            return InternalError("Unable to launch kernel %s", kernel_name_.c_str());
        }*/
        return tensorflow::Status::OK();
    }

}  // namespace npu
