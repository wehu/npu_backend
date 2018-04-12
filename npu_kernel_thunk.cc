//
// Created by wehu on 18-4-11.
//

#include "npu_kernel_thunk.h"
#include "npu_stream.h"
#include "npu_executable.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"

namespace se = ::perftools::gputools;

namespace npu {

    using ComputeFunctionType0 = void (*)();
    using ComputeFunctionType1 = void (*)(char*);
    using ComputeFunctionType2 = void (*)(char*, char*);
    using ComputeFunctionType3 = void (*)(char*, char*, char*);
    using ComputeFunctionType4 = void (*)(char*, char*, char*, char*);
    using ComputeFunctionType5 = void (*)(char*, char*, char*, char*, char*);
    using ComputeFunctionType6 = void (*)(char*, char*, char*, char*, char*, char*);

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

        VLOG(3) << "Launching " << kernel_name_;

        std::vector<char*> addrs;
        for (const BufferAllocation *arg : args_) {
            const auto &buf = buffer_allocations.GetDeviceAddress(arg->index());
            addrs.push_back((char*)buf.opaque());
            VLOG(3) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << " ("
                    << buf.size() << "B)";
        }

        llvm::JITSymbol sym = jit_->FindCompiledSymbol(kernel_name_);
        // We expect to find the symbol provided with entry_function_name; otherwise
        // this is an internal error.
        CHECK(sym) << "Symbol " << kernel_name_ << " not found.";
        // getAddress can do work under the hood in the jit, so it needs to be
        // guarded by the mutex.

        auto sym_addr = llvm::cantFail(sym.getAddress());

        AsNpuStream(stream)->EnqueueTask(
                [addrs, sym_addr]() {

                    switch (addrs.size()) {
                        case 0: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType0>(sym_addr);
                            compute_function();
                            break;
                        }
                        case 1: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType1>(sym_addr);
                            compute_function(addrs[0]);
                            break;
                        }
                        case 2: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType2>(sym_addr);
                            compute_function(addrs[0], addrs[1]);
                            break;
                        }
                        case 3: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType3>(sym_addr);
                            compute_function(addrs[0], addrs[1], addrs[2]);
                            break;
                        }
                        case 4: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType4>(sym_addr);
                            compute_function(addrs[0], addrs[1], addrs[2], addrs[3]);
                            break;
                        }
                        case 5: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType5>(sym_addr);
                            compute_function(addrs[0], addrs[1], addrs[2], addrs[3], addrs[4]);
                            break;
                        }
                        case 6: {
                            auto compute_function =
                                    reinterpret_cast<ComputeFunctionType6>(sym_addr);
                            compute_function(addrs[0], addrs[1], addrs[2], addrs[3], addrs[4], addrs[5]);
                            break;
                        }
                    }
                });

        return tensorflow::Status::OK();
    }

}  // namespace npu
