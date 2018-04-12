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
#include "boost/preprocessor/repetition.hpp"

namespace se = ::perftools::gputools;

#define MAX_FUNC_PARAMS_SIZE 24

#define PARAM_TYPE(z, n, t) t

#define DECLARE_FUNC_TYPE(z, n, used) \
  using ComputeFunctionType ## n  = void (*)(BOOST_PP_ENUM(n, PARAM_TYPE, char *));

#define PARAM_REF(z, n, p) p[n]

#define SWITCH_CASE(z, n, p) \
  case n: { \
    auto compute_function = \
        reinterpret_cast<ComputeFunctionType ## n>(sym_addr); \
    compute_function(BOOST_PP_ENUM(n, PARAM_REF, p)); \
    break; \
  }

namespace xla {
    namespace npu {

        BOOST_PP_REPEAT(MAX_FUNC_PARAMS_SIZE, DECLARE_FUNC_TYPE, ~)

        NpuKernelThunk::NpuKernelThunk(
                tensorflow::gtl::ArraySlice<const BufferAllocation *> args,
                const string &kernel_name, const HloInstruction *hlo_instruction,
                xla::cpu::SimpleOrcJIT *jit)
                : NpuThunk(Kind::kKernel, hlo_instruction),
                  args_(args.begin(), args.end()),
                  kernel_name_(kernel_name),
                  jit_(jit) {}

        tensorflow::Status NpuKernelThunk::ExecuteOnStream(
                const NpuBufferAllocations &buffer_allocations, se::Stream *stream) {

            VLOG(3) << "Launching " << kernel_name_;

            if (args_.size() > MAX_FUNC_PARAMS_SIZE) {
                return Unimplemented("kernel parameters size is too big on NPU.");
            }

            std::vector<char *> params;
            for (const BufferAllocation *arg : args_) {
                const auto &buf = buffer_allocations.GetDeviceAddress(arg->index());
                params.push_back((char *) buf.opaque());
                VLOG(3) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << " ("
                        << buf.size() << "B)";
            }

            llvm::JITSymbol sym = jit_->FindCompiledSymbol(kernel_name_);
            // We expect to find the symbol provided with kernel_name_; otherwise
            // this is an internal error.
            CHECK(sym) << "Symbol " << kernel_name_ << " not found.";
            // getAddress can do work under the hood in the jit, so it needs to be
            // guarded by the mutex.

            auto sym_addr = llvm::cantFail(sym.getAddress());

            AsNpuStream(stream)->EnqueueTask(
                    [params, sym_addr]() {

                        switch (params.size()) {
                            BOOST_PP_REPEAT(MAX_FUNC_PARAMS_SIZE, SWITCH_CASE, params)
                        }
                    });

            return tensorflow::Status::OK();
        }

    }  // namespace npu
} // namespace xla
