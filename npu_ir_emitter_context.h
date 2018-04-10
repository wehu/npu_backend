#ifndef TENSORFLOW_NPU_IR_EMITTER_CONTEXT_H_
#define TENSORFLOW_NPU_IR_EMITTER_CONTEXT_H_

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"


namespace npu {

    using namespace xla;

    class IrEmitterContext {
    public:
        IrEmitterContext(const HloModule* hlo_module,
                         const BufferAssignment* buffer_assignment,
                         const perftools::gputools::DeviceDescription* device_desc,
                         llvm::Module* llvm_module)
                : hlo_module_(hlo_module),
                  buffer_assignment_(buffer_assignment),
                  device_desc_(device_desc),
                  llvm_module_(llvm_module) {}
        // Disallow copy and assign.
        IrEmitterContext(const IrEmitterContext&) = delete;
        IrEmitterContext& operator=(const IrEmitterContext&) = delete;

        // Simple accessors.
        const HloModule& hlo_module() const { return *hlo_module_; }
        const BufferAssignment& buffer_assignment() const { return *buffer_assignment_; }

        const perftools::gputools::DeviceDescription& device_description() const {
          return *device_desc_;
        }

        llvm::Module* llvm_module() { return llvm_module_; }

        NameUniquer* name_uniquer() { return &name_uniquer_; }

    private:
        const HloModule* hlo_module_;
        const BufferAssignment* buffer_assignment_;
        const perftools::gputools::DeviceDescription* device_desc_;
        llvm::Module* llvm_module_;
        NameUniquer name_uniquer_;
    };

}  // namespace npu

#endif  // TENSORFLOW_NPU_IR_EMITTER_CONTEXT_H_
