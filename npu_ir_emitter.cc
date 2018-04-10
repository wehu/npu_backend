#include "npu_ir_emitter.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace npu {

    IrEmitter::IrEmitter(const HloModuleConfig &hlo_module_config,
                         IrEmitterContext *ir_emitter_context)
            : ir_emitter_context_(ir_emitter_context),
              module_(ir_emitter_context->llvm_module()),
              ir_builder_(module_->getContext()),
              hlo_module_config_(hlo_module_config) {
        ir_builder_.setFastMathFlags(llvm_ir::GetFastMathFlags(
                /*fast_math_enabled=*/hlo_module_config.debug_options()
                                              .xla_enable_fast_math()));
    }

    Status IrEmitter::DefaultAction(HloInstruction *hlo) {
        return Status::OK();
    }

    Status IrEmitter::HandleConstant(HloInstruction *constant) {
        return Unimplemented("Constant is not implement on NPU");
    }

    Status IrEmitter::HandleBitcast(HloInstruction *bitcast) {
        return Unimplemented("Bitcast is not implement on NPU");
    }

    Status IrEmitter::HandleGetTupleElement(HloInstruction *get_tuple_element) {
        return Unimplemented("GetTupleElement is not implement on NPU");
    }

    Status IrEmitter::HandleSort(HloInstruction *) {
        return Unimplemented("Sort is not implemented on NPU");
    }

    Status IrEmitter::HandleSend(HloInstruction *) {
        return Unimplemented("Send is not implemented on NPU");
    }

    Status IrEmitter::HandleSendDone(HloInstruction *) {
        return Unimplemented("Send-Done is not implemented on NPU");
    }

    Status IrEmitter::HandleRecv(HloInstruction *) {
        return Unimplemented("Recv is not implemented on NPU");
    }

    Status IrEmitter::HandleRecvDone(HloInstruction *) {
        return Unimplemented("Recv-done is not implemented on NPU");
    }

    Status IrEmitter::HandleTuple(HloInstruction *tuple) {
        return Unimplemented("Tuple is not implemented on NPU");
    }

    Status IrEmitter::HandleSelect(HloInstruction *select) {
        return Unimplemented("select is not implemented on NPU");
    }

    namespace {
        llvm::Value *Real(llvm::Value *x, llvm::IRBuilder<> *ir_builder) {
            return ir_builder->CreateExtractValue(x, {0});
        }

        llvm::Value *Imag(llvm::Value *x, llvm::IRBuilder<> *ir_builder) {
            return ir_builder->CreateExtractValue(x, {1});
        }

        std::pair<llvm::Value *, llvm::Value *> MultiplyComplex(
                llvm::Value *lhs_value, llvm::Value *rhs_value,
                llvm::IRBuilder<> *ir_builder) {
          llvm::Value *lhs_real = Real(lhs_value, ir_builder);
          llvm::Value *lhs_imag = Imag(lhs_value, ir_builder);
          llvm::Value *rhs_real = Real(rhs_value, ir_builder);
          llvm::Value *rhs_imag = Imag(rhs_value, ir_builder);
          llvm::Value *real_result1 = ir_builder->CreateFMul(lhs_real, rhs_real);
          llvm::Value *real_result2 = ir_builder->CreateFMul(lhs_imag, rhs_imag);
          llvm::Value *real_result = ir_builder->CreateFSub(real_result1, real_result2);
          llvm::Value *imag_result1 = ir_builder->CreateFMul(lhs_real, rhs_imag);
          llvm::Value *imag_result2 = ir_builder->CreateFMul(lhs_imag, rhs_real);
          llvm::Value *imag_result = ir_builder->CreateFAdd(imag_result1, imag_result2);
          return {real_result, imag_result};
        }
    }  // namespace

    Status IrEmitter::HandleDot(HloInstruction *dot) {
        return Unimplemented("CrossReplicaSum is not implemented on NPU.");
    }

    Status IrEmitter::HandleConvolution(HloInstruction *convolution) {
        if (ShapeUtil::HasZeroElements(convolution->shape())) {
            // Emit no code for an empty output.
            return Status::OK();
        }
        return Unimplemented(
                "Hit a case for convolution that is not implemented on NPU.");
    }

    Status IrEmitter::HandleFft(HloInstruction *fft) {
        if (ShapeUtil::HasZeroElements(fft->shape())) {
            // Emit no code for an empty output.
            return Status::OK();
        }
        return Unimplemented("Hit a case for fft that is not implemented on NPU.");
    }

    Status IrEmitter::HandleCrossReplicaSum(HloInstruction *crs) {
        return Unimplemented("CrossReplicaSum is not implemented on NPU.");
    }

    Status IrEmitter::HandleParameter(HloInstruction *parameter) {
        return Status::OK();
    }

    Status IrEmitter::HandleReduce(HloInstruction *reduce) {
        return Unimplemented("Reduce is not implemented on NPU.");
    }

    Status IrEmitter::HandleFusion(HloInstruction *fusion) {
        return Unimplemented("Fusion is not implemented on NPU.");
    }

    Status IrEmitter::HandleCall(HloInstruction *call) {
        return Unimplemented("Call is not implemented on NPU.");
    }

    Status IrEmitter::HandleCustomCall(HloInstruction *) {
        return Unimplemented("custom-call is not implemented on NPU.");
    }

    Status IrEmitter::HandleInfeed(HloInstruction *) {
        return Unimplemented("Infeed is not supported on NPU.");
    }

    Status IrEmitter::HandleOutfeed(HloInstruction *) {
        return Unimplemented("Outfeed is not supported on NPU.");
    }

    Status IrEmitter::HandleRng(HloInstruction *random) {
        return Unimplemented("Rng is not implemented on NPU.");
    }

    Status IrEmitter::HandleWhile(HloInstruction *xla_while) {
        return Unimplemented("While is not implement on NPU");
    }

    Status IrEmitter::HandleGather(HloInstruction *gather) {
        return Unimplemented("Cather is not implement on NPU");
    }

    Status IrEmitter::HandleCopy(HloInstruction *copy) {
        return Unimplemented("Copy is not implement on NPU");
    }

    Status IrEmitter::HandleBatchNormInference(HloInstruction *) {
        return Unimplemented(
                "The NPU backend does not implement BatchNormInference directly.  It "
                        "should be lowered before IR emission to HLO-soup using "
                        "BatchNormRewriter or to a cudnn CustomCall using "
                        "CudnnBatchNormRewriter.");
    }

    Status IrEmitter::HandleBatchNormTraining(HloInstruction *) {
        return Unimplemented(
                "The NPU backend does not implement BatchNormTraining directly.  It "
                        "should be lowered before IR emission to HLO-soup using "
                        "BatchNormRewriter or to a cudnn CustomCall using "
                        "CudnnBatchNormRewriter.");
    }

    Status IrEmitter::HandleBatchNormGrad(HloInstruction *) {
        return Unimplemented(
                "The NPU backend does not implement BatchNormGrad directly.  It should "
                        "be lowered before IR emission to HLO-soup (using BatchNormRewriter) or "
                        "to a cudnn CustomCall using CudnnBatchNormRewriter.");
    }


}  // namespace npu
