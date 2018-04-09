#ifndef TENSORFLOW_NPU_IR_EMITTER_H_
#define TENSORFLOW_NPU_IR_EMITTER_H_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

#include "npu_ir_emitter_context.h"

namespace npu {

    using namespace xla;

    class IrEmitter : public DfsHloVisitorWithDefault {
    public:
        IrEmitter(const HloModuleConfig& hlo_module_config,
                  IrEmitterContext* ir_emitter_context);
        IrEmitter(const IrEmitter&) = delete;
        IrEmitter& operator=(const IrEmitter&) = delete;

        Status DefaultAction(HloInstruction* hlo) override;
        Status HandleConstant(HloInstruction* constant) override;
        Status HandleBitcast(HloInstruction* bitcast) override;
        Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
        Status HandleDot(HloInstruction* dot) override;
        Status HandleConvolution(HloInstruction* convolution) override;
        Status HandleFft(HloInstruction* fft) override;
        Status HandleCrossReplicaSum(HloInstruction* crs) override;
        Status HandleInfeed(HloInstruction* infeed) override;
        Status HandleOutfeed(HloInstruction* outfeed) override;
        Status HandleSort(HloInstruction* sort) override;
        Status HandleSend(HloInstruction* send) override;
        Status HandleSendDone(HloInstruction* send_done) override;
        Status HandleRecv(HloInstruction* recv) override;
        Status HandleRecvDone(HloInstruction* recv_done) override;
        Status HandleParameter(HloInstruction* parameter) override;
        Status HandleReduce(HloInstruction* reduce) override;
        Status HandleTuple(HloInstruction* tuple) override;
        Status HandleSelect(HloInstruction* select) override;
        Status HandleFusion(HloInstruction* fusion) override;
        Status HandleCall(HloInstruction* call) override;
        Status HandleCustomCall(HloInstruction* custom_call) override;
        Status HandleRng(HloInstruction* random) override;
        Status HandleBatchNormInference(HloInstruction* batch_norm) override;
        Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
        Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

        Status FinishVisit(HloInstruction* root) override { return Status::OK(); }

    protected:

        IrEmitterContext* ir_emitter_context_;
        llvm::Module* module_;

        llvm::IRBuilder<> ir_builder_;

        const HloModuleConfig& hlo_module_config_;

    private:

    };

}  // namespace npu

#endif  // TENSORFLOW_NPU_IR_EMITTER_H_
