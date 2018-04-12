#include "npu_ir_emitter.h"
#include "npu_tuple_thunk.h"
#include "npu_constants.h"

#include <string>
#include <unordered_map>
#include <utility>

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
#include "tensorflow/core/platform/logging.h"

namespace npu {


    using llvm_ir::IrName;
    using tensorflow::gtl::ArraySlice;
    using tensorflow::gtl::nullopt;
    using tensorflow::gtl::optional;
    using tensorflow::strings::StrCat;

    IrEmitter::IrEmitter(const HloModuleConfig &hlo_module_config,
                         const HloComputation* hlo_computation,
                         IrEmitterContext *ir_emitter_context)
            : ir_emitter_context_(ir_emitter_context),
              module_(ir_emitter_context->llvm_module()),
              ir_builder_(module_->getContext()),
              bindings_(ir_emitter_context->hlo_module(),
                        &ir_emitter_context->buffer_assignment(), &ir_builder_, module_,
                        false),
              hlo_module_config_(hlo_module_config),
              hlo_computation_(hlo_computation) {
        ir_builder_.setFastMathFlags(llvm_ir::GetFastMathFlags(
                /*fast_math_enabled=*/hlo_module_config.debug_options()
                                              .xla_enable_fast_math()));

        thunk_sequence_.reset(new NpuThunkSequence());
    }

    Status IrEmitter::DefaultAction(HloInstruction *hlo) {
        thunk_sequence_->emplace_back(BuildKernelThunk(hlo));
        ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
        for (const HloInstruction* operand : hlo->operands()) {
            operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
                return GetIrArray(*operand, *hlo)
                        .EmitReadArrayElement(index, &ir_builder_);
            };
        }
        return EmitTargetElementLoop(
                *hlo, ElementalIrEmitter(hlo_module_config_, module_, &ir_builder_)
                        .MakeElementGenerator(hlo, operand_to_generator));
    }

    Status IrEmitter::HandleConstant(HloInstruction *constant) {
        const Literal& literal = constant->literal();
        llvm::Constant* initializer =
                llvm_ir::ConvertLiteralToIrConstant(literal, module_);
        llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
                *module_, initializer->getType(),
                /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, initializer,
                /*Name=*/"");
        VLOG(2) << "HandleConstant: " << constant->ToString() << std::endl
                << "  emitted_value: " << llvm_ir::DumpToString(*global_for_const)
                << std::endl
                << "  its type: "
                << llvm_ir::DumpToString(*global_for_const->getType());
        bindings_.BindHloToIrValue(*constant, global_for_const);
        return Status::OK();
    }

    Status IrEmitter::HandleBitcast(HloInstruction *bitcast) {
        VLOG(2) << "HandleBitcast: " << bitcast->ToString();
        const HloInstruction* operand = bitcast->operand(0);
        if (bindings_.BoundToIrValue(*operand)) {
            bindings_.BindHloToIrValue(*bitcast, GetBasePointer(*operand));
        }
        return Status::OK();
    }

    Status IrEmitter::HandleGetTupleElement(HloInstruction *get_tuple_element) {
        return Status::OK();
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
        bool all_tuple_elements_have_buffer =
                c_all_of(tuple->operands(), [&](HloInstruction* tuple_element) {
                    return ir_emitter_context_->buffer_assignment().HasTopLevelAllocation(
                            tuple_element);
                });
        if (all_tuple_elements_have_buffer) {
            std::vector<BufferAllocation::Slice> tuple_element_buffers;
            for (const HloInstruction* tuple_element : tuple->operands()) {
                tuple_element_buffers.push_back(GetAllocationSlice(*tuple_element));
            }
            thunk_sequence_->emplace_back(MakeUnique<NpuTupleThunk>(
                    tuple_element_buffers, GetAllocationSlice(*tuple), tuple));

            return Status::OK();
        }
        //thunk_sequence_->emplace_back(BuildKernelThunk(tuple));
        std::vector<llvm::Value*> base_ptrs;
        for (const HloInstruction* operand : tuple->operands()) {
            base_ptrs.push_back(GetBasePointer(*operand));
        }
        llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &ir_builder_,
                           module_);

        return Status::OK();
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
        thunk_sequence_->push_back(BuildKernelThunk(random));
        ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
        for (const HloInstruction* operand : random->operands()) {
            operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
                return GetIrArray(*operand, *random)
                        .EmitReadArrayElement(index, &ir_builder_);
            };
        }

        return llvm_ir::LoopEmitter(
                ElementalIrEmitter(hlo_module_config_, module_, &ir_builder_)
                        .MakeElementGenerator(random, operand_to_generator),
                GetIrArray(*random, *random), &ir_builder_)
                .EmitLoop(IrName(random));
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

    Status IrEmitter::HandleConditional(HloInstruction *copy) {
        return Unimplemented("Conditional is not implement on NPU");
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

    Status IrEmitter::EmitTargetElementLoopInThunk(
            const HloInstruction& hlo,
            const llvm_ir::ElementGenerator& element_generator, NpuKernelThunk* thunk) {
        VLOG(3) << bindings_.ToString();
        const Shape& element_shape = hlo.IsMultiOutputFusion()
                                     ? ShapeUtil::GetSubshape(hlo.shape(), {0})
                                     : hlo.shape();
        if (!hlo.IsMultiOutputFusion()) {
            return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo),
                                        &ir_builder_)
                    .EmitLoop(IrName(&hlo));
        }

        // For multiple outputs fusion, we need to emit each operand and the root.
        std::vector<llvm_ir::IrArray> output_arrays;
        for (int64 i = 0; i < ShapeUtil::TupleElementCount(hlo.shape()); ++i) {
            output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
        }
        TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, output_arrays,
                                                &ir_builder_)
                                   .EmitLoop(IrName(&hlo)));

        std::vector<llvm::Value*> tuple_operand_ptrs;
        for (int64 i = 0; i < output_arrays.size(); ++i) {
            tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
        }
        ir_builder_.SetInsertPoint(ir_builder_.GetInsertBlock()->getTerminator());
        llvm_ir::EmitTuple(GetIrArray(hlo, hlo), tuple_operand_ptrs, &ir_builder_,
                           module_);
        return Status::OK();
    }

    Status IrEmitter::EmitTargetElementLoop(
            const HloInstruction& hlo,
            const llvm_ir::ElementGenerator& element_generator) {
        CHECK(NpuThunk::Kind::kKernel == LastThunk()->kind());
        return EmitTargetElementLoopInThunk(hlo, element_generator,
                                            static_cast<NpuKernelThunk*>(LastThunk()));
    }

    llvm::Function* IrEmitter::BuildKernelPrototype(
            const HloInstruction& inst,
            tensorflow::gtl::ArraySlice<const BufferAllocation*> args) {
        // Compute the kernel name. The opcode string may contain "-" which cannot be
        // in a PTX function name, so sanitize the name before uniquifying it.
        string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
                llvm_ir::SanitizeFunctionName(inst.name()));

        // Create the kernel and add it to the module.
        llvm::Module* module = ir_emitter_context_->llvm_module();
        llvm::LLVMContext& context = module->getContext();
        llvm::FunctionType* kernel_type = llvm::FunctionType::get(
                /*Result=*/llvm::Type::getVoidTy(context),
                           std::vector<llvm::Type*>(args.size(), ir_builder_.getInt8PtrTy()),
                /*isVarArg=*/false);
        llvm::Function* kernel =
                llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                                       kernel_name.c_str(), module);

        // Add dereferenceable and alignment information to each of the kernel's
        // parameters.
        auto arg_it = kernel->arg_begin();
        for (size_t arg_no = 0; arg_no < args.size(); ++arg_no) {
            const BufferAllocation* alloc = args[arg_no];
            llvm::Argument* fn_arg = &*arg_it;
            ++arg_it;

            kernel->addDereferenceableAttr(arg_no + 1, alloc->size());
            kernel->addParamAttr(
                    arg_no, llvm::Attribute::get(context, llvm::Attribute::Alignment,
                                                 npuAlignBytes));

            if (alloc->IsPreallocatedTempBuffer()) {
                fn_arg->setName("temp_buf");
            } else {
                fn_arg->setName(llvm_ir::AsStringRef(StrCat("alloc", alloc->index())));
            }
        }

         // Update the insert point to the entry basic block.
        llvm::BasicBlock* entry_bb =
                llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

        // Emit a "return void" at entry_bb's end, and set the insert point before
        // that return instruction.
        ir_builder_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

        return kernel;
    }

    optional<BufferAllocation::Slice> GetKnownAtRuntimeSlice(
            const HloInstruction* instr, const ShapeIndex& index,
            const BufferAssignment& buffer_assn) {
        auto maybe_slice = buffer_assn.GetUniqueSlice(instr, index);
        if (!maybe_slice.ok()) {
            return nullopt;
        }
        // BufferAllocation gives a slice and alloc to every buffer accessed by XLA,
        // but we don't necessarily know the runtime address of sub-buffers of input
        // parameters.
        const BufferAllocation::Slice& slice = maybe_slice.ValueOrDie();
        const BufferAllocation* alloc = slice.allocation();
        if (alloc->IsInputOrOutput() && !alloc->maybe_live_out() &&
            !alloc->param_shape_index().empty()) {
            return nullopt;
        }

        // Otherwise, we will know the address of this slice at runtime without having
        // to dereference a tuple.
        return slice;
    }

    static std::map<std::pair<const HloInstruction*, ShapeIndex>,
            std::pair<BufferAllocation::Slice, ShapeIndex>>
    GetHloBufferSlices(const HloInstruction* hlo,
                       const BufferAssignment& buffer_assn) {
        std::map<std::pair<const HloInstruction*, ShapeIndex>,
                std::pair<BufferAllocation::Slice, ShapeIndex>>
                slices;

        // Tries to find a slice plus an array of indices i1, ..., iN such that the
        // sub-buffer for instr at index can be found at slice[i1]...[iN].
        auto find_slice_for = [&](const HloInstruction* instr,
                                  const ShapeIndex& index)
                -> optional<std::pair<BufferAllocation::Slice, ShapeIndex>> {
            // Simple, common case: Is the buffer for instr known at runtime?  If so,
            // we're done.
            auto slice = GetKnownAtRuntimeSlice(instr, index, buffer_assn);
            if (slice.has_value()) {
                return {{*slice, ShapeIndex()}};
            }

            // If we don't know the buffer for instr at index, see if we know the buffer
            // for instr at index without its last element.  If so, we can dynamically
            // find the buffer for instr by dereferencing a pointer in that buffer.
            // Continue looking this way until we run out of elements in 'index'.
            ShapeIndex new_index = index;
            ShapeIndex gte_indices;
            while (!new_index.empty()) {
                gte_indices.push_front(new_index.back());
                new_index.pop_back();
                auto slice = GetKnownAtRuntimeSlice(instr, new_index, buffer_assn);
                if (slice.has_value()) {
                    return {{*slice, gte_indices}};
                }
            }

            // If *that* didn't work, check whether instr is a GTE instruction.  If it
            // is, see if we can get a buffer for its parent, and continue walking up
            // parents until we find a defined buffer or we hit something that's not a
            // GTE.
            const HloInstruction* parent = instr;
            while (parent->opcode() == HloOpcode::kGetTupleElement) {
                gte_indices.push_front(parent->tuple_index());
                parent = parent->operand(0);

                auto slice = GetKnownAtRuntimeSlice(parent, {}, buffer_assn);
                if (slice.has_value()) {
                    return {{*slice, gte_indices}};
                }
            }

            return nullopt;
        };

        // Adds entries for all subshapes of instr to `slices`.
        auto add_slices_for = [&](const HloInstruction* instr) {
            // GPU constants don't have buffers; don't bother looking for one.
            if (instr->IsConstant()) {
                return;
            }

            ShapeUtil::ForEachSubshape(
                    instr->shape(), [&](const Shape& /*shape*/, const ShapeIndex& index) {
                        if (slices.count({instr, index})) {
                            // HLOs can have duplicate operands; don't bother redoing work.
                            return;
                        }
                        auto maybe_slice = find_slice_for(instr, index);
                        if (maybe_slice.has_value()) {
                            slices[{instr, index}] = *maybe_slice;
                        } else {
                            VLOG(1) << "Couldn't find buffer for " << instr->ToString()
                                    << " at index " << index.ToString();
                        }
                    });
        };

        add_slices_for(hlo);
        for (const HloInstruction* operand : hlo->operands()) {
            // Conservatively assume we'll need the buffers for all subshapes of the
            // operand.
            add_slices_for(operand);
        }

        return slices;
    }

    std::unique_ptr<NpuKernelThunk> IrEmitter::BuildKernelThunk(const HloInstruction* inst) {
        const BufferAssignment& buffer_assn =
                ir_emitter_context_->buffer_assignment();

        std::map<std::pair<const HloInstruction*, ShapeIndex>,
                std::pair<BufferAllocation::Slice, ShapeIndex>>
                hlo_slices = GetHloBufferSlices(inst, buffer_assn);

        // Figure out which buffer allocations need to be passed as arguments to our
        // kernel.  This is simply all of the allocations referenced in hlo_slices,
        // plus the XLA temp buffer (if we have it).  We always include the temp
        // buffer because even if the kernel itself doesn't use it, a nested
        // subcomputation within the kernel (e.g. a kMap's computation) might.
        std::unordered_set<const BufferAllocation*> buffers_needed;
        for (const auto& kv : hlo_slices) {
            buffers_needed.insert(kv.second.first.allocation());
        }
        tensorflow::gtl::optional<const BufferAllocation*> temp_buffer;
        for (const BufferAllocation& alloc : buffer_assn.Allocations()) {
            if (alloc.IsPreallocatedTempBuffer()) {
                if (!temp_buffer.has_value()) {
                    temp_buffer = &alloc;
                } else {
                    LOG(FATAL) << "Multiple temp buffers found, but only one is allowed!";
                }
            }
        }
        if (temp_buffer.has_value()) {
            buffers_needed.insert(*temp_buffer);
        }

        // We'll pass a pointer to each of the elements of `buffers` to our kernel, in
        // this order.
        std::vector<const BufferAllocation*> buffers(buffers_needed.begin(),
                                                     buffers_needed.end());
        std::sort(buffers.begin(), buffers.end(),
                  [](const BufferAllocation* a, const BufferAllocation* b) {
                      return a->index() < b->index();
                  });

        llvm::Function* kernel = BuildKernelPrototype(*inst, buffers);

        // Build a map from a BufferAllocation to the corresponding argument in our
        // kernel.
        std::unordered_map<const BufferAllocation*, llvm::Value*> kernel_args;
        {
            auto arg_it = kernel->arg_begin();
            auto buffers_it = buffers.begin();
            for (; arg_it != kernel->arg_end(); ++arg_it, ++buffers_it) {
                kernel_args[*buffers_it] = arg_it;
            }
        }

        // For each buffer our kernel might want to touch, bind it to a value derived
        // from our kernel args.
        for (const auto& kv : hlo_slices) {
            const HloInstruction* instr = kv.first.first;
            const ShapeIndex& index = kv.first.second;
            const BufferAllocation::Slice& slice = kv.second.first;
            const ShapeIndex& gte_index = kv.second.second;

            VLOG(3) << "Buffer for " << instr->ToString() << " at " << index.ToString()
                    << " is found in slice " << slice.ToString() << " at GTE index "
                    << gte_index.ToString();

            llvm::Value* loc =
                    ir_builder_.CreateInBoundsGEP(kernel_args.at(slice.allocation()),
                                                  {ir_builder_.getInt64(slice.offset())});

            // If gte_index is nonempty, we have to dereference `loc` to get to the
            // value we're ultimately interested in.
            llvm::Type* int8_double_pointer =
                    llvm::PointerType::get(ir_builder_.getInt8PtrTy(), /*AddressSpace=*/0);
            for (int64 idx : gte_index) {
                loc = ir_builder_.CreateBitCast(loc, int8_double_pointer);
                loc = ir_builder_.CreateLoad(
                        ir_builder_.CreateInBoundsGEP(loc, {ir_builder_.getInt64(idx)}));
            }

            bindings_.BindHloToIrValue(*instr, loc, index);
        }

        // Bind the temp buffer so that nested subcomputations can find it if they
        // need.
        if (temp_buffer.has_value()) {
            bindings_.SetTempBufferBase(kernel_args.at(*temp_buffer));
        } else {
            bindings_.SetTempBufferBase(
                    llvm::ConstantPointerNull::get(ir_builder_.getInt8PtrTy()));
        }

        return MakeUnique<NpuKernelThunk>(buffers, llvm_ir::AsString(kernel->getName()),
                                       inst, ir_emitter_context_->jit());
    }

}  // namespace npu
