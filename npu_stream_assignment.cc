//
// Created by wehu on 18-4-10.
//

#include "npu_stream_assignment.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
    namespace npu {

        bool NpuStreamAssignment::HasStreamAssigned(const HloInstruction &hlo) const {
            return hlo_to_stream_number_.count(&hlo);
        }

        int NpuStreamAssignment::StreamNumberForHlo(const HloInstruction &hlo) const {
            return FindOrDie(hlo_to_stream_number_, &hlo);
        }

        void NpuStreamAssignment::AssignStreamToHlo(const HloInstruction *hlo,
                                                    int stream_no) {
            CHECK_GE(stream_no, 0);
            if (stream_no >= stream_count_) {
                stream_count_ = stream_no + 1;
            }
            InsertOrDie(&hlo_to_stream_number_, hlo, stream_no);
            VLOG(2) << "Assign stream #" << stream_no << " to " << hlo->ToString();
        }

        namespace {

            bool CanRunConcurrently(const HloInstruction &a, const HloInstruction &b,
                                    const HloReachabilityMap &reachability) {
                return !reachability.IsConnected(&a, &b);
            }

            int ComputeStreamToAssign(
                    const HloInstruction &hlo, const NpuStreamAssignment &stream_assignment,
                    const HloReachabilityMap &reachability,
                    const std::vector<const HloInstruction *> &seen_gemms) {
                if (hlo.opcode() == HloOpcode::kParameter ||
                    hlo.opcode() == HloOpcode::kConstant) {
                    // kParameter and kConstant do not need a thunk.
                    return -1;
                }

                if (hlo.GetModule()
                        ->config()
                        .debug_options()
                        .xla_gpu_disable_multi_streaming()) {
                    return 0;
                }

                std::set<int> forbidden_stream_numbers;
                for (const auto *seen_gemm : seen_gemms) {
                    int stream_no = stream_assignment.StreamNumberForHlo(*seen_gemm);
                    if (!forbidden_stream_numbers.count(stream_no) &&
                        CanRunConcurrently(*seen_gemm, hlo, reachability)) {
                        forbidden_stream_numbers.insert(stream_no);
                    }
                }

                for (int stream_no = 0; stream_no < stream_assignment.StreamCount();
                     ++stream_no) {
                    if (!forbidden_stream_numbers.count(stream_no)) {
                        return stream_no;
                    }
                }
                return stream_assignment.StreamCount();
            }

        }  // namespace

        std::unique_ptr<NpuStreamAssignment> AssignStreams(const HloModule &module) {
            auto stream_assignment = MakeUnique<NpuStreamAssignment>();
            const HloComputation &computation = *module.entry_computation();
            std::unique_ptr<HloReachabilityMap> reachability =
                    computation.ComputeReachability();
            std::vector<const HloInstruction *> seen_gemms;
            for (const auto *hlo : computation.MakeInstructionPostOrder()) {
                int stream_no = ComputeStreamToAssign(*hlo, *stream_assignment,
                                                      *reachability, seen_gemms);
                if (stream_no != -1) {
                    stream_assignment->AssignStreamToHlo(hlo, stream_no);
                }
            }
            return stream_assignment;
        }

    }  // namespace npu
} // namespace xla