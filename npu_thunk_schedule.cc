//
// Created by wehu on 18-4-10.
//

#include "npu_thunk_schedule.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
    namespace npu {

        void NpuThunkSchedule::AddDependenciesOnTransitiveOperands(
                const NpuThunk &thunk, const HloInstruction &operand,
                const std::unordered_map<const HloInstruction *, NpuThunk *> &hlo_to_thunk) {
            if (hlo_to_thunk.count(&operand)) {
                // If `operand` is mapped to a thunk, adds `operand` to `thunk`'s dependency
                // list if `operand` is assigned to a different stream. As an optimization,
                // we skip `operand`'s operands because `operand` depends on them already.
                if (stream_assignment_->StreamNumberForHlo(operand) !=
                    stream_assignment_->StreamNumberForHlo(*thunk.hlo_instruction())) {
                    depends_on_[&thunk].push_back(FindOrDie(hlo_to_thunk, &operand));
                }
            } else {
                // If `operand` doesn't need a thunk (e.g. bitcast), continue with its
                // operands.
                for (const auto *operand_of_operand : operand.operands()) {
                    AddDependenciesOnTransitiveOperands(thunk, *operand_of_operand,
                                                        hlo_to_thunk);
                }
            }
        }

        NpuThunkSchedule::NpuThunkSchedule(
                std::unique_ptr<NpuThunkSequence> thunks,
                std::unique_ptr<NpuStreamAssignment> stream_assignment,
                const std::vector<const HloInstruction *> &hlo_total_order)
                : thunks_(std::move(thunks)),
                  stream_assignment_(std::move(stream_assignment)) {
            std::unordered_map<const HloInstruction *, NpuThunk *> hlo_to_thunk;
            for (const auto &thunk : *thunks_) {
                InsertOrDie(&hlo_to_thunk, thunk->hlo_instruction(), thunk.get());
            }

            for (const HloInstruction *hlo : hlo_total_order) {
                if (hlo_to_thunk.count(hlo)) {
                    thunk_total_order_.push_back(FindOrDie(hlo_to_thunk, hlo));
                }
            }

            for (const NpuThunk *thunk : thunk_total_order_) {
                const auto *dst = thunk->hlo_instruction();
                CHECK(stream_assignment_->HasStreamAssigned(*dst));
                for (const auto *src : dst->operands()) {
                    AddDependenciesOnTransitiveOperands(*thunk, *src, hlo_to_thunk);
                }
            }

            RemoveRedundantDependencyEdges();

            // Compute `depended_by_`, the inverse of `depends_on_`.
            for (const auto &dependency : depends_on_) {
                for (const auto *depended : dependency.second) {
                    depended_by_.insert(depended);
                }
            }
        }

        void NpuThunkSchedule::RemoveRedundantDependencyEdges() {
            std::unordered_map<const NpuThunk *, int> thunk_to_total_order;
            for (int i = 0; i < thunk_total_order_.size(); ++i) {
                InsertOrDie(&thunk_to_total_order, thunk_total_order_[i], i);
            }

            int stream_count = stream_assignment_->StreamCount();
            // S1  S2
            //
            // T1<----+
            //        |
            // T3<--+ |
            //      | | depends on
            //     T4 |
            //        |
            //     T2-+
            //
            // Suppose thunk T1 and T3 are scheduled on stream S1, and T2 and T4 are on
            // stream S2. If T2 depends on T1 and T4 depends on T3, and
            // order(T1)<order(T3)<order(T4)<order(T2), the dependency of T2 on T1 is
            // redundant.
            //
            // To efficiently detect such redundancy, we leverage array `last_dependency`.
            // last_dependency[S1][S2] indicates the last thunk (with the maximum order
            // number) on stream S2 that thunks on S1 depends on. Therefore, if a future
            // S1 thunk depends on a S2 thunk ordered <=last_dependency[S1][S2], that is a
            // redundant dependency edge.
            Array2D<int> last_dependency(stream_count, stream_count, -1);
            for (const NpuThunk *dst : thunk_total_order_) {
                if (!depends_on_.count(dst)) {
                    continue;
                }

                int dst_stream =
                        stream_assignment_->StreamNumberForHlo(*dst->hlo_instruction());
                std::list<const NpuThunk *> &sources = FindOrDie(depends_on_, dst);
                for (auto iter = sources.begin(); iter != sources.end();) {
                    const NpuThunk *src = *iter;
                    // `dst` depends on `src`.
                    int src_stream =
                            stream_assignment_->StreamNumberForHlo(*src->hlo_instruction());
                    int src_order = FindOrDie(thunk_to_total_order, src);
                    if (src_order <= last_dependency(dst_stream, src_stream)) {
                        iter = sources.erase(iter);
                    } else {
                        last_dependency(dst_stream, src_stream) = src_order;
                        ++iter;
                    }
                }
                if (sources.empty()) {
                    depends_on_.erase(dst);
                }
            }
        }

        const std::list<const NpuThunk *> &NpuThunkSchedule::DependsOn(
                const NpuThunk *thunk) const {
            if (depends_on_.count(thunk)) {
                return FindOrDie(depends_on_, thunk);
            } else {
                return empty_thunk_list_;
            }
        }

        string NpuThunkSchedule::ToString() const {
            string result = "Total order:\n";
            for (NpuThunk *thunk : thunk_total_order_) {
                tensorflow::strings::StrAppend(&result, "\t",
                                               thunk->hlo_instruction()->ToString(), "\n");
            }
            tensorflow::strings::StrAppend(&result, "Dependencies:\n");
            for (const auto &entry : depends_on_) {
                const NpuThunk *dependent = entry.first;
                for (const NpuThunk *dependency : entry.second) {
                    tensorflow::strings::StrAppend(
                            &result, "\t", dependent->hlo_instruction()->name(), " depends on ",
                            dependency->hlo_instruction()->name(), "\n");
                }
            }
            return result;
        }

    }  // namespace npu
} // namespace xla