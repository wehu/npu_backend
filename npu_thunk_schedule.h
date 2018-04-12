//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_THUNK_SCHEDULE_H
#define TENSORFLOW_NPU_THUNK_SCHEDULE_H

#include "npu_stream_assignment.h"
#include "npu_thunk.h"

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"

namespace npu {

    using namespace xla;

    class NpuThunkSchedule {
    public:
        NpuThunkSchedule(std::unique_ptr<NpuThunkSequence> thunks,
                      std::unique_ptr<NpuStreamAssignment> stream_assignment,
                      const std::vector<const HloInstruction *> &hlo_total_order);

        // Returns the total order of executing all the thunks.
        const std::vector<NpuThunk *> &TotalOrder() const { return thunk_total_order_; }

        // Thunks that `thunk` depends on.
        const std::list<const NpuThunk *> &DependsOn(const NpuThunk *thunk) const;

        // Whether `thunk` is depended by another thunk.
        bool Depended(const NpuThunk *thunk) const { return depended_by_.count(thunk); }

        // Delegates to StreamAssignment.
        int StreamCount() const { return stream_assignment_->StreamCount(); }

        int StreamNumberForHlo(const HloInstruction &hlo) const {
            return stream_assignment_->StreamNumberForHlo(hlo);
        }

        string ToString() const;

    private:
        void RemoveRedundantDependencyEdges();

        void AddDependenciesOnTransitiveOperands(
                const NpuThunk &thunk, const HloInstruction &operand,
                const std::unordered_map<const HloInstruction *, NpuThunk *> &hlo_to_thunk);

        std::unique_ptr<NpuThunkSequence> thunks_;
        std::vector<NpuThunk *> thunk_total_order_;

        std::unordered_map<const NpuThunk *, std::list<const NpuThunk *>> depends_on_;
        std::set<const NpuThunk *> depended_by_;
        std::list<const NpuThunk *> empty_thunk_list_;

        std::unique_ptr<NpuStreamAssignment> stream_assignment_;
    };

}  // namespace npu

#endif //TENSORFLOW_NPU_THUNK_SCHEDULE_H
