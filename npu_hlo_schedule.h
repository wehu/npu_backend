//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_HLO_SCHEDULE_H
#define TENSORFLOW_NPU_HLO_SCHEDULE_H

#include "npu_stream_assignment.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
    namespace npu {

        class NpuHloSchedule {
        public:
            // Constructs an HloSchedule for the given module, based on the given stream
            // assignment.
            static StatusOr<std::unique_ptr<NpuHloSchedule>> Build(
                    const HloModule &module, const NpuStreamAssignment &stream_assignment,
                    int64 pointer_size);

            // Returns the total order of thunk launches, represented in terms of HLO
            // instructions.
            const std::vector<const HloInstruction *> &ThunkLaunchOrder() const {
                return thunk_launch_order_;
            }

            // Returns the partial order of HLO instructions. This method may only be
            // called once. The order is based on the total order of thunk lanches, the
            // stream assignment, and the data dependencies in the HLO DAG.
            std::unique_ptr<HloOrdering> ConsumeHloOrdering() {
                return std::move(hlo_ordering_);
            }

        private:
            NpuHloSchedule();

            std::vector<const HloInstruction *> thunk_launch_order_;
            std::unique_ptr<HloOrdering> hlo_ordering_;
        };


    }  // namespace npu
} // namespace xla

#endif //TENSORFLOW_NPU_HLO_SCHEDULE_H
