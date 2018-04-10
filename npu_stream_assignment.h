//
// Created by wehu on 18-4-10.
//

#ifndef TENSORFLOW_NPU_STREAM_ASSIGNMENT_H
#define TENSORFLOW_NPU_STREAM_ASSIGNMENT_H

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace npu {

    using namespace xla;

    class NpuStreamAssignment {
    public:
        int StreamCount() const { return stream_count_; }

        int StreamNumberForHlo(const HloInstruction &hlo) const;

        bool HasStreamAssigned(const HloInstruction &hlo) const;

        // `hlo` needs to outlive this StreamAssignment object.
        void AssignStreamToHlo(const HloInstruction *hlo, int stream_no);

    private:
        int stream_count_ = 1;  // At least the main stream.
        tensorflow::gtl::FlatMap<const HloInstruction *, int> hlo_to_stream_number_;
    };

    std::unique_ptr<NpuStreamAssignment> AssignStreams(const HloModule &module);

}  // namespace npu

#endif //TENSORFLOW_NPU_STREAM_ASSIGNMENT_H
