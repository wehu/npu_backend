#include "npu_event.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace npu {

    NpuEvent::NpuEvent(NpuExecutor* parent)
            : parent_(parent) {}

    NpuEvent::~NpuEvent() {}

    port::Status NpuEvent::Init() {
        return port::Status::OK();
    }

    port::Status NpuEvent::Destroy() {
        return port::Status::OK();
    }

    port::Status NpuEvent::Record(NpuStream* stream) {
        return port::Status::OK();
    }

    Event::Status NpuEvent::PollForStatus() {
        return Event::Status::kComplete;
    }

}  // namespace npu
