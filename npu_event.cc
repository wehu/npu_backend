#include "npu_event.h"

#include "tensorflow/stream_executor/lib/statusor.h"

namespace npu {

    NpuEvent::NpuEvent(NpuExecutor* parent)
            : parent_(parent) {}

    NpuEvent::~NpuEvent() {}

    port::Status NpuEvent::Init() {
        streams_.clear();
        return port::Status::OK();
    }

    port::Status NpuEvent::Destroy() {
        streams_.clear();
        return port::Status::OK();
    }

    port::Status NpuEvent::Record(Stream* stream) {
        streams_.insert(stream);
        return port::Status::OK();
    }

    Event::Status NpuEvent::PollForStatus() {
        for(Stream* stream : streams_) {
            if(!AsNpuStream(stream)->IsIdle()) {
                return Event::Status::kPending;
            }
        }
        return Event::Status::kComplete;
    }

    NpuEvent *AsNpuEvent(Event *event) {
        DCHECK(event != nullptr);
        return static_cast<NpuEvent *>(event->implementation());
    }


}  // namespace npu
