#ifndef TENSORFLOW_NPU_EVENT_H_
#define TENSORFLOW_HPU_EVENT_H_

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"

#include "npu_stream.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuEvent : public internal::EventInterface {
    public:
        explicit NpuEvent(NpuExecutor* parent);

        ~NpuEvent() override;

        port::Status Init();

        port::Status Destroy();

        port::Status Record(Stream* stream);

        Event::Status PollForStatus();

    private:
        NpuExecutor* parent_;

        tensorflow::gtl::FlatSet<Stream*> streams_;

    };

    NpuEvent *AsNpuEvent(Event *event);

}  // namespace npu

#endif  // TENSORFLOW_NPU_EVENT_H_
