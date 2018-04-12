#ifndef TENSORFLOW_NPU_EVENT_H_
#define TENSORFLOW_HPU_EVENT_H_

#include "npu_stream.h"

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace se = perftools::gputools;

namespace xla {
    namespace npu {

        class NpuEvent : public se::internal::EventInterface {
        public:
            explicit NpuEvent(NpuExecutor *parent);

            ~NpuEvent() override;

            se::port::Status Init();

            se::port::Status Destroy();

            se::port::Status Record(se::Stream *stream);

            se::Event::Status PollForStatus();

        private:
            NpuExecutor *parent_;

            tensorflow::gtl::FlatSet<se::Stream *> streams_;

        };

        NpuEvent *AsNpuEvent(se::Event *event);

    }  // namespace npu
} // namespace npu

#endif  // TENSORFLOW_NPU_EVENT_H_
