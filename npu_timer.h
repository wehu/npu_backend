#ifndef TENSORFLOW_NPU_TIMER_H_
#define TENSORFLOW_NPU_TIMER_H_

#include "npu_executor.h"
#include "npu_stream.h"

#include <chrono>

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = perftools::gputools;

namespace xla {
    namespace npu {

        class NpuTimer : public se::internal::TimerInterface {
        public:
            explicit NpuTimer(NpuExecutor *parent)
                    : parent_(parent) {}

            ~NpuTimer() override {}

            bool Init();

            void Destroy();

            bool Start(se::Stream *stream);

            bool Stop(se::Stream *stream);

            se::uint64 Microseconds() const override;

            se::uint64 Nanoseconds() const override;

        private:
            NpuExecutor *parent_;

            using clock = std::chrono::high_resolution_clock;

            clock::time_point start_time_;
            clock::duration duration_;

            // Actually starts (rather than enqueues starting) the timer.
            void StartNow();

            // Actually stops (rather than enqueues stopping) the timer.
            void StopNow();

        };

    }  // namespace npu
} // namespace xla

#endif  // TENSORFLOW_NPU_TIMER_H_
