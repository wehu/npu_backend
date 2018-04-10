#ifndef TENSORFLOW_NPU_TIMER_H_
#define TENSORFLOW_NPU_TIMER_H_

#include <chrono>

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "npu_executor.h"
#include "npu_stream.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuTimer : public internal::TimerInterface {
    public:
        explicit NpuTimer(NpuExecutor *parent)
                : parent_(parent) {}

        ~NpuTimer() override {}

        bool Init();

        void Destroy();

        bool Start(Stream *stream);

        bool Stop(Stream *stream);

        uint64 Microseconds() const override;
        uint64 Nanoseconds() const override;

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

#endif  // TENSORFLOW_NPU_TIMER_H_
