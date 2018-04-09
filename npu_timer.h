#ifndef TENSORFLOW_NPU_TIMER_H_
#define TENSORFLOW_NPU_TIMER_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuExecutor;
    class NpuStream;

    class NpuTimer : public internal::TimerInterface {
    public:
        explicit NpuTimer(NpuExecutor *parent)
                : parent_(parent) {}

        ~NpuTimer() override {}

        bool Init();

        void Destroy();

        bool Start(NpuStream *stream);

        bool Stop(NpuStream *stream);

        uint64 GetElapsedMilliseconds() const;

        uint64 Microseconds() const override {
            return GetElapsedMilliseconds() * 1000;
        }

        uint64 Nanoseconds() const override { return GetElapsedMilliseconds() * 1000; }

    private:
        NpuExecutor *parent_;
    };

}  // namespace npu

#endif  // TENSORFLOW_NPU_TIMER_H_
