#include "npu_timer.h"

#include "npu_executor.h"
#include "npu_stream.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace npu {

    bool NpuTimer::Init() {
        return true;
    }

    void NpuTimer::Destroy() {
    }

    uint64 NpuTimer::GetElapsedMilliseconds() const {
        return 0;
    }

    bool NpuTimer::Start(NpuStream *stream) {
        return true;
    }

    bool NpuTimer::Stop(NpuStream *stream) {
        return true;
    }

}  // namespace npu
