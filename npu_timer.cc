#include "npu_timer.h"
#include "npu_executor.h"
#include "npu_stream.h"

#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {
    namespace npu {

        using namespace perftools::gputools;
        using std::chrono::duration_cast;

        bool NpuTimer::Init() {
            return true;
        }

        void NpuTimer::Destroy() {
        }

        bool NpuTimer::Start(Stream *stream) {
            return stream->ThenDoHostCallback([this]() { this->StartNow(); }).ok();
        }

        bool NpuTimer::Stop(Stream *stream) {
            return stream->ThenDoHostCallback([this]() { this->StopNow(); }).ok();
        }

        uint64 NpuTimer::Microseconds() const {
            return duration_cast<std::chrono::microseconds>(duration_).count();
        }

        uint64 NpuTimer::Nanoseconds() const {
            return duration_cast<std::chrono::nanoseconds>(duration_).count();
        }

        void NpuTimer::StartNow() { start_time_ = clock::now(); }

        void NpuTimer::StopNow() { duration_ = clock::now() - start_time_; }

    }  // namespace npu
} // namespace xla
