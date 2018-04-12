#ifndef TENSORFLOW_NPU_PLATFORM_H_
#define TENSORFLOW_NPU_PLATFORM_H_

#include "npu_platform_id.h"

#include <memory>
#include <vector>

#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace se = perftools::gputools;

namespace xla {
    namespace npu {

        class NpuPlatform : public se::Platform {
        public:
            NpuPlatform();

            ~NpuPlatform() override;

            se::Platform::Id id() const override;

            int VisibleDeviceCount() const override;

            const se::string &Name() const override;

            se::port::StatusOr<se::StreamExecutor *> ExecutorForDevice(int ordinal) override;

            se::port::StatusOr<se::StreamExecutor *> ExecutorForDeviceWithPluginConfig(
                    int ordinal, const se::PluginConfig &config) override;

            se::port::StatusOr<se::StreamExecutor *> GetExecutor(
                    const se::StreamExecutorConfig &config) override;

            se::port::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
                    const se::StreamExecutorConfig &config) override;

            void RegisterTraceListener(std::unique_ptr<se::TraceListener> listener) override;

            void UnregisterTraceListener(se::TraceListener *listener) override;

        private:

            se::string name_;

            se::ExecutorCache executor_cache_;

            SE_DISALLOW_COPY_AND_ASSIGN (NpuPlatform);
        };

    }  // namespace npu
} // namespace xla

#endif  // TENSORFLOW_NPU_PLATFORM_H_
