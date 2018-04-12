#include "npu_platform.h"
#include "npu_executor.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace npu {

    namespace {

        const DeviceOptions GetDeviceOptionsFromEnv() {
          return perftools::gputools::DeviceOptions::Default();
        }
    }  // namespace


    NpuPlatform::NpuPlatform()
            : name_("NPU") {}

    NpuPlatform::~NpuPlatform() {}

    int NpuPlatform::VisibleDeviceCount() const {
      return 1;
    }

    Platform::Id NpuPlatform::id() const { return npuPlatformId; }

    const string& NpuPlatform::Name() const { return name_; }

    port::StatusOr<StreamExecutor*> NpuPlatform::ExecutorForDevice(int ordinal) {
      StreamExecutorConfig config;
      config.ordinal = ordinal;
      config.plugin_config = PluginConfig();
      config.device_options = GetDeviceOptionsFromEnv();
      return GetExecutor(config);
    }

    port::StatusOr<StreamExecutor*> NpuPlatform::ExecutorForDeviceWithPluginConfig(
            int device_ordinal, const PluginConfig& plugin_config) {
      StreamExecutorConfig config;
      config.ordinal = device_ordinal;
      config.plugin_config = plugin_config;
      config.device_options = GetDeviceOptionsFromEnv();
      return GetExecutor(config);
    }

    port::StatusOr<StreamExecutor*> NpuPlatform::GetExecutor(
            const StreamExecutorConfig& config) {
      return executor_cache_.GetOrCreate(
              config, [&]() { return GetUncachedExecutor(config); });
    }

    port::StatusOr<std::unique_ptr<StreamExecutor>>
    NpuPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
      auto executor = port::MakeUnique<StreamExecutor>(
              this, port::MakeUnique<NpuExecutor>(config.plugin_config));
      auto init_status = executor->Init(config.ordinal, config.device_options);
      if (!init_status.ok()) {
        return port::Status{
                port::error::INTERNAL,
                port::Printf(
                        "failed initializing StreamExecutor for Npu device ordinal %d: %s",
                        config.ordinal, init_status.ToString().c_str())};
      }

      return std::move(executor);
    }

    void NpuPlatform::RegisterTraceListener(
            std::unique_ptr<TraceListener> listener) {
      LOG(FATAL) << "not yet implemented: register Npu trace listener";
    }

    void NpuPlatform::UnregisterTraceListener(TraceListener* listener) {
      LOG(FATAL) << "not yet implemented: unregister Npu trace listener";
    }

    static void InitializeNpuPlatform() {
      // Disabling leak checking, MultiPlatformManager does not destroy its
      // registered platforms.

      std::unique_ptr<npu::NpuPlatform> platform(new npu::NpuPlatform);
      SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
    }

}  // namespace npu

REGISTER_MODULE_INITIALIZER(npu_platform,
                            npu::InitializeNpuPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(npu_platform, multi_platform_manager);
