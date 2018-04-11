#include "npu_executor.h"
#include "npu_stream.h"
#include "npu_event.h"
#include "npu_timer.h"
#include "npu_kernel.h"
#include "npu_platform_id.h"

#include <unistd.h>

#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"
#include "tensorflow/stream_executor/lib/numbers.h"

namespace npu {


    port::Status NpuExecutor::Init(int device_ordinal,
                                   DeviceOptions device_options) {
        device_ordinal_ = device_ordinal;
        return port::Status::OK();
    }


    bool NpuExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                                KernelBase *kernel) {
        return false;
    }

    void NpuExecutor::UnloadKernel(const KernelBase *kernel) {
    }

    bool NpuExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                             const BlockDim &block_dims, const KernelBase &kernel,
                             const KernelArgsArrayBase &args) {
        return false;
    }

    void *NpuExecutor::Allocate(uint64 size) {
        return new char[size];
    }

    void *NpuExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                         uint64 offset_bytes, uint64 size_bytes) {
        return reinterpret_cast<char *>(mem->opaque()) + offset_bytes;
    }

    void NpuExecutor::Deallocate(DeviceMemoryBase *mem) {
        if (!mem->is_sub_buffer()) {
            delete[] static_cast<char *>(mem->opaque());
        }
    }

    bool NpuExecutor::HostMemoryRegister(void *location, uint64 size) {
        return true;
    }

    bool NpuExecutor::HostMemoryUnregister(void *location) {
        return true;
    }

    bool NpuExecutor::SynchronizeAllActivity() {
        return false;
    }

    bool NpuExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
        memset(location->opaque(), 0, size);
        return true;
    }

    bool NpuExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                        uint64 size) {
        memset(location->opaque(), value, size);
        return true;
    }

    port::Status NpuExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                                const void *host_src,
                                                uint64 size) {
        memcpy(gpu_dst->opaque(), host_src, size);
        return port::Status::OK();
    }

    port::Status NpuExecutor::SynchronousMemcpy(void *host_dst,
                                                const DeviceMemoryBase &gpu_src,
                                                uint64 size) {
        memcpy(host_dst, gpu_src.opaque(), size);
        return port::Status::OK();
    }

    port::Status NpuExecutor::SynchronousMemcpyDeviceToDevice(
            DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
        memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
        return port::Status::OK();
    }

    bool NpuExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                              uint64 size) {

        void *gpu_mem = location->opaque();
        AsNpuStream(stream)->EnqueueTask(
                [gpu_mem, size]() { memset(gpu_mem, 0, size); });
        return true;
    }

    bool NpuExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                             uint8 pattern, uint64 size) {
        void *gpu_mem = location->opaque();
        AsNpuStream(stream)->EnqueueTask(
                [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
        return true;
    }

    bool NpuExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                               uint32 pattern, uint64 size) {
        void *gpu_mem = location->opaque();
        AsNpuStream(stream)->EnqueueTask(
                [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
        return true;
    }

    bool NpuExecutor::Memcpy(Stream *stream, void *host_dst,
                             const DeviceMemoryBase &gpu_src, uint64 size) {
        void *src_mem = const_cast<void *>(gpu_src.opaque());
        AsNpuStream(stream)->EnqueueTask(
                [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
        return true;
    }

    bool NpuExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                             const void *host_src, uint64 size) {
        void *dst_mem = gpu_dst->opaque();
        AsNpuStream(stream)->EnqueueTask(
                [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
        return true;
    }

    bool NpuExecutor::MemcpyDeviceToDevice(Stream *stream,
                                           DeviceMemoryBase *gpu_dst,
                                           const DeviceMemoryBase &gpu_src,
                                           uint64 size) {
        void *dst_mem = gpu_dst->opaque();
        void *src_mem = const_cast<void *>(gpu_src.opaque());
        AsNpuStream(stream)->EnqueueTask(
                [src_mem, dst_mem, size]() { memcpy(src_mem, dst_mem, size); });
        return true;
    }

    bool NpuExecutor::HostCallback(Stream *stream,
                                   std::function<void()> callback) {
        AsNpuStream(stream)->EnqueueTask(callback);
        return true;
    }

    port::Status NpuExecutor::AllocateEvent(Event *event) {
        //return port::Status{port::error::UNIMPLEMENTED, ""};
        return AsNpuEvent(event)->Init();
    }

    port::Status NpuExecutor::DeallocateEvent(Event *event) {
        return AsNpuEvent(event)->Destroy();
    }

    port::Status NpuExecutor::RecordEvent(Stream *stream, Event *event) {
        return AsNpuEvent(event)->Record(stream);
    }

    port::Status NpuExecutor::WaitForEvent(Stream *stream, Event *event) {
        AsNpuStream(stream)->EnqueueTask(
                [stream]() { AsNpuStream(stream)->BlockUntilDone(); });
        return port::Status::OK();
    }

    Event::Status NpuExecutor::PollForEventStatus(Event *event) {
        return AsNpuEvent(event)->PollForStatus();
    }

    bool NpuExecutor::AllocateStream(Stream *stream) {
        return true;
    }

    void NpuExecutor::DeallocateStream(Stream *stream) {
    }

    bool NpuExecutor::AllocateTimer(Timer *timer) {
        return true;
    }

    void NpuExecutor::DeallocateTimer(Timer *timer) {
    }

    bool NpuExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
        AsNpuStream(dependent)->EnqueueTask(
                [other]() { SE_CHECK_OK(other->BlockHostUntilDone()); });
        AsNpuStream(dependent)->BlockUntilDone();
        return true;
    }

    bool NpuExecutor::StartTimer(Stream *stream, Timer *timer) {
        dynamic_cast<NpuTimer *>(timer->implementation())->Start(stream);
        return true;
    }

    bool NpuExecutor::StopTimer(Stream *stream, Timer *timer) {
        dynamic_cast<NpuTimer *>(timer->implementation())->Stop(stream);
        return true;
    }

    port::Status NpuExecutor::BlockHostUntilDone(Stream *stream) {
        AsNpuStream(stream)->BlockUntilDone();
        return port::Status::OK();
    }

    bool NpuExecutor::SupportsBlas() const {
        return PluginRegistry::Instance()
                ->GetFactory<PluginRegistry::BlasFactory>(npuPlatformId,
                                                          plugin_config_.blas())
                .ok();
    }

    blas::BlasSupport *NpuExecutor::CreateBlas() {
        PluginRegistry *registry = PluginRegistry::Instance();
        port::StatusOr<PluginRegistry::BlasFactory> status =
                registry->GetFactory<PluginRegistry::BlasFactory>(npuPlatformId,
                                                                  plugin_config_.blas());
        if (!status.ok()) {
            LOG(ERROR) << "Unable to retrieve BLAS factory: "
                       << status.status().error_message();
            return nullptr;
        }

        return status.ValueOrDie()(this);
    }

    bool NpuExecutor::SupportsDnn() const {
        return false;
    }

    dnn::DnnSupport *NpuExecutor::CreateDnn() {
        return nullptr;
    }

    bool NpuExecutor::SupportsFft() const {
        return PluginRegistry::Instance()
                ->GetFactory<PluginRegistry::FftFactory>(npuPlatformId,
                                                         plugin_config_.fft())
                .ok();
    }

    fft::FftSupport *NpuExecutor::CreateFft() {
        PluginRegistry *registry = PluginRegistry::Instance();
        port::StatusOr<PluginRegistry::FftFactory> status =
                registry->GetFactory<PluginRegistry::FftFactory>(npuPlatformId,
                                                                 plugin_config_.fft());
        if (!status.ok()) {
            LOG(ERROR) << "Unable to retrieve FFT factory: "
                       << status.status().error_message();
            return nullptr;
        }

        return status.ValueOrDie()(this);
    }


    bool NpuExecutor::SupportsRng() const {
        return PluginRegistry::Instance()
                ->GetFactory<PluginRegistry::RngFactory>(npuPlatformId,
                                                         plugin_config_.rng())
                .ok();
    }

    rng::RngSupport *NpuExecutor::CreateRng() {
        PluginRegistry *registry = PluginRegistry::Instance();
        port::StatusOr<PluginRegistry::RngFactory> status =
                registry->GetFactory<PluginRegistry::RngFactory>(npuPlatformId,
                                                                 plugin_config_.rng());
        if (!status.ok()) {
            LOG(ERROR) << "Unable to retrieve RNG factory: "
                       << status.status().error_message();
            return nullptr;
        }

        return status.ValueOrDie()(this);
    }

    bool NpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
        return true;
    }

    port::Status NpuExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
        return port::Status::OK();
    }

    SharedMemoryConfig NpuExecutor::GetDeviceSharedMemoryConfig() {
        LOG(INFO) << "Shared memory configuration is unsupported for NPU "
                  << "executors.";
        return SharedMemoryConfig::kDefault;
    }

    port::Status NpuExecutor::SetDeviceSharedMemoryConfig(
            SharedMemoryConfig config) {
        string error_msg{
                "Shared memory configuration is unsupported for NPU "
                        "executors."};
        LOG(INFO) << error_msg;
        return port::Status{port::error::UNIMPLEMENTED, error_msg};
    }

    bool NpuExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
        return false;
    }

    bool NpuExecutor::GetSymbol(const string& symbol_name, void **mem,
                                size_t *bytes) {
        return false;
    }

    std::unique_ptr<internal::EventInterface>
    NpuExecutor::CreateEventImplementation() {
        return std::unique_ptr<internal::EventInterface>(new NpuEvent(this));
    }

    std::unique_ptr<internal::KernelInterface>
    NpuExecutor::CreateKernelImplementation() {
        return std::unique_ptr<internal::KernelInterface>(new NpuKernel());
    }

    std::unique_ptr<internal::StreamInterface>
    NpuExecutor::GetStreamImplementation() {
        return std::unique_ptr<internal::StreamInterface>(new NpuStream(this));
    }

    std::unique_ptr<internal::TimerInterface>
    NpuExecutor::GetTimerImplementation() {
        return std::unique_ptr<internal::TimerInterface>(new NpuTimer(this));
    }

    DeviceDescription *NpuExecutor::PopulateDeviceDescription() const {
        internal::DeviceDescriptionBuilder builder;

        builder.set_device_address_bits(64);

        // TODO(rspringer): How to report a value that's based in reality but that
        // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
        builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);

        float cycle_counter_frequency = 1e9;
        builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

        auto built = builder.Build();
        return built.release();
    }

    namespace gpu = ::perftools::gputools;

    void initialize_npu_executor() {
        *gpu::internal::MakeCUDAExecutorImplementation() = [](
                const gpu::PluginConfig &config) {
            return new npu::NpuExecutor{config};
        };
    }

}  // namespace npu

REGISTER_MODULE_INITIALIZER(
    npu_executor, {npu::initialize_npu_executor();});
