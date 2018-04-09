#include "npu_executor.h"
#include "npu_stream.h"
#include "npu_event.h"
#include "npu_timer.h"
#include "npu_kernel.h"

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
        return true;
    }

    void NpuExecutor::UnloadKernel(const KernelBase *kernel) {
    }

    bool NpuExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                             const BlockDim &block_dims, const KernelBase &kernel,
                             const KernelArgsArrayBase &args) {
        return true;
    }

    void *NpuExecutor::Allocate(uint64 size) {
        return nullptr;
    }

    void *NpuExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                         uint64 offset_bytes, uint64 size_bytes) {
        return nullptr;
    }

    void NpuExecutor::Deallocate(DeviceMemoryBase *mem) {
    }

    bool NpuExecutor::HostMemoryRegister(void *location, uint64 size) {
        return true;
    }

    bool NpuExecutor::HostMemoryUnregister(void *location) {
        return true;
    }

    bool NpuExecutor::SynchronizeAllActivity() {
        return true;
    }

    bool NpuExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
        return true;
    }

    bool NpuExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                        uint64 size) {
        return true;
    }

    port::Status NpuExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                                const void *host_src,
                                                uint64 size) {
        return port::Status::OK();
    }

    port::Status NpuExecutor::SynchronousMemcpy(void *host_dst,
                                                const DeviceMemoryBase &gpu_src,
                                                uint64 size) {
        return port::Status::OK();
    }

    port::Status NpuExecutor::SynchronousMemcpyDeviceToDevice(
            DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
        return port::Status::OK();
    }

    bool NpuExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                              uint64 size) {
        return true;
    }

    bool NpuExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                             uint8 pattern, uint64 size) {
        return true;
    }

    bool NpuExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                               uint32 pattern, uint64 size) {
        return true;
    }

    bool NpuExecutor::Memcpy(Stream *stream, void *host_dst,
                             const DeviceMemoryBase &gpu_src, uint64 size) {
        return true;
    }

    bool NpuExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                             const void *host_src, uint64 size) {
        return true;
    }

    bool NpuExecutor::MemcpyDeviceToDevice(Stream *stream,
                                           DeviceMemoryBase *gpu_dst,
                                           const DeviceMemoryBase &gpu_src,
                                           uint64 size) {
        return true;
    }

    bool NpuExecutor::HostCallback(Stream *stream,
                                   std::function<void()> callback) {
        return true;
    }

    port::Status NpuExecutor::AllocateEvent(Event *event) {
        return port::Status::OK();
    }

    port::Status NpuExecutor::DeallocateEvent(Event *event) {
        return port::Status::OK();
    }

    port::Status NpuExecutor::RecordEvent(Stream *stream, Event *event) {
        return port::Status::OK();
    }

    port::Status NpuExecutor::WaitForEvent(Stream *stream, Event *event) {
        return port::Status::OK();
    }

    Event::Status NpuExecutor::PollForEventStatus(Event *event) {
        return Event::Status::kComplete;
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
        return true;
    }

    bool NpuExecutor::StartTimer(Stream *stream, Timer *timer) {
        return true;
    }

    bool NpuExecutor::StopTimer(Stream *stream, Timer *timer) {
        return true;
    }

    port::Status NpuExecutor::BlockHostUntilDone(Stream *stream) {
        return port::Status::OK();
    }

    blas::BlasSupport *NpuExecutor::CreateBlas() {
        return nullptr;
    }

    dnn::DnnSupport *NpuExecutor::CreateDnn() {
        return nullptr;
    }

    fft::FftSupport *NpuExecutor::CreateFft() {
        return nullptr;
    }

    rng::RngSupport *NpuExecutor::CreateRng() {
        return nullptr;
    }

    bool NpuExecutor::SupportsDnn() const {
        return false;
    }

    bool NpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
        return true;
    }

    port::Status NpuExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
        return port::Status::OK();
    }

    SharedMemoryConfig NpuExecutor::GetDeviceSharedMemoryConfig() {
        return SharedMemoryConfig::kDefault;
    }

    port::Status NpuExecutor::SetDeviceSharedMemoryConfig(
            SharedMemoryConfig config) {
        return port::Status::OK();
    }

    bool NpuExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
        return true;
    }

    bool NpuExecutor::GetSymbol(const string& symbol_name, void **mem,
                                size_t *bytes) {
        return true;
    }

    bool NpuExecutor::SupportsBlas() const { return false; }

    bool NpuExecutor::SupportsFft() const { return false; }

    bool NpuExecutor::SupportsRng() const { return false; }

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
