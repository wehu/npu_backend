#ifndef TENSORFLOW_NPU_EXECUTOR_H_
#define TENSORFLOW_NPU_EXECUTOR_H_

#include <set>
#include <unordered_map>

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuExecutor : public perftools::gputools::internal::StreamExecutorInterface {
    public:
        explicit NpuExecutor(const PluginConfig &plugin_config)
                : device_ordinal_(0),
                  plugin_config_(plugin_config) {}

        ~NpuExecutor() override {};

        port::Status Init(int device_ordinal, DeviceOptions device_options) override;

        bool GetKernel(const MultiKernelLoaderSpec &spec,
                       KernelBase *kernel) override;
        void UnloadKernel(const KernelBase *kernel) override;

        bool Launch(Stream *stream, const ThreadDim &thread_dims,
                    const BlockDim &block_dims, const KernelBase &k,
                    const KernelArgsArrayBase &args) override;

        void *Allocate(uint64 size) override;

        void *AllocateSubBuffer(DeviceMemoryBase *mem, uint64 offset_bytes,
                                uint64 size_bytes) override;

        void Deallocate(DeviceMemoryBase *mem) override;

        void *HostMemoryAllocate(uint64 size) override {
            return new char[size];
        }

        void HostMemoryDeallocate(void *location) override {
        }

        bool HostMemoryRegister(void *location, uint64 size) override;

        bool HostMemoryUnregister(void *location) override;

        bool SynchronizeAllActivity() override;

        bool SynchronousMemZero(DeviceMemoryBase *location, uint64 size) override;

        bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                               uint64 size) override;

        port::Status SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                       const void *host_src, uint64 size) override;

        port::Status SynchronousMemcpy(void *host_dst,
                                       const DeviceMemoryBase &gpu_src,
                                       uint64 size) override;

        port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *gpu_dst,
                                                     const DeviceMemoryBase &gpu_src,
                                                     uint64 size) override;

        bool MemZero(Stream *stream, DeviceMemoryBase *location,
                     uint64 size) override;
        bool Memset(Stream *stream, DeviceMemoryBase *location, uint8 pattern,
                    uint64 size) override;
        bool Memset32(Stream *stream, DeviceMemoryBase *location, uint32 pattern,
                      uint64 size) override;

        bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &gpu_src,
                    uint64 size) override;
        bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst, const void *host_src,
                    uint64 size) override;

        bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                                  const DeviceMemoryBase &gpu_src,
                                  uint64 size) override;

        bool HostCallback(Stream *stream, std::function<void()> callback) override;

        bool AllocateStream(Stream *stream) override;

        void DeallocateStream(Stream *stream) override;

        bool CreateStreamDependency(Stream *dependent, Stream *other) override;

        bool AllocateTimer(Timer *timer) override;

        void DeallocateTimer(Timer *timer) override;

        bool StartTimer(Stream *stream, Timer *timer) override;

        bool StopTimer(Stream *stream, Timer *timer) override;

        port::Status AllocateEvent(Event *event) override;

        port::Status DeallocateEvent(Event *event) override;

        port::Status RecordEvent(Stream *stream, Event *event) override;

        port::Status WaitForEvent(Stream *stream, Event *event) override;

        Event::Status PollForEventStatus(Event *event) override;

        port::Status BlockHostUntilDone(Stream *stream) override;

        int PlatformDeviceCount() override { return 1; }

        port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override;

        bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override;

        SharedMemoryConfig GetDeviceSharedMemoryConfig() override;

        port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config) override;

        bool DeviceMemoryUsage(int64 *free, int64 *total) const override;

        bool GetSymbol(const string& symbol_name, void **mem, size_t *bytes) override;

        DeviceDescription *PopulateDeviceDescription() const override;

        bool SupportsBlas() const override;

        blas::BlasSupport *CreateBlas() override;

        bool SupportsFft() const override;

        fft::FftSupport *CreateFft() override;

        bool SupportsRng() const override;

        rng::RngSupport *CreateRng() override;

        bool SupportsDnn() const override;

        dnn::DnnSupport *CreateDnn() override;

        std::unique_ptr<perftools::gputools::internal::EventInterface> CreateEventImplementation() override;

        std::unique_ptr<perftools::gputools::internal::KernelInterface> CreateKernelImplementation() override;

        std::unique_ptr<perftools::gputools::internal::StreamInterface> GetStreamImplementation() override;

        std::unique_ptr<perftools::gputools::internal::TimerInterface> GetTimerImplementation() override;

    private:

        int device_ordinal_;

        PluginConfig plugin_config_;

        SE_DISALLOW_COPY_AND_ASSIGN(NpuExecutor);
    };

}  // namespace npu

#endif  // TENSORFLOW_NPU_EXECUTOR_H_
