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

namespace se = perftools::gputools;

namespace xla {
    namespace npu {

        class NpuExecutor : public se::internal::StreamExecutorInterface {
        public:
            explicit NpuExecutor(const se::PluginConfig &plugin_config)
                    : device_ordinal_(0),
                      plugin_config_(plugin_config) {}

            ~NpuExecutor() override {};

            se::port::Status Init(int device_ordinal, se::DeviceOptions device_options) override;

            bool GetKernel(const se::MultiKernelLoaderSpec &spec,
                           se::KernelBase *kernel) override;

            void UnloadKernel(const se::KernelBase *kernel) override;

            bool Launch(se::Stream *stream, const se::ThreadDim &thread_dims,
                        const se::BlockDim &block_dims, const se::KernelBase &k,
                        const se::KernelArgsArrayBase &args) override;

            void *Allocate(se::uint64 size) override;

            void *AllocateSubBuffer(se::DeviceMemoryBase *mem, se::uint64 offset_bytes,
                                    se::uint64 size_bytes) override;

            void Deallocate(se::DeviceMemoryBase *mem) override;

            void *HostMemoryAllocate(se::uint64 size) override {
                return new char[size];
            }

            void HostMemoryDeallocate(void *location) override {
            }

            bool HostMemoryRegister(void *location, se::uint64 size) override;

            bool HostMemoryUnregister(void *location) override;

            bool SynchronizeAllActivity() override;

            bool SynchronousMemZero(se::DeviceMemoryBase *location, se::uint64 size) override;

            bool SynchronousMemSet(se::DeviceMemoryBase *location, int value,
                                   se::uint64 size) override;

            se::port::Status SynchronousMemcpy(se::DeviceMemoryBase *gpu_dst,
                                           const void *host_src, se::uint64 size) override;

            se::port::Status SynchronousMemcpy(void *host_dst,
                                           const se::DeviceMemoryBase &gpu_src,
                                           se::uint64 size) override;

            se::port::Status SynchronousMemcpyDeviceToDevice(se::DeviceMemoryBase *gpu_dst,
                                                         const se::DeviceMemoryBase &gpu_src,
                                                             se::uint64 size) override;

            bool MemZero(se::Stream *stream, se::DeviceMemoryBase *location,
                         se::uint64 size) override;

            bool Memset(se::Stream *stream, se::DeviceMemoryBase *location, se::uint8 pattern,
                        se::uint64 size) override;

            bool Memset32(se::Stream *stream, se::DeviceMemoryBase *location, se::uint32 pattern,
                          se::uint64 size) override;

            bool Memcpy(se::Stream *stream, void *host_dst, const se::DeviceMemoryBase &gpu_src,
                        se::uint64 size) override;

            bool Memcpy(se::Stream *stream, se::DeviceMemoryBase *gpu_dst, const void *host_src,
                        se::uint64 size) override;

            bool MemcpyDeviceToDevice(se::Stream *stream, se::DeviceMemoryBase *gpu_dst,
                                      const se::DeviceMemoryBase &gpu_src,
                                      se::uint64 size) override;

            bool HostCallback(se::Stream *stream, std::function<void()> callback) override;

            bool AllocateStream(se::Stream *stream) override;

            void DeallocateStream(se::Stream *stream) override;

            bool CreateStreamDependency(se::Stream *dependent, se::Stream *other) override;

            bool AllocateTimer(se::Timer *timer) override;

            void DeallocateTimer(se::Timer *timer) override;

            bool StartTimer(se::Stream *stream, se::Timer *timer) override;

            bool StopTimer(se::Stream *stream, se::Timer *timer) override;

            se::port::Status AllocateEvent(se::Event *event) override;

            se::port::Status DeallocateEvent(se::Event *event) override;

            se::port::Status RecordEvent(se::Stream *stream, se::Event *event) override;

            se::port::Status WaitForEvent(se::Stream *stream, se::Event *event) override;

            se::Event::Status PollForEventStatus(se::Event *event) override;

            se::port::Status BlockHostUntilDone(se::Stream *stream) override;

            int PlatformDeviceCount() override { return 1; }

            se::port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override;

            bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override;

            se::SharedMemoryConfig GetDeviceSharedMemoryConfig() override;

            se::port::Status SetDeviceSharedMemoryConfig(se::SharedMemoryConfig config) override;

            bool DeviceMemoryUsage(se::int64 *free, se::int64 *total) const override;

            bool GetSymbol(const se::string &symbol_name, void **mem, size_t *bytes) override;

            se::DeviceDescription *PopulateDeviceDescription() const override;

            bool SupportsBlas() const override;

            se::blas::BlasSupport *CreateBlas() override;

            bool SupportsFft() const override;

            se::fft::FftSupport *CreateFft() override;

            bool SupportsRng() const override;

            se::rng::RngSupport *CreateRng() override;

            bool SupportsDnn() const override;

            se::dnn::DnnSupport *CreateDnn() override;

            std::unique_ptr<se::internal::EventInterface> CreateEventImplementation() override;

            std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation() override;

            std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation() override;

            std::unique_ptr<se::internal::TimerInterface> GetTimerImplementation() override;

        private:

            int device_ordinal_;

            se::PluginConfig plugin_config_;

            SE_DISALLOW_COPY_AND_ASSIGN (NpuExecutor);
        };

    }  // namespace npu
} // namespace xla

#endif  // TENSORFLOW_NPU_EXECUTOR_H_
