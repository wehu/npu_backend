#ifndef TENSORFLOW_NPU_KERNEL_H_
#define TENSORFLOW_NPU_KERNEL_H_

#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace npu {

    using namespace perftools::gputools;

    class NpuKernel : public internal::KernelInterface {
    public:
        NpuKernel() : arity_(0),
                      preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

        ~NpuKernel() override {}

        void set_arity(unsigned arity) { arity_ = arity; }
        unsigned Arity() const override { return arity_; }

        void SetPreferredCacheConfig(KernelCacheConfig config) override {
            preferred_cache_config_ = config;
        }

        KernelCacheConfig GetPreferredCacheConfig() const override {
            return preferred_cache_config_;
        }

    private:
        unsigned arity_;            // Number of formal parameters the kernel takes.

        KernelCacheConfig preferred_cache_config_;
    };

    inline const NpuKernel *AsNpuKernel(const KernelBase *kernel) {
        return static_cast<const NpuKernel *>(kernel->implementation());
    }

    inline NpuKernel *AsNpuKernel(KernelBase *kernel) {
        return static_cast<NpuKernel *>(kernel->implementation());
    }

}  // namespace npu

#endif  // TENSORFLOW_NPU_KERNEL_H_
