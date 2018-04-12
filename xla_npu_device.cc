#include "npu_platform_id.h"

#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"

namespace npu {

    using namespace tensorflow;

    const char* const DEVICE_NPU_XLA_JIT = "XLA_NPU_JIT";
    const char* const DEVICE_XLA_NPU = "XLA_NPU";


    class XlaNpuDeviceFactory : public DeviceFactory {
    public:
        Status CreateDevices(const SessionOptions& options,
                             const string& name_prefix,
                             std::vector<Device*>* devices) override;
    };

    Status XlaNpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                              const string& name_prefix,
                                              std::vector<Device*>* devices) {
        static XlaDeviceOpRegistrations* registrations =
                RegisterXlaDeviceKernels(DEVICE_XLA_NPU, DEVICE_NPU_XLA_JIT);
        (void)registrations;

        std::unique_ptr<XlaDevice> device;
        Status status = XlaDevice::Create("NPU", DEVICE_XLA_NPU, 0,
                                          DEVICE_NPU_XLA_JIT, options, name_prefix,
                /*register_device_for_compilation=*/true,
                /*transfer_as_literal=*/false, &device);
        if (!status.ok()) {
            // Treat failures as non-fatal; there might not be a GPU in the machine.
            VLOG(1) << "Failed to create XLA_NPU device: " << status;
            return Status::OK();
        }
        devices->push_back(device.release());
        return Status::OK();
    }

    // NPU XLA Factory registration

    REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_NPU, XlaNpuDeviceFactory);

    // Kernel registrations

    constexpr std::array<DataType, 6> kAllXlaNpuTypes = {
            {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BOOL}};

    REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_NPU, XlaLocalLaunchOp, kAllXlaNpuTypes);
    REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_NPU, kAllXlaNpuTypes);

    // XLA backend registration

    // NPU operator filter
    bool NpuOpFilter(KernelDef* kdef) {
        return true;
    }

    constexpr std::array<DataType, 8> kNpuAllTypes = {
            {DT_UINT32, DT_UINT64, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE,
                    DT_COMPLEX64, DT_BOOL}};

    REGISTER_XLA_BACKEND(DEVICE_NPU_XLA_JIT, kNpuAllTypes, NpuOpFilter);

}  // namespace npu


static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
    return xla::MakeUnique<xla::ComputationPlacer>();
}

static bool InitModule() {
    xla::ComputationPlacer::RegisterComputationPlacer(npu::npuPlatformId,
                                                      &CreateComputationPlacer);
    return true;
}

static bool module_initialized = InitModule();
