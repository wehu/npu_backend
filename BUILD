# Description:
#   NPU-specific components in XLA service implementation.

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "npu_hlo_to_ir_bindings",
    srcs = ["npu_hlo_to_ir_bindings.cc"],
    hdrs = ["npu_hlo_to_ir_bindings.h"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "npu_executable",
    srcs = [
        "npu_executable.cc",
        "npu_kernel_thunk.cc",
        "npu_thunk_schedule.cc",
        "npu_tuple_thunk.cc",
    ],
    hdrs = [
        "npu_executable.h",
        "npu_kernel_thunk.h",
        "npu_thunk.h",
        "npu_thunk_schedule.h",
        "npu_tuple_thunk.h",
    ],
    deps = [
        ":npu_buffer_allocations",
        ":npu_hlo_schedule",
        ":npu_stream_assignment",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:tuple_points_to_analysis",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "npu_platform_id",
    srcs = ["npu_platform_id.cc"],
    hdrs = ["npu_platform_id.h"],
    deps = ["//tensorflow/stream_executor"],
)

cc_library(
    name = "npu_stream_executor",
    srcs = [
        "npu_event.cc",
        "npu_executor.cc",
        "npu_stream.cc",
        "npu_timer.cc",
    ],
    hdrs = [
        "npu_event.h",
        "npu_executor.h",
        "npu_kernel.h",
        "npu_stream.h",
        "npu_timer.h",
    ],
    deps = [
        ":npu_platform_id",
        "//tensorflow/stream_executor",
    ],
)

cc_library(
    name = "npu_hlo_schedule",
    srcs = ["npu_hlo_schedule.cc"],
    hdrs = ["npu_hlo_schedule.h"],
    deps = [
        ":npu_stream_assignment",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:hlo_scheduling",
    ],
)

cc_library(
    name = "npu_constants",
    srcs = ["npu_constants.cc"],
    hdrs = ["npu_constants.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "npu_ir_emitter",
    srcs = ["npu_ir_emitter.cc"],
    hdrs = [
        "npu_ir_emitter.h",
        "npu_ir_emitter_context.h",
    ],
    deps = [
        ":npu_executable",
        ":npu_hlo_to_ir_bindings",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ops",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:core",
        "@llvm//:support",
    ],
)

cc_library(
    name = "npu_compiler",
    srcs = ["npu_compiler.cc"],
    hdrs = [
        "npu_compiler.h",
    ],
    deps = [
        ":npu_executable",
        ":npu_ir_emitter",
        ":npu_platform_id",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "@llvm//:core",
    ],
)

cc_library(
    name = "npu_infeed_manager",
    srcs = ["npu_infeed_manager.cc"],
    hdrs = ["npu_infeed_manager.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "npu_buffer_allocations",
    srcs = ["npu_buffer_allocations.cc"],
    hdrs = ["npu_buffer_allocations.h"],
    deps = [
        ":npu_constants",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "npu_stream_assignment",
    srcs = ["npu_stream_assignment.cc"],
    hdrs = ["npu_stream_assignment.h"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "npu_transfer_manager",
    srcs = ["npu_transfer_manager.cc"],
    hdrs = [
        "npu_transfer_manager.h",
    ],
    deps = [
        ":npu_compiler",
        ":npu_infeed_manager",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "npu_platform",
    srcs = ["npu_platform.cc"],
    hdrs = [
        "npu_platform.h",
    ],
    deps = [
        ":npu_platform_id",
        ":npu_stream_executor",
        "//tensorflow/stream_executor",
    ],
)

cc_library(
    name = "xla_npu_device",
    srcs = ["xla_npu_device.cc"],
    deps = [
        ":npu_compiler",
        ":npu_executable",
        ":npu_infeed_manager",
        ":npu_platform",
        ":npu_transfer_manager",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
