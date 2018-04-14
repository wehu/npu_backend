# npu_backend

This is a tenserflow backend which is like a gpu backend but all kernels is jit compiled to run on cpu.

## How to enable this backend

* Clone https://github.com/tensorflow/tensorflow
* Clone this https://github.com/wehu/npu_backend to tensorflow/compiler/plugin
* Change tensorflow/compiler/plugin/BUILD to add this backend
* Since this backend uses the boost preprocessor and boost into WORKSPACE

## How to run a simple test

* Change your python test to run with XLA_NPU device
* Run your test

for example
