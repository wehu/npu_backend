# npu_backend

This is a tenserflow backend which is like a gpu backend but all kernels are jit-compiled to run on cpu.

NOTE: Just an example and not all is implemented.

## How to enable this backend

* Clone https://github.com/tensorflow/tensorflow
* Clone this https://github.com/wehu/npu_backend to tensorflow/compiler/plugin
* Run configure to enable jit by default
* Change tensorflow/compiler/plugin/BUILD to add this backend
```
--- a/tensorflow/compiler/plugin/BUILD
+++ b/tensorflow/compiler/plugin/BUILD
@@ -37,7 +37,7 @@ package(
 cc_library(
     name = "plugin",
     deps = [
-        #"//tensorflow/compiler/plugin/example:example_lib",
+        "//tensorflow/compiler/plugin/npu_backend:xla_npu_device",
     ],
 )
```
* Since this backend uses the boost preprocessor and boost into WORKSPACE
```
git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "239ce40e42ab0e3fe7ce84c2e9303ff8a277c41a",
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

```

## How to run a simple test

* Change your python test to run with XLA_NPU device
* Run your test

for example,
```
--- a/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
+++ b/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
@@ -103,7 +103,8 @@ def train():
   with tf.name_scope('dropout'):
     keep_prob = tf.placeholder(tf.float32)
     tf.summary.scalar('dropout_keep_probability', keep_prob)
-    dropped = tf.nn.dropout(hidden1, keep_prob)
+    with tf.device("/job:localhost/replica:0/task:0/device:XLA_NPU:0"):
+      dropped = tf.nn.dropout(hidden1, keep_prob)
 
   # Do not apply softmax activation yet, see below.
   y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
```

`bazel test tensorflow/examples/tutorials/mnist:mnist_with_summaries_test`
