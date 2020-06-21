import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

ctx = tvm.gpu()
# load the module back.
loaded_json = open("./deploy/model-y1-test2/deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy/model-y1-test2/deploy_lib.so")
loaded_params = bytearray(open("./deploy/model-y1-test2/deploy_param.params", "rb").read())

data_shape = (1, 112, 112, 3)
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Tiny benchmark test.
import time
for i in range(100):
   t0 = time.time()
   module.run(data=input_data)
   print(time.time() - t0)

# Inference = 10ms on GTX1060