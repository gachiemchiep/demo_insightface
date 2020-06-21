import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd
import os

# ssh-model-final doesn't work

prefix = "../3rdparty/insightface/models/model-y1-test2/model"
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
image_size = (112, 112)
opt_level = 3

shape_dict = {'data': (1, 3, *image_size)}
# compile for cuda
target = tvm.target.cuda("llvm device=0")
# "target" means your target platform you want to compile.

#target = tvm.target.create("llvm -mcpu=broadwell")
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
with nnvm.compiler.build_config(opt_level=opt_level):
   graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)

out_dir = "deploy/model-y1-test2"
os.makedirs(out_dir)

lib.export_library("{}/deploy_lib.so".format(out_dir))
print('lib export succeefully')
with open("{}/deploy_graph.json".format(out_dir), "w") as fo:
   fo.write(graph.json())
with open("{}/deploy_param.params".format(out_dir), "wb") as fo:
   fo.write(nnvm.compiler.save_param_dict(params))