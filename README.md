# 

SSH : Single Stage Headless Face Detector

TVM Stack: End to End Optimization for Deep Learning

TVM : mxnet's deep learnong optimazor, still version 0.6, doesn't know whether it is mature enough

TVM can work on embedded device [tinyml](https://tvm.apache.org/2020/06/04/tinyml-how-tvm-is-taming-tiny)


[insightface + tvm tutorial](https://github.com/deepinsight/insightface/wiki/Tutorial:-Deploy-Face-Recognition-Model-via-TVM)

Two steps : 
1. compile the trained model
2. Inferecing using cpp


[Tutorial: Deploy Face Recognition Model via TVM](https://github.com/deepinsight/insightface/wiki/Tutorial:-Deploy-Face-Recognition-Model-via-TVM)

## Installation

```bash
llvm 6.0.1
clang 6.0.1
tvm v0.6
python 3.7
```

llvm 6.01, build from source

```bash
# From https://releases.llvm.org/download.html
# see https://clang.llvm.org/get_started.html to build with clang enabled
# Download "LLVM source code" and "Clang source code"

# Put into the following location
$HOME/opt/llvm/
├── cfe-6.0.1.src               clang
├── llvm-6.0.1.build            build directory
├── llvm-6.0.1.src              llvm


# we use the miniconda3 's py2 environment as linked python version
# python is used for test suit so it is not important
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=cfe-6.0.1.src -DPYTHON_EXECUTABLE=/home/gachiemchiep/miniconda3/envs/py2/bin/python ../llvm-6.0.1.src

make -j4
make check-clang

# add llvm/build/bin to your path
export PATH="$HOME/opt/tvm/llvm-6.0.1.build/bin:$PATH"

# compile tvm
# use the 0.6 : tvm drop support for nnvm from 0,7, currently insightface using nnvm
git clone --recursive -b v0.6 https://github.com/apache/incubator-tvm tvm
git submodule init
git submodule update
mkdir build; cd build
cp ../cmake/config.cmake .

# edit config.cmake
set(USE_CUDA ON)
set(USE_LLVM ON)
set(USE_BLAS openblas)
set(USE_CUDNN ON)
set(USE_CUBLAS ON)

# build
cmake ..
make -j4

# install python binding
# developer
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
# for user
cd python; python setup.py install --record files.txt
cd topi/python ; python setup.py install --record files.txt
cd nnvm/python; python setup.py install --record files.txt
```

## Convert models

Go to [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), then download and put inside *insightface/models

```bash


```

## Reference

1. https://llvm.org/docs/GettingStarted.html
2.  https://clang.llvm.org/get_started.html
3. https://releases.llvm.org/download.html
4. https://github.com/deepinsight/insightface/wiki/Tutorial:-Deploy-Face-Recognition-Model-via-TVM