TVM_ROOT=$(shell cd ../../nnvm/tvm; pwd)
NNVM_PATH=$(shell cd ../../nnvm; pwd)
DMLC_CORE=${TVM_ROOT}/dmlc-core
CUDA_PATH=/usr/local/cuda

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\
	-I${CUDA_PATH}/include\

PKG_LDFLAGS = -L${CUDA_PATH}/lib64 -lcuda -lcudart -lnvrtc -ldl -lpthread

.PHONY: clean all

all: resnet18_classifier

libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

inference.o: inference.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

resnet18_classifier: inference.o libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

clean:
	rm -f *.o resnet18_classifier
