#include "../../nnvm/tvm/dlpack/include/dlpack/dlpack.h"
#include "../../nnvm/tvm/include/tvm/runtime/module.h"
#include "../../nnvm/tvm/include/tvm/runtime/registry.h"
#include "../../nnvm/tvm/include/tvm/runtime/packed_func.h"

#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <vector>

int main()
{
	// tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("model/model_lib.so");

    // json graph
    std::ifstream json_in("model/model_graph.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in("model/model_graph.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_cpu = kDLCPU;
    int device_gpu = kDLGPU;
    int device_cpu_id = 0;
    int device_gpu_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_gpu, device_gpu_id);

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224};
    int nbytes_float32 = 4;
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_cpu, device_cpu_id, &x);
    // load image data saved in binary
    std::ifstream data_fin("data/cat.bin", std::ios::binary);
    data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * nbytes_float32);

    // load synset saved in string
    std::ifstream synset_fin("data/synset", std::ios::in);
    std::string synset_data((std::istreambuf_iterator<char>(synset_fin)), std::istreambuf_iterator<char>());
    synset_fin.close();
    std::vector<std::string> synset;
    int pos = 0;
    int size = synset_data.length();
    for (int i = 0; i < size; ++i)
    {
    	pos = synset_data.find('\n', i);
    	if (pos != -1)
    	{
    		std::string s = synset_data.substr(i, pos-i);
    		synset.push_back(s);
    		i = pos;
    	}
    }

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* y;
    int out_ndim = 1;
    int64_t out_shape[1] = {1000, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_cpu, device_cpu_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    // get the maximum position in output vector
    auto y_iter = static_cast<float*>(y->data);
    auto max_iter = std::max_element(y_iter, y_iter + 1000);
    auto max_index = std::distance(y_iter, max_iter);
    std::cout << synset[(int)max_index] << std::endl;

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
