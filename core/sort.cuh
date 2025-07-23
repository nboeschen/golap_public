#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "cub/cub.cuh"

#include "dev_structs.cuh"

namespace golap {


template <typename KEY, typename VALUE>
class Sort{
public:
    Sort(KEY* keyin, KEY* keyout, VALUE* valuein, VALUE* valueout, uint32_t num, uint32_t bits = 0):temp_mem(),keyin(keyin),keyout(keyout),
        valuein(valuein),valueout(valueout),num(num),bits(bits){
        if(bits == 0) this->bits = sizeof(KEY)*8;
        checkCudaErrors(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keyin, keyout, valuein, valueout, num, 0, this->bits));
        temp_mem.resize_num<char>(temp_bytes);
    }

    void execute(cudaStream_t stream=0){
        checkCudaErrors(cub::DeviceRadixSort::SortPairs((void*)temp_mem.ptr<char>(), temp_bytes, keyin, keyout, valuein, valueout, num, 0, bits, stream));
    }
private:
    DeviceMem temp_mem;
    KEY* keyin;
    KEY* keyout;
    VALUE* valuein;
    VALUE* valueout;
    uint32_t num;
    uint64_t temp_bytes;
    uint32_t bits;
};


} // end of namespace