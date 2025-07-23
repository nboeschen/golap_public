#pragma once
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "cub/cub.cuh"

#include "dev_structs.cuh"

namespace golap {


template <typename IN_TYPE, typename RESULT_TYPE, typename REDUCTION>
class Aggregate{
public:
    Aggregate(IN_TYPE* in, RESULT_TYPE* out, uint32_t num, REDUCTION reduction, IN_TYPE init):temp_mem(),in(in),out(out),num(num),reduction(reduction),init(init){
        // (void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT d_in, OutputIteratorT d_out, int num_items, ReductionOpT reduction_op, T init, cudaStream_t stream=0, bool debug_synchronous=false)
        checkCudaErrors(cub::DeviceReduce::Reduce(nullptr, temp_bytes, in, out, num, reduction, init));
        temp_mem.resize_num<char>(temp_bytes);
    }

    void execute(cudaStream_t stream=0){
        // checkCudaErrors(cub::DeviceRadixSort::SortKeys((void*)temp_mem.ptr<char>(), temp_bytes, in, out, num, 0, sizeof(T)*8, stream));
        checkCudaErrors(cub::DeviceReduce::Reduce((void*)temp_mem.ptr<char>(), temp_bytes, in, out, num, reduction, init, stream));
    }
private:
    DeviceMem temp_mem;
    IN_TYPE* in;
    RESULT_TYPE* out;
    IN_TYPE init;
    REDUCTION reduction;
    uint32_t num;
    uint64_t temp_bytes;
};


} // end of namespace
