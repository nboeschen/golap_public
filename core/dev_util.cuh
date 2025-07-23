#pragma once
#include <cstdint>

namespace util{


template <uint64_t SHADER_FREQ_KHZ = 1410000>
__global__ void waiting_kernel(uint64_t time_us){
    // should be used with one block one thread. Only first thread waits anyway:

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id != 0) return;

    uint64_t start = clock64();

    while( 1000 * (clock64() - start) / SHADER_FREQ_KHZ < time_us);
}

} // end of namespace util
