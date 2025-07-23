#pragma once
#include <type_traits>
#include <limits>
#include <string>
#include <variant>
#include <any>

#include <cub/cub.cuh>

#include "types.hpp"

namespace golap {

// https://stackoverflow.com/a/71593852
template <class> struct is_bounded_char_array : std::false_type {};
template <size_t N> 
struct is_bounded_char_array<char[N]> : std::true_type {};

template <class> struct is_bounded_array : std::false_type {};
template <class T, size_t N> 
struct is_bounded_array<T[N]> : std::true_type {};


template <typename T>
static void mininit(T& min){
    if constexpr (std::is_arithmetic_v<T>){
        min = std::numeric_limits<T>::max();
    }else if constexpr(is_bounded_char_array<T>{}){
        memset(min, '~', sizeof(T)-1);
    }else if constexpr (std::is_same_v<T,util::Datetime>){
        std::stringstream s{"9000-01-01 00:00:00"};
        s >> min;
    }else if constexpr (std::is_same_v<T,util::Date>){
        std::stringstream s{"9000-01-01"};
        s >> min;
    }else if constexpr (std::is_same_v<T,util::Decimal32>){
        min.val = 2147483648;
    }else if constexpr (std::is_same_v<T,util::Decimal64>){
        min.val = 9223372036854775808ul;
    }else{
        // util::Padded
        memset(min.d, '~', sizeof(typename T::value_t));
    }
}

template <typename T>
static void maxinit(T& max){
    if constexpr (std::is_arithmetic_v<T>){
        max = std::numeric_limits<T>::min();
    }else if constexpr(is_bounded_char_array<T>{}){
        memset(max,' ', sizeof(T));
    }else if constexpr (std::is_same_v<T,util::Datetime>){
        std::stringstream s{"0000-01-01 00:00:00"};
        s >> max;
    }else if constexpr (std::is_same_v<T,util::Date>){
        std::stringstream s{"0000-01-01"};
        s >> max;
    }else if constexpr (std::is_same_v<T,util::Decimal32>){
        max.val = 0;
    }else if constexpr (std::is_same_v<T,util::Decimal64>){
        max.val = 0;
    }else{
        // util::Padded
        memset(max.d, ' ', sizeof(typename T::value_t));
    }
}

template <typename T>
struct DefaultCmp{
    static bool cmp(T* a, T* b){
        return *a < *b;
    }
};

template <typename T>
struct LexCmp{
    static bool cmp(T* a, T* b){
        return std::string(*a) < std::string(*b);
    }
};

struct CustomMin{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};
struct CustomMax{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (a < b) ? b : a;
    }
};

enum MetaFlags: uint8_t {
    DATA=0b00000001, META=0b00000010
};
using MetaFlagsType = std::underlying_type<MetaFlags>::type;
static MetaFlags operator&(MetaFlags x, MetaFlags y){
    return MetaFlags(MetaFlagsType(x) & MetaFlagsType(y));
}
static MetaFlags operator|(MetaFlags x, MetaFlags y){
    return MetaFlags(MetaFlagsType(x) | MetaFlagsType(y));
}


template <typename T>
struct MinMaxMeta {
    MinMaxMeta(uint64_t chunk_num, uint64_t max_tuples_in_chunk) : chunk_num(chunk_num){
        checkCudaErrors(cudaMalloc(&mins, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMalloc(&maxs, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMallocHost(&mins_hst, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMallocHost(&maxs_hst, chunk_num * sizeof(T)));

        mininit(fixed_min);
        maxinit(fixed_max);

        // assume max needs the same amount of temp space
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, (T*)nullptr, (T*)nullptr, max_tuples_in_chunk, CustomMin(), fixed_min);
        checkCudaErrors(cudaMalloc(&temp_storage, temp_storage_bytes));
        HOST_ALLOCATED.fetch_add(2*chunk_num * sizeof(T));
        DEVICE_ALLOCATED.fetch_add(2*chunk_num * sizeof(T) + temp_storage_bytes);
        is_orig = true;
    }
    // dummy constructor
    MinMaxMeta(){
        is_orig = false;
    }

    /**
     * This copy constructor lets you pass a MinMaxMeta object by value to a cuda kernel,
     * without double-frees of the dynamic memory.
     */
    MinMaxMeta(const MinMaxMeta& original){
        *this = original;
        is_orig = false;
    }

    ~MinMaxMeta(){
        if (!is_orig) return;
        cudaFree(mins);
        cudaFree(maxs);
        cudaFreeHost(mins_hst);
        cudaFreeHost(maxs_hst);

        if (temp_storage != nullptr){
            cudaFree(temp_storage);
            DEVICE_ALLOCATED.fetch_sub(temp_storage_bytes);
        }
        HOST_ALLOCATED.fetch_sub(2*chunk_num * sizeof(T));
        DEVICE_ALLOCATED.fetch_sub(2*chunk_num * sizeof(T));
    }

    __host__
    void init_chunk(uint64_t chunk_idx, T* data, uint64_t num_items, cudaStream_t stream = 0){
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, data, &mins[chunk_idx], num_items, CustomMin(), fixed_min, stream);
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, data, &maxs[chunk_idx], num_items, CustomMax(), fixed_max, stream);
    }

    __host__
    void free_temp(){
        cudaFree(temp_storage);
        temp_storage = nullptr;
        DEVICE_ALLOCATED.fetch_sub(temp_storage_bytes);
    }

    __host__
    void to_host(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(mins_hst, mins, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(maxs_hst, maxs, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
    }

    __host__
    void to_device(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(mins, mins_hst, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(maxs, maxs_hst, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
    }

    __host__
    void print_debug(){
        to_host();

        for(uint64_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx){
            if constexpr (std::is_arithmetic_v<T>) {
                std::cout << "Chunk["<<chunk_idx<<"] Min/Max: "<<+mins_hst[chunk_idx]<<" / "<<+maxs_hst[chunk_idx]<<"\n";
            }else {
                std::cout << "Chunk["<<chunk_idx<<"] Min/Max: "<<mins_hst[chunk_idx]<<" / "<<maxs_hst[chunk_idx]<<"\n";
            }
        }
    }

    __host__
    bool check_pred_host(uint64_t chunk_idx, T low, T high){
        T chunkmin = mins_hst[chunk_idx];
        T chunkmax = maxs_hst[chunk_idx];

        return ((low <= chunkmax) && (high >= chunkmin));
    }

    __device__
    void check_pred(uint64_t chunk_idx, uint16_t *out, T low, T high){
        T chunkmin = mins[chunk_idx];
        T chunkmax = maxs[chunk_idx];

        if ((low <= chunkmax) && (high >= chunkmin)){
            out[chunk_idx] = 1;
        }

    }

    T *mins,*maxs;
    T *mins_hst,*maxs_hst;

    uint64_t chunk_num;
    void *temp_storage = nullptr;
    uint64_t temp_storage_bytes = 0;
    bool is_orig = true;
    T fixed_min,fixed_max;
};


template <typename T>
struct EqHistogram {
    /**
     * A general equal width histogram for any datatype that has a reasonably defined difference metric
     */
    EqHistogram(uint64_t chunk_num, uint64_t bucket_num, uint64_t max_tuples_in_chunk) : chunk_num(chunk_num),bucket_num(bucket_num){
        checkCudaErrors(cudaMalloc(&mins, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMalloc(&maxs, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMallocHost(&mins_hst, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMallocHost(&maxs_hst, chunk_num * sizeof(T)));
        checkCudaErrors(cudaMalloc(&hist, chunk_num * bucket_num * sizeof(uint64_t)));
        checkCudaErrors(cudaMallocHost(&hist_hst, chunk_num * bucket_num * sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(hist, 0, chunk_num * bucket_num * sizeof(uint64_t)));

        mininit(fixed_min);
        maxinit(fixed_max);

        // assume max needs the same amount of temp space
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, (T*)nullptr, (T*)nullptr, max_tuples_in_chunk, CustomMin(), fixed_min);
        checkCudaErrors(cudaMalloc(&temp_storage, temp_storage_bytes));
        HOST_ALLOCATED.fetch_add(2*chunk_num * sizeof(T) + chunk_num * bucket_num * sizeof(uint64_t));
        DEVICE_ALLOCATED.fetch_add(2*chunk_num * sizeof(T) + chunk_num * bucket_num * sizeof(uint64_t) + temp_storage_bytes);
        is_orig = true;
    }
    // dummy constructor
    EqHistogram(uint64_t bucket_num):bucket_num(bucket_num){
        is_orig = false;
    }
    EqHistogram(){
        printf("EqHistogram default constructor. This should not happen! -----------------------------\n");
    }

    /**
     * This copy constructor lets you pass a EqHistogram object by value to a cuda kernel,
     * without double-frees of the dynamic memory.
     */
    EqHistogram(const EqHistogram& original){
        *this = original;
        is_orig = false;
    }

    ~EqHistogram(){
        if (!is_orig) return;
        cudaFree(mins);
        cudaFree(maxs);
        cudaFree(hist);
        cudaFreeHost(mins_hst);
        cudaFreeHost(maxs_hst);
        cudaFreeHost(hist_hst);
        if (temp_storage != nullptr){
            cudaFree(temp_storage);
            DEVICE_ALLOCATED.fetch_sub(temp_storage_bytes);
        }
        HOST_ALLOCATED.fetch_sub(2*chunk_num * sizeof(T) + chunk_num * bucket_num * sizeof(uint64_t));
        DEVICE_ALLOCATED.fetch_sub(2*chunk_num * sizeof(T) + chunk_num * bucket_num * sizeof(uint64_t));
    }

    __host__
    void init_chunk(uint64_t chunk_idx, T* data, uint64_t num_items, cudaStream_t stream = 0){
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, data, &mins[chunk_idx], num_items, CustomMin(), fixed_min, stream);
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, data, &maxs[chunk_idx], num_items, CustomMax(), fixed_max, stream);
    }

    __host__
    void free_temp(){
        cudaFree(temp_storage);
        temp_storage = nullptr;
        DEVICE_ALLOCATED.fetch_sub(temp_storage_bytes);
    }

    __host__
    void to_host(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(mins_hst, mins, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(maxs_hst, maxs, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(hist_hst, hist, chunk_num * bucket_num * sizeof(uint64_t), cudaMemcpyDefault, stream));
    }

    __host__
    void to_device(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(mins, mins_hst, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(maxs, maxs_hst, chunk_num * sizeof(T), cudaMemcpyDefault, stream));
        checkCudaErrors(cudaMemcpyAsync(hist, hist_hst, chunk_num * bucket_num * sizeof(uint64_t), cudaMemcpyDefault, stream));
    }

    __host__
    void print_debug(){
        to_host();

        for(uint64_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx){
            if constexpr (std::is_arithmetic_v<T>) {
                std::cout << "Chunk[" << chunk_idx<<"] Min/Max: "<<+mins_hst[chunk_idx]<<" / "<<+maxs_hst[chunk_idx]<<"\n";
            }else {
                std::cout << "Chunk[" << chunk_idx<<"] Min/Max: "<<mins_hst[chunk_idx]<<" / "<<maxs_hst[chunk_idx]<<"\n";
            }

            for(uint64_t bucket_idx = 0; bucket_idx < bucket_num; ++bucket_idx){
                // std::cout << "    Bucket["<<bucket_idx<<"] Min/Max/Count: "<<bucketmin<<" / "<<bucketmax<<" / "<<hist_hst[bucket_idx]<<"\n";
                std::cout << "    Bucket["<<bucket_idx<<"] Count: "<<hist_hst[chunk_idx*bucket_num + bucket_idx]<<"\n";
            }
        }
    }

    __device__
    void add(uint64_t chunk_idx, T val){
        uint64_t bucket_idx = 0;
        if (maxs[chunk_idx] != mins[chunk_idx]) {
            double inv_bucket_size = (double)bucket_num/double(maxs[chunk_idx]-mins[chunk_idx]);

            bucket_idx = ((val-mins[chunk_idx])*inv_bucket_size);
            // if val is the max, idx would be bucket_num
            bucket_idx = min(bucket_num-1,bucket_idx);
        }

        // printf("Value: %d maps to bucket_idx %lu\n",val,bucket_idx);
        uint64_t idx =  bucket_num*chunk_idx + bucket_idx;

        if (bucket_idx >= bucket_num){
            printf("Got out-of-bounds bucket_idx=%lu\n",bucket_idx);
            return;
        }

        atomicAdd((unsigned long long* ) &hist[idx], (unsigned long long) 1);
    }

    __device__
    void check_pred(uint64_t chunk_idx, int64_t bucket_idx, uint16_t *out, T low, T high){
        if (mins[chunk_idx] == maxs[chunk_idx]){
            // add method above uses the first bucket
            if (bucket_idx != 0) return;
            // mins = maxs and bucket_idx == 0
            if (low <= mins[chunk_idx] && high >= maxs[chunk_idx] && hist[chunk_idx*bucket_num + 0] != 0){
                out[chunk_idx] = 1;
            }
        }else{
            double inv_bucket_size = (double)bucket_num/double(maxs[chunk_idx]-mins[chunk_idx]);
            int64_t first_bucket = ((double(low)-mins[chunk_idx])*inv_bucket_size);
            int64_t last_bucket = ((double(high)-mins[chunk_idx])*inv_bucket_size);
            if (low == maxs[chunk_idx]) first_bucket = bucket_num-1;
            // if (bucket_idx == 0){
            //     printf("Chunk[%llu] MinMax [%u,%u] first_bucket / last_bucket = %lld / %lld\n",chunk_idx,mins[chunk_idx],maxs[chunk_idx],first_bucket,last_bucket);
            // }

            // set output if this is a relevant bucket and its content is nonzero
            if (first_bucket <= bucket_idx && last_bucket >= bucket_idx && hist[chunk_idx*bucket_num + bucket_idx] != 0){
                // printf("chunk_idx=%lu, bucket_idx=%ld is setting the bit.\n",chunk_idx,bucket_idx);
                out[chunk_idx] = 1;
            }
        }
    }

    T *mins,*maxs,*mins_hst,*maxs_hst;
    uint64_t *hist,*hist_hst;

    uint64_t chunk_num,bucket_num;
    void *temp_storage = nullptr;
    uint64_t temp_storage_bytes = 0;
    bool is_orig = true;
    T fixed_min,fixed_max;
};

template <typename T>
struct BloomMeta {
    /**
     * https://github.com/Claudenw/BloomFilter/wiki/Bloom-Filters----An-overview
     * (n, p) Bloomfilter is in dev_structs, this is a (p,m) Bloomfilter
     * FUTURE: maybe (m,n) or (m, k) would be better
     * The parameters of a bloom filter:
     * n - Number of items that can be mapped into the filter with the given fpr p[derived]
     * p - Probability of false positive [input]
     * m - bytes of the filter [input]
     * k - number of hash functions [derived]
     */
    BloomMeta(uint64_t chunk_num, double p, uint64_t m, uint64_t max_tuples_in_chunk):chunk_num(chunk_num),p(p),m(m){
        uint64_t mbits = m<<3;

        n = ceil((mbits * log(pow(0.5, log(2.0))) / log(p)));
        k = ceil(log(2.0) * (double) mbits / n);
        // n = ceil(mbits / (-k / log(1 - exp(log(p) / k))));

        // util::Log::get().info_fmt("BloomMeta for m=%lubytes =%lubits and p=%f: Calculated n=%lu, k=%lu",m,mbits,p,n,k);

        checkCudaErrors(cudaMalloc(&state, m*chunk_num));
        checkCudaErrors(cudaMallocHost(&state_hst, m*chunk_num));
        checkCudaErrors(cudaMemset(state, 0, m*chunk_num));
        golap::DEVICE_ALLOCATED += m*chunk_num;
        golap::HOST_ALLOCATED += m*chunk_num;
        is_orig = true;
    }

    BloomMeta(const BloomMeta& original){
        *this = original;
        is_orig = false;
    }
    // dummy constructor
    BloomMeta(double p, uint64_t m):p(p),m(m){
        is_orig = false;
    }
    BloomMeta(){
        printf("BloomMeta default constructor. This should not happen! -----------------------------\n");
    }

    ~BloomMeta(){
        if (!is_orig) return;
        cudaFree(state);
        cudaFreeHost(state_hst);

        golap::DEVICE_ALLOCATED -= m*chunk_num;
        golap::HOST_ALLOCATED -= m*chunk_num;
    }

    __host__
    void to_host(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(state_hst, state, m*chunk_num, cudaMemcpyDefault, stream));
    }

    __host__
    void to_device(cudaStream_t stream = 0){
        checkCudaErrors(cudaMemcpyAsync(state, state_hst, m*chunk_num, cudaMemcpyDefault, stream));
    }


    __host__
    void print_debug(){
        to_host();

        uint64_t total_bits_set = 0;

        printf("BloomFilter Metadata (m=%lubytes,p=%.2f,k=%lu,n=%lu)\n",m,p,k,n);
        for(uint64_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx){
            printf("Chunk[%lu]:\n",chunk_idx);
            total_bits_set = 0;
            for(uint64_t byte_idx = 0; byte_idx < m; ++byte_idx){
                printf("%02X ",(unsigned char)state_hst[((chunk_idx*m)>>2) + byte_idx]);
                total_bits_set += __builtin_popcount((unsigned char)state_hst[((chunk_idx*m)>>2) + byte_idx]);
            }
            printf("\ntotal_bits_set=%lu, equivalent of perfectly hashing %.2f values\n",total_bits_set, double(total_bits_set) / k);
        }
    }

    __device__ inline
    uint64_t hash(uint64_t val){
        val ^= val >> 33;
        val *= 0xff51afd7ed558ccd;
        val ^= val >> 33;
        val *= 0xc4ceb9fe1a85ec53;
        val ^= val >> 33;
        // return val & ((m<<3)-1);
        return val % (m<<3);
    }

    __device__
    void add(uint64_t chunk_idx, T val){
        uint64_t longval = _to_uint64_t(val);

        // compute start of that chunks bloom filter
        uint32_t *bloom_chunk_start = state + ((chunk_idx*m)>>2);

        // add to the bloomfilter of this chunk
        for (uint32_t i = 0; i<k; ++i){
            // k-th hash
            uint64_t bitidx = hash(longval + i);
            uint64_t wordidx = bitidx>>5;
            bitidx &= 31; // idx in word

            atomicOr(bloom_chunk_start+wordidx, (uint32_t)(1<<bitidx));
        }
    }

    __device__
    void check_pred(uint64_t chunk_idx, uint16_t *out, T val){
        uint64_t longval = _to_uint64_t(val);

        auto res = true;
        uint32_t *bloom_chunk_start = state + ((chunk_idx*m)>>2);
        for (uint32_t i = 0; i<k; ++i){
            // k-th hash
            uint64_t bitidx = hash(longval + i);
            uint64_t wordidx = bitidx>>5;
            bitidx &= 31; // idx in word

            if ((bloom_chunk_start[wordidx] & (1<<bitidx)) == 0){
                // definitely not in set
                res = false;
                // break; // let all thread do all k lookups less divergence?
            }
        }
        if (res) out[chunk_idx] = 1;

    }

    uint64_t chunk_num;
    uint64_t n;
    double p;
    uint64_t m;

    uint32_t *state,*state_hst;

    bool is_orig = true;
    uint64_t k = 0;
private:

    __device__ inline
    uint64_t _to_uint64_t(T val){
        if constexpr(std::is_integral_v<T>){
            return (uint64_t) val;
        }else if constexpr(std::is_arithmetic_v<T>){
            double x = val;
            return *((uint64_t*)&x);
        }else{
            return val.hash();
        }
        __builtin_unreachable();
    }
};

template <typename T>
__global__ void check_mmmeta(MinMaxMeta<T> mmmeta, uint16_t *out, T low, T high){
    uint64_t idx,num=mmmeta.chunk_num;

    // grid stride loop over the tuples
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        mmmeta.check_pred(idx, out, low, high);
    }
}

template <typename T>
__global__ void check_mmmeta_multiple_pts(MinMaxMeta<T> mmmeta, uint16_t *out, T* val, uint64_t* val_num){
    uint64_t idx,val_idx,num=mmmeta.chunk_num;

    // grid stride loop over the chunks
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        for (val_idx = 0; val_idx < *val_num; ++val_idx){
            mmmeta.check_pred(idx, out, val[val_idx], val[val_idx]);
        }
    }
}


template <typename T>
__global__ void fill_hist(EqHistogram<T> histogram, T* data, uint64_t chunkidx, uint64_t num){
    uint64_t idx;

    // grid stride loop over the tuples
    for (idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num; idx += blockDim.x * gridDim.x){
        histogram.add(chunkidx, data[idx]);
    }
}

template <typename T>
__global__ void check_hist(EqHistogram<T> histogram, uint16_t *out, T low, T high){
    uint64_t idx,num=histogram.chunk_num*histogram.bucket_num;
    uint64_t chunk_idx,bucket_idx;

    // grid stride loop over the buckets
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        bucket_idx = idx % histogram.bucket_num;
        chunk_idx = idx / histogram.bucket_num;
        histogram.check_pred(chunk_idx, bucket_idx, out, low, high);
        __syncwarp();
    }
}

template <typename T>
__global__ void fill_bloom(BloomMeta<T> bloommeta, T* data, uint64_t chunkidx, uint64_t num){
    uint64_t idx;

    // grid stride loop over the tuples
    for (idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num; idx += blockDim.x * gridDim.x){
        bloommeta.add(chunkidx, data[idx]);
    }
}
template <typename T>
__global__ void check_bloom(BloomMeta<T> bloommeta, uint16_t *out, T val){
    uint64_t idx,num=bloommeta.chunk_num;

    // grid stride loop over the tuples
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        bloommeta.check_pred(idx, out, val);
    }
}



template <typename T>
__global__ void check_hist_multiple_pts(EqHistogram<T> histogram, uint16_t *out, T* val, uint64_t* val_num){
    uint64_t idx,val_idx,num=histogram.chunk_num*histogram.bucket_num;
    uint64_t chunk_idx,bucket_idx;

    // grid stride loop over the buckets
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        bucket_idx = idx % histogram.bucket_num;
        chunk_idx = idx / histogram.bucket_num;
        for (val_idx = 0; val_idx < *val_num; ++val_idx){
            histogram.check_pred(chunk_idx, bucket_idx, out, val[val_idx], val[val_idx]);
        }
    }
}

__global__ static void negate(uint16_t *out, uint16_t *out_a, uint64_t num){
    for (uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        out[idx] = out_a[idx] ^ uint16_t(1);
    }
}
__global__ static void combine_or(uint16_t *out, uint16_t *out_a, uint16_t *out_b, uint64_t num){
    for (uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        out[idx] = out_a[idx] | out_b[idx];
    }
}
__global__ static void combine_and(uint16_t *out, uint16_t *out_a, uint16_t *out_b, uint64_t num){
    for (uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        out[idx] = out_a[idx] & out_b[idx];
    }
}

struct MetaChecker{
    MetaChecker(uint64_t block_num, uint64_t chunk_num, uint16_t *combined_check_dev, uint16_t *tmp_dev, cudaStream_t &stream):
        block_num(block_num), chunk_num(chunk_num),combined_check_dev(combined_check_dev), tmp_dev(tmp_dev), stream(stream){

    }

    template <typename T>
    void add_minmax(std::any &meta, T lo, T hi){
        std::any_cast<golap::MinMaxMeta<T>&>(meta).to_device(stream);

        check_mmmeta<<<block_num,512,0,stream>>>(std::any_cast<golap::MinMaxMeta<T>&>(meta), tmp_dev, lo, hi);
        // todo: do this to be safe for now, make it nicer later
        if (added){
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, combined_check_dev, tmp_dev, chunk_num);
        }else{
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, tmp_dev, tmp_dev, chunk_num);
        }

        cudaMemsetAsync(tmp_dev, 0, chunk_num*sizeof(uint16_t), stream);
        added = true;
    }

    template <typename T>
    void add_hist(std::any &meta, T lo, T hi){
        std::any_cast<golap::EqHistogram<T>&>(meta).to_device(stream);

        check_hist<<<block_num,512,0,stream>>>(std::any_cast<golap::EqHistogram<T>&>(meta), tmp_dev, lo, hi);
        // todo: do this to be safe for now, make it nicer later
        if (added){
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, combined_check_dev, tmp_dev, chunk_num);
        }else{
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, tmp_dev, tmp_dev, chunk_num);
        }

        cudaMemsetAsync(tmp_dev, 0, chunk_num*sizeof(uint16_t), stream);
        added = true;
    }

    template <typename T>
    void add_bloom(std::any &meta, T val){
        std::any_cast<golap::BloomMeta<T>&>(meta).to_device(stream);

        check_bloom<<<block_num,512,0,stream>>>(std::any_cast<golap::BloomMeta<T>&>(meta), tmp_dev, val);
        // todo: do this to be safe for now, make it nicer later
        if (added){
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, combined_check_dev, tmp_dev, chunk_num);
        }else{
            golap::combine_and<<<block_num,512,0,stream>>>(combined_check_dev, tmp_dev, tmp_dev, chunk_num);
        }

        cudaMemsetAsync(tmp_dev, 0, chunk_num*sizeof(uint16_t), stream);
        added = true;
    }

private:
    uint64_t block_num;
    uint64_t chunk_num;
    cudaStream_t stream;
    uint16_t *combined_check_dev,*tmp_dev;
    bool added = false;
};



} // end of namespace