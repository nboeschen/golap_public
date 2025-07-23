#pragma once
#include <iostream>
#include <atomic>
#include <numaif.h>

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "util.hpp"


namespace golap {

static std::atomic<int64_t> DEVICE_ALLOCATED{0};
static std::atomic<int64_t> HOST_ALLOCATED{0};

/**
 * A struct encapsulating a type. Use e.g. as Tag<int>{} in constructors for device/host mem.
 * See https://stackoverflow.com/a/31616949.
 */
template<class T>
struct Tag{
    using type=T;
};

/**
 * A struct for defining the location of a memory region.
 */
struct MemLoc{
    enum {GPU, CPU} type;
    int info;
    friend std::ostream& operator<<(std::ostream &out, MemLoc const& obj){
        switch (obj.type){
            case GPU: out << "GPU"; break;
            case CPU: out << "CPU"; break;
        }
        out << "#" << obj.info;
        return out;
    }
};

/**
 * A helper struct for allocating aligned memory ptrs.
 */
struct AllocHelper{
    AllocHelper(uint64_t bytes, uint64_t alloc_unit, uint64_t alignment):
        bytes(bytes),alloc_unit(alloc_unit),alignment(alignment){}

    uint64_t alloc_size(){
        uint64_t best_case = util::div_ceil(bytes, alloc_unit)*alloc_unit;
        uint64_t worst_case = best_case + alignment; // could be improved by subtracting cuda alignment guarantees (512B)
        return worst_case;
    }

    void* align(void* ptr){
        uint64_t orig = (uint64_t) ptr;
        // printf("The original ptr returned by malloc: %p, %lu\n",ptr,orig);
        // deal with this possibly misaligned ptr:
        if (alignment == 0 || orig % alignment == 0) return ptr;
        else return (void*) (orig + (alignment - (orig % alignment)));
    }    
    uint64_t bytes;
    uint64_t alloc_unit;
    uint64_t alignment;
};

/**
 * Abstract Base Class of all types of memory.
 */
struct MemBase{
    MemBase(const MemBase &obj) = delete;
    MemBase(MemBase&& other) noexcept{
        this->data = other.data;
        this->bytes = other.bytes;
        this->alloc_unit = other.alloc_unit;
        other.data = nullptr;
        other.bytes = 0;
        other.alloc_unit = 1;
    }
    MemBase():MemBase(0,1,0){}
    MemBase(uint64_t bytes, uint64_t alloc_unit, uint64_t alignment):bytes(bytes),alloc_unit(alloc_unit),alignment(alignment){}
    virtual ~MemBase(){};

    template <typename T>
    uint64_t size() const { return bytes/sizeof(T); }

    template <typename T>
    T* ptr() const { return reinterpret_cast<T*>(data); }

    uint64_t size_bytes() const { return bytes; }

    template <typename T>
    void resize_num(uint64_t num, uint64_t alloc_unit = 0, uint64_t alignment = 0, bool keep = false){
        if (alloc_unit != 0) this->alloc_unit = alloc_unit;
        this->alignment = alignment;

        resize(sizeof(T), num, keep);
    }

    MemLoc& memloc(){
        return loc;
    }

    virtual void set(int val, cudaStream_t stream = 0) = 0;

    friend std::ostream& operator<<(std::ostream &out, MemBase const& obj){
        out << "Memory(loc="<<obj.loc<<",bytes="<<obj.bytes<<")";
        return out;
    }

    /**
     * data pointers
     */
    void* data = (void*)0xFAFAFA;
    void* real_data = (void*)0xFEFEFE;
protected:
    /**
     * Re-allocation should be done in this function (to be overridden).
     */
    virtual void resize(uint64_t type_size, uint64_t num, bool keep = false) = 0;
    /**
     * allocated bytes
     */
    uint64_t bytes;         // the publicly visible bytes, not the bytes of <real_data>
    uint64_t alloc_unit;
    uint64_t alignment;
    /**
     * Memory Location info
     */
    MemLoc loc;
};

/**
 * DeviceMem manages memory on device.
 */
class DeviceMem : public MemBase{
public:
    DeviceMem (const DeviceMem &obj) = delete;
    DeviceMem(DeviceMem&& other) noexcept{
        this->data = other.data;
        this->real_data = other.real_data;
        this->bytes = other.bytes;
        this->alloc_unit = other.alloc_unit;
        this->alignment = other.alignment;
        other.data = nullptr;
        other.bytes = 0;
        other.alloc_unit = 1;
    }
    /**
     * Constructor without allocation.
     */
    DeviceMem() : MemBase(0,1,0) {
        loc = MemLoc{MemLoc::GPU, -1};
    }
    /**
     * Templated constructor to allocate for a number of T elements.
     */
    template <typename T>
    DeviceMem(Tag<T> tag, uint64_t num, uint64_t alloc_unit = 1, uint64_t alignment = 0) :
            MemBase(util::div_ceil(sizeof(T)*num,alloc_unit)*alloc_unit, alloc_unit, alignment) {

        loc = MemLoc{MemLoc::GPU, -1};
        if (bytes == 0) return;

        AllocHelper allochelp{sizeof(T)*num, alloc_unit, alignment};

        checkCudaErrors(cudaMalloc(&real_data, allochelp.alloc_size()));
        data = allochelp.align(real_data);
        checkCudaErrors(cudaGetDevice(&loc.info));
        DEVICE_ALLOCATED.fetch_add(bytes);
    }

    void set(int val, cudaStream_t stream = 0){
        if (bytes == 0) return;
        if (stream == 0) checkCudaErrors(cudaMemset(data, val, bytes));
        else checkCudaErrors(cudaMemsetAsync(data, val, bytes, stream));
    }

    /**
     * Frees allocated memory (if any).
     */
    ~DeviceMem() {
        if (bytes == 0) return;
        checkCudaErrors(cudaFree(real_data));
        real_data = nullptr;
        DEVICE_ALLOCATED.fetch_sub(bytes);
    }
protected:
    /**
     * Resizes the allocated memory and
     * copies the current data (if any) to the new memory.
     */
    void resize(uint64_t type_size, uint64_t num, bool keep = false) {
        uint64_t new_bytes = util::div_ceil(type_size*num,alloc_unit)*alloc_unit;

        AllocHelper allochelp{type_size*num, alloc_unit, alignment};

        void* old_ptr = data;
        void* old_real_ptr = real_data;
        if(keep){
            uint64_t cpy_size = new_bytes > bytes ? bytes : new_bytes;
            checkCudaErrors(cudaMalloc(&real_data, allochelp.alloc_size()));
            data = allochelp.align(real_data);
            if(bytes != 0){
                checkCudaErrors(cudaMemcpy(data, old_ptr, cpy_size, cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaFree(old_real_ptr));
            }
        }else {
            if(bytes != 0) checkCudaErrors(cudaFree(old_real_ptr));
            if(new_bytes != 0){
                checkCudaErrors(cudaMalloc(&real_data, allochelp.alloc_size()));
                data = allochelp.align(real_data);
                checkCudaErrors(cudaGetDevice(&loc.info));
            }
        }
        DEVICE_ALLOCATED.fetch_add(new_bytes - bytes);
        bytes = new_bytes;
    }
};

/**
 * HostMem allocates in host memory, but uses
 * cudaMallocHost to create pinned memory for faster data transfer.
 * This is also suitable for zero-copy access from the device, relying on Unified Virtual Addressing.
 */
class HostMem : public MemBase{
public:
    HostMem (const HostMem &obj) = delete;
    HostMem(HostMem&& other) noexcept{
        this->data = other.data;
        this->real_data = other.real_data;
        this->bytes = other.bytes;
        this->alloc_unit = other.alloc_unit;
        this->alignment = other.alignment;
        other.data = nullptr;
        other.bytes = 0;
        other.alloc_unit = 1;
    }
    /**
     * Constructor without allocation.
     */
    HostMem() : MemBase(0,1,0) {
        loc = MemLoc{MemLoc::CPU, -1};
    }
    /**
     * Templated constructor to allocate for a number of T elements.
     */
    template <typename T>
    HostMem(Tag<T> tag, uint64_t num, uint64_t alloc_unit = 1, uint64_t alignment = 0) :
            MemBase(util::div_ceil(sizeof(T)*num,alloc_unit)*alloc_unit, alloc_unit, alignment) {
        loc = MemLoc{MemLoc::CPU, -1};
        if (bytes == 0) return;

        AllocHelper allochelp{sizeof(T)*num, alloc_unit, alignment};

        checkCudaErrors(cudaMallocHost(&real_data, allochelp.alloc_size()));
        data = allochelp.align(real_data);
        get_mempolicy(&loc.info, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
        HOST_ALLOCATED.fetch_add(bytes);
    }

    void set(int val, cudaStream_t stream = 0){
        if (bytes == 0) return;
        memset(data, val, bytes);
    }

    /**
     * Frees allocated memory (if any).
     */
    ~HostMem() {
        if (bytes == 0) return;
        checkCudaErrors(cudaFreeHost(real_data));
        real_data = nullptr;
        HOST_ALLOCATED.fetch_sub(bytes);
    }
protected:
    /**
     * Resizes the allocated memory and
     * copies the current data (if any) to the new memory.
     */
    void resize(uint64_t type_size, uint64_t num, bool keep = false) {
        uint64_t new_bytes = util::div_ceil(type_size*num,alloc_unit)*alloc_unit;

        AllocHelper allochelp{type_size*num, alloc_unit, alignment};

        void* old_ptr = data;
        void* old_real_ptr = real_data;
        if(keep){
            uint64_t cpy_size = new_bytes > bytes ? bytes : new_bytes;
            checkCudaErrors(cudaMallocHost(&real_data, allochelp.alloc_size()));
            data = allochelp.align(real_data);
            if(bytes != 0){
                checkCudaErrors(cudaMemcpy(data, old_ptr, cpy_size, cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaFreeHost(old_real_ptr));
            }
        }else {
            if(bytes != 0) checkCudaErrors(cudaFreeHost(old_real_ptr));
            if(new_bytes != 0){
                checkCudaErrors(cudaMallocHost(&real_data, allochelp.alloc_size()));
                data = allochelp.align(real_data);
                get_mempolicy(&loc.info, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
            }
        }
        HOST_ALLOCATED.fetch_add(new_bytes - bytes);
        bytes = new_bytes;
    }
};

/**
 * Composite class for data stored in both host and device memory.
 * Host memory is pinned to speed up transfers.
 * Data transfer is possible host->device (sync_to_device),
 * and device->host (sync_to_host).
 */
class MirrorMem{
public:
    MirrorMem (const MirrorMem &obj) = delete;
    MirrorMem(MirrorMem&&) = default;
    MirrorMem() : dev(), hst() {}

    template <typename T>
    MirrorMem(Tag<T> tag, uint64_t num, uint64_t alloc_unit = 1, uint64_t alignment = 0) : dev(tag,num,alloc_unit,alignment), hst(tag,num,alloc_unit,alignment) {
    }

    DeviceMem dev;
    HostMem hst;

    /**
     * Copy from device to host memory.
     */
    void sync_to_host(cudaStream_t stream = 0) {
        if(stream == 0){
            checkCudaErrors(cudaMemcpy(hst.data, dev.data, size_bytes(),
                                       cudaMemcpyDeviceToHost));
        }else {
            checkCudaErrors(cudaMemcpyAsync(hst.data, dev.data, size_bytes(),
                                            cudaMemcpyDeviceToHost, stream));
        }
    }

    /**
     * Copy from host to device memory.
     */
    void sync_to_device(cudaStream_t stream = 0) {
        if(stream == 0){
            checkCudaErrors(cudaMemcpy(dev.data, hst.data, size_bytes(),
                                            cudaMemcpyHostToDevice));
        }else {
            checkCudaErrors(cudaMemcpyAsync(dev.data, hst.data, size_bytes(),
                                            cudaMemcpyHostToDevice, stream));
        }
    }

    uint64_t size_bytes() { return hst.size_bytes(); }

    template <typename T>
    uint64_t size() { return size_bytes()/sizeof(T); }

    template <typename T>
    void resize_num(uint64_t num, bool keep = false){
        dev.resize_num<T>(num,keep);
        hst.resize_num<T>(num,keep);
    }


    friend std::ostream& operator<<(std::ostream &out, MirrorMem & obj){
        out << "MirrorMem(bytes="<<obj.size_bytes()<<")";
        return out;
    }

};



/**
 * UnifiedMem allocates managed (unified) memory.
 */
class UnifiedMem : public MemBase{
public:
    UnifiedMem (const UnifiedMem &obj) = delete;
    UnifiedMem(UnifiedMem&& other) noexcept{
        this->data = other.data;
        this->real_data = other.real_data;
        this->bytes = other.bytes;
        this->alloc_unit = other.alloc_unit;
        this->alignment = other.alignment;
        other.data = nullptr;
        other.bytes = 0;
        other.alloc_unit = 1;
    }
    /**
     * Constructor without allocation.
     */
    UnifiedMem() : MemBase(0,1,0) {
        loc = MemLoc{MemLoc::CPU, -1};
    }
    /**
     * Templated constructor to allocate for a number of T elements.
     */
    template <typename T>
    UnifiedMem(Tag<T> tag, uint64_t num, uint64_t alloc_unit = 1, uint64_t alignment = 0) :
            MemBase(util::div_ceil(sizeof(T)*num,alloc_unit)*alloc_unit, alloc_unit, alignment) {
        loc = MemLoc{MemLoc::CPU, -1};
        if (bytes == 0) return;

        AllocHelper allochelp{sizeof(T)*num, alloc_unit, alignment};

        checkCudaErrors(cudaMallocManaged(&real_data, allochelp.alloc_size()));
        data = allochelp.align(real_data);
        get_mempolicy(&loc.info, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
        HOST_ALLOCATED.fetch_add(bytes);
    }

    void set(int val, cudaStream_t stream = 0){
        if (bytes == 0) return;
        // checkCudaErrors(cudaMemset(data, val, bytes)); // actually lets assume we want to do this in host memory
        memset(data,val,bytes);
    }

    /**
     * Frees allocated memory (if any).
     */
    ~UnifiedMem() {
        if (bytes == 0) return;
        checkCudaErrors(cudaFree(real_data));
        real_data = nullptr;
        HOST_ALLOCATED.fetch_sub(bytes);
    }
protected:
    /**
     * Resizes the allocated memory and
     * copies the current data (if any) to the new memory.
     */
    void resize(uint64_t type_size, uint64_t num, bool keep = false) {
        uint64_t new_bytes = util::div_ceil(type_size*num,alloc_unit)*alloc_unit;
        AllocHelper allochelp{type_size*num, alloc_unit, alignment};

        void* old_ptr = data;
        void* old_real_ptr = real_data;
        if(keep){
            uint64_t cpy_size = new_bytes > bytes ? bytes : new_bytes;
            checkCudaErrors(cudaMallocManaged(&real_data, allochelp.alloc_size()));
            data = allochelp.align(real_data);
            if(bytes != 0){
                checkCudaErrors(cudaMemcpy(data, old_ptr, cpy_size, cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaFree(old_real_ptr));
            }
        }else {
            if(bytes != 0) checkCudaErrors(cudaFree(old_real_ptr));
            if(new_bytes != 0){
                checkCudaErrors(cudaMallocManaged(&real_data, allochelp.alloc_size()));
                data = allochelp.align(real_data);
                get_mempolicy(&loc.info, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
            }
        }
        HOST_ALLOCATED.fetch_add(new_bytes - bytes);
        bytes = new_bytes;
    }
};

} // end of namespace