#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/fs.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>


#include "table.hpp"
#include "util.hpp"

namespace golap {

static void checkCuFileError(CUfileError_t errt){
    if(errt.err == CU_FILE_SUCCESS) return;

    std::cout << "CuFileError:" << errt.err << "=" << CUFILE_ERRSTR(errt.err) << "\n";
    std::exit(EXIT_FAILURE);
}

/**
 * The magic alignment that makes buffer registration work?
 */
constexpr uint64_t CUFILE_MAGIC_NUMBER = (1<<16);


/**
 * StorageManager for a block device, handling writes, reads, serialization from host, device.
 */
class StorageManager : public util::Singleton<StorageManager> {
public:
    void init(std::string dev_path, uint64_t start_offset){
        if (fd != -1) return;
        this->dev_path = dev_path;
        this->cur_offset = start_offset;

        checkCuFileError(cuFileDriverOpen());

        fd = open(dev_path.c_str(), O_RDWR | O_DIRECT);
        if(fd < 0){
            util::Log::get().error("File open failed");
            std::cout << std::strerror(errno) << "\n";
            std::exit(1);
        }
        struct stat st;
        if(ioctl(fd,BLKGETSIZE64,&size)!=0){
            if(fstat(fd, &st)!=0){
                util::Log::get().warn("IOCTL and FSTAT both failed, wrong path?");
                std::cout << std::strerror(errno) << "\n";
                std::exit(1);
            }
            size = st.st_size;
        }

        util::Log::get().debug(dev_path+" opened, size "+std::to_string(size)+" bytes.");


        memset((void *)&desc, 0, sizeof(CUfileDescr_t));
        memset((void *)&cfh, 0, sizeof(CUfileHandle_t));
        desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        desc.handle.fd = fd;

        checkCuFileError(cuFileHandleRegister(&cfh, &desc));
        checkCuFileError(cuFileDriverSetMaxPinnedMemSize(67108864)); // 64 gb in kb
    }

    ~StorageManager(){
        if (fd==-1) return;
        cuFileHandleDeregister(cfh);
        if (close(fd)){
            util::Log::get().warn("StorageManager close fd returned non-zero!");
        }
        fd = -1;

        // this triggers some cuda warnings, let it be done implicitly
        // checkCuFileError(cuFileDriverClose());
        util::Log::get().debug("File closed");
    }


    /**
     * Read @p bytes number of bytes from CUfileHandle_t plus @offset to @p dev_ptr.
     */
    bool dev_read_bytes(void* dev_ptr, uint64_t bytes, uint64_t offset){
        // If you want to register a buffer, cufile driver is kind of picky about the size.
        // For our driver version, multiples of (1<<18) bytes seem to work. Also see VarLoader
        // checkCuFileError(cuFileBufRegister(dev_ptr, bytes, 0));

        int64_t ret = cuFileRead(cfh, dev_ptr, bytes, offset, 0);

        // checkCuFileError(cuFileBufDeregister(dev_ptr));
        return ret == bytes;
    }

    /**
     * Write @p bytes number of bytes to CUfileHandle_t plus @offset from @p dev_ptr.
     */
    bool dev_write_bytes(void* dev_ptr, uint64_t bytes, uint64_t offset){
        // If you want to register a buffer, cufile driver is kind of picky about the size.
        // For our driver version, multiples of (1<<18) bytes seem to work. Also see VarLoader
        // checkCuFileError(cuFileBufRegister(dev_ptr, bytes, 0));

        int64_t ret = cuFileWrite(cfh, dev_ptr, bytes, offset, 0);

        // checkCuFileError(cuFileBufDeregister(dev_ptr));
        return ret == bytes;
    }

    /**
     * Read @p bytes number of bytes from file descriptor plus @offset to @p hst_ptr.
     */
    bool host_read_bytes(void* hst_ptr, uint64_t bytes, uint64_t offset){
        // int64_t read = pread64(fd, hst_ptr, bytes, offset);

        int64_t read = 0;
        uint64_t orig_bytes = bytes;
        uint64_t cur_transfer;
        do{
            cur_transfer = std::min(bytes, (uint64_t) 0x7fffe000);
            read += pread64(fd, (char*)hst_ptr+read, cur_transfer, offset+read);
            bytes -= cur_transfer;
        }while(bytes!=0 && read!=0);


        if(read != orig_bytes){
            std::cout << "Host read failed. Ret was: " << read << ", "<< std::strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    /**
     * Write @p bytes number of bytes to file descriptor plus @offset from @p hst_ptr.
     */
    bool host_write_bytes(void* hst_ptr, uint64_t bytes, uint64_t offset){
        // int64_t written = pwrite64(fd, hst_ptr, bytes, offset);

        int64_t written = 0;
        uint64_t orig_bytes = bytes;
        uint64_t cur_transfer;
        do{
            cur_transfer = std::min(bytes, (uint64_t) 0x7fffe000);
            written += pwrite64(fd, (char*)hst_ptr+written, cur_transfer, offset+written);
            bytes -= cur_transfer;
        }while(bytes!=0 && written!=0);

        if(written != orig_bytes){
            std::cout << "Host write failed. Ret was: " << written << ", "<< std::strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    /**
     * Convenience method (slower), calling either dev_read_bytes, or host_read_bytes
     */
    bool read_bytes(void* ptr, uint64_t bytes, uint64_t offset){
        cudaPointerAttributes attributes;
        checkCudaErrors(cudaPointerGetAttributes(&attributes,ptr));
        if (attributes.type == cudaMemoryTypeHost) return host_read_bytes(ptr,bytes,offset);
        else if (attributes.type == cudaMemoryTypeDevice) return dev_read_bytes(ptr,bytes,offset);
        else return false;
    }
    /**
     * Convenience method (slower), calling either dev_write_bytes, or host_write_bytes
     */
    bool write_bytes(void* ptr, uint64_t bytes, uint64_t offset){
        cudaPointerAttributes attributes;
        checkCudaErrors(cudaPointerGetAttributes(&attributes,ptr));
        if (attributes.type == cudaMemoryTypeHost) return host_write_bytes(ptr,bytes,offset);
        else if (attributes.type == cudaMemoryTypeDevice) return dev_write_bytes(ptr,bytes,offset);
        else return false;
    }


    std::string get_path(){
        return dev_path;
    }

    uint64_t get_size(){
        return size;
    }

    uint64_t get_offset(){
        return cur_offset;
    }
    void set_offset(uint64_t cur_offset){
        if (cur_offset >= size){
            util::Log::get().warn_fmt("Offset in storage is set passed size of the underlying blockdevice or file!");
        }
        this->cur_offset = cur_offset;
    }

    int fd = -1;
    CUfileHandle_t cfh;

private:
    CUfileDescr_t desc;
    uint64_t size;
    std::string dev_path;
    // this can be used to coordinate where the next block of data can be written.
    // for CPU-parallel write access, this should be handled differently though
    uint64_t cur_offset;

};


// class DiskCacheManager : public util::Singleton<DiskCacheManager> {
// public:
//     uint64_t store_offset = 0;
//     uint64_t total_size = StorageManager::get().size();
//     void add(std::string &colname, uint64_t num_tuples, golap::CompInfo &compinfo){
//     }
// };



} // end of namespace