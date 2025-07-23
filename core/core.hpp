#pragma once

#include <iostream>
#include <thread>

#include <nvToolsExtCudaRt.h>
#include "cuda_runtime.h"
#include "nvToolsExt.h"

#include "util.hpp"

namespace golap{

struct Parameter{
    // ######################################
    // ################ INPUT ###############
    // ######################################
    uint32_t cuda_device;   // no print
    uint32_t block_limit;   // no print
    std::string query;
    std::string dataflow;
    uint32_t scale_factor;
    std::string core_pin;
    uint32_t workers;
    uint64_t chunk_bytes;
    uint64_t nvchunk;
    int num_RLEs;           // no print
    int num_deltas;         // no print
    int use_bp;             // no print
    uint64_t store_offset;  // no print
    std::string comp_algo;
    bool verify;            // no print
    std::string sort_by;
    uint64_t max_gpu_um_memory;
    std::string pruning;
    uint64_t pruning_param;
    double pruning_p;
    uint64_t pruning_m;
    std::string col_filter_lo;
    std::string col_filter_hi;
    int64_t simulate_compute_us = -1;
    bool event_sync;
    std::vector<uint64_t> chunk_size_vec; // no print
    // ######################################
    // ############### OUTPUT ###############
    // ######################################
    uint64_t device_mem_total;
    uint64_t device_mem_used;
    uint64_t host_mem_used;
    uint64_t comp_bytes;
    uint64_t uncomp_bytes;
    uint64_t pruned_bytes;
    double population_ms=-1.f;
    double comp_ms=-1.f;
    double prune_ms=-1.f;
    double time_ms=-1.f;
    std::string debug_0{};
    std::string debug_1{};
    std::string debug_2{};
    std::string debug_3{};

    std::string to_pretty(){
        double bw = (1000.0 / (1<<30)) * ((double)uncomp_bytes/time_ms);
        double storage_bw = (1000.0 / (1<<30)) * ((double)comp_bytes/time_ms);
        double ratio = (comp_bytes/(double)uncomp_bytes);

        std::stringstream ss;
        ss << "Param(query="<<query<<", dataflow="<<dataflow<<", scale_factor="<< scale_factor<<", comp_algo="<<comp_algo<<", sort_by="<<sort_by;
        ss << ", pruning="<<pruning<<", comp_bytes="<<(double)comp_bytes/(1<<30)<<"GB, uncomp_bytes="<<(double)uncomp_bytes/(1<<30)<<"GB,\n\tpopulation_ms="<<population_ms;
        ss << ",comp_ms=" << comp_ms <<"ms, prune_ms=" <<prune_ms<< "ms, time_ms="<<time_ms<<"ms, bw="<<bw<<"GB/s, storage_bw="<<storage_bw<<"GB/s, comp_ratio="<<ratio<<")";
        return ss.str();
    }

    std::string to_csv(){
        std::stringstream ss;
        ss << query<<','<<dataflow<<','<<scale_factor<<','<<core_pin<<','<<workers<<','<<chunk_bytes<<',';
        ss << nvchunk<<','<<comp_algo<<','<<sort_by<<','<<max_gpu_um_memory<<','<<pruning <<','<<pruning_param<<','<<pruning_p<<','<<pruning_m;
        ss << ',' << col_filter_lo << ',' << col_filter_hi;
        ss << ',' << simulate_compute_us << ',' << event_sync << ',' << device_mem_total<<','<<device_mem_used<<','<<host_mem_used;
        ss << ',' <<comp_bytes <<','<<uncomp_bytes<<','<<pruned_bytes<<','<<population_ms<<','<<comp_ms<<','<<prune_ms<<','<<time_ms;
        ss << ',' << debug_0 << ',' << debug_1 <<',' << debug_2 <<',' << debug_3;

        return ss.str();
    }

    std::string repr(bool pretty){
        return pretty ? to_pretty() : to_csv();
    }

    static std::string csv_header(){return "query,dataflow,scale_factor,core_pin,workers,chunk_bytes,nvchunk,comp_algo,sort_by,max_gpu_um_memory,pruning,pruning_param,pruning_p,pruning_m,col_filter_lo,col_filter_hi,simulate_compute_us,event_sync,device_mem_total,device_mem_used,host_mem_used,comp_bytes,uncomp_bytes,pruned_bytes,population_ms,comp_ms,prune_ms,time_ms,debug_0,debug_1,debug_2,debug_3";};
};


struct Op {
    Op(const Op &obj) = delete;
    Op(Op&& other)noexcept{
        this->event = other.event;
        this->child = other.child;
        other.event = nullptr;
    }
    Op(){
        checkCudaErrors(cudaEventCreate(&event));
    }
    ~Op(){
        if (event == nullptr) return;
        cudaEventDestroy(event);
    }
    virtual void set_child(Op* op){child = op;}
    virtual bool step(cudaStream_t stream, cudaEvent_t parent_event) = 0;
    virtual void finish_step(cudaStream_t stream){ checkCudaErrors(cudaStreamSynchronize(stream));}
    virtual bool skip_step(cudaStream_t stream, cudaEvent_t parent_event) = 0;
    void describe(){
        std::cout << typeid(*this).name();
        if (child == nullptr) std::cout << "\n";
        else{
            std::cout << " <-- ";
            child->describe();
        }
    }
    uint64_t last_produced;
    cudaEvent_t event;
protected:
    Op* child = nullptr;
};


class CountPipe : public Op{
public:
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->step(stream,event)) return false;
        util::Log::get().debug_fmt("Step in CountPipe %lu",steps);
        steps ++;
        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child == nullptr || !child->skip_step(stream,event)) return false;
        util::Log::get().debug_fmt("SkipStep in CountPipe %lu",steps);
        steps ++;
        return true;
    }
    uint64_t steps = 0;
};

class Memset : public Op{
public:
    Memset(char* ptr, uint64_t bytes, int val):ptr(ptr),bytes(bytes),val(val){}
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child != nullptr && !child->step(stream,event)) return false;
        util::Log::get().debug_fmt("Step in Memset, setting %lu bytes at ptr %p to %d", bytes, ptr, val);

        if (is_dev_ptr) checkCudaErrors(cudaMemsetAsync(ptr, val, bytes, stream));
        else{
            checkCudaErrors(cudaStreamSynchronize(stream));
            util::Log::get().debug_fmt("Synchronizing in Memset, since ptr %p is host memory, are you sure this is what you wanted?",ptr);
            memset(ptr, bytes, val);
        }

        return true;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if(child != nullptr && !child->skip_step(stream,event)) return false;
        util::Log::get().debug_fmt("SkipStep in Memset");

        return true;
    }
private:
    char* ptr;
    uint64_t bytes;
    int val;
    bool is_dev_ptr;
};

class DoXTimes : public Op{
public:
    DoXTimes(uint64_t credit):credit(credit){}
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        util::Log::get().debug_fmt("Step in DoXTimes %lu",credit);
        return (credit--) != 0;
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        util::Log::get().debug_fmt("SkipStep in DoXTimes %lu",credit);
        return (credit--) != 0;
    }
    void set_child(Op *op){
        throw std::runtime_error("Can't set child on DoXTimes!");
    }
private:
    uint64_t credit;
};


struct CStream{
    CStream(const CStream &obj) = delete;
    CStream(CStream&& other){
        this->stream = other.stream;
        this->name = other.name;
        other.stream = nullptr;
    }
    CStream(std::string str):name(str){
        checkCudaErrors(cudaStreamCreate(&stream));
        nvtxNameCudaStreamA(stream, name.c_str());
    }
    ~CStream(){
        if (stream == nullptr) return;
        cudaStreamDestroy(stream);
    }
    cudaStream_t stream;
    std::string name;
};

struct CEvent{
    CEvent(const CEvent &obj) = delete;
    CEvent(CEvent&& other) noexcept {
        this->event = other.event;
        other.event = nullptr;
    }
    CEvent(){
        checkCudaErrors(cudaEventCreate(&event));
    }
    ~CEvent(){
        if (event == nullptr) return;
        cudaEventDestroy(event);
    }
    cudaEvent_t event;
};

struct Executor{
    Executor(Op *op, uint32_t cuda_device):op(op),cuda_device(cuda_device){
        checkCudaErrors(cudaEventCreate(&event));
    }
    ~Executor(){
        cudaEventDestroy(event);
    }
    void start(cudaStream_t stream){
        thread = std::thread{[&]{
            
            cudaSetDevice(cuda_device);
            while(op->step(stream,event));
        }};
    }
    void wait(){
        thread.join();
    }
private:
    Op *op;
    uint32_t cuda_device;
    cudaEvent_t event;
    std::thread thread;
};


class Collector : public Op{
public:
    Collector(const Collector &obj) = delete;
    Collector(Collector&&) = default;
    Collector():Op(){}
    /**
     * Steps each child in its own stream.
              ┌───┐    ┌───┐
       Stream1│Lo ├────►Dec├─────┐
              └───┘    └───┘     │
                                 │
                                 │
                               ┌─▼───────┐    ┌──────┐
                        Stream3│Collector├────►Query │
                               └─▲───────┘    └──────┘
                                 │
                                 │
              ┌───┐    ┌───┐     │
       Stream2│Lo ├────►Dec├─────┘
              └───┘    └───┘
     * The childstreams need to wait for the parent event, and the main stream needs to wait on the children.
     */
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        util::Log::get().debug_fmt("Step in Collector, %lu children", children.size());
        bool res = false;
        Op* op;
        cudaStream_t opstream;
        for (auto &tup:children){
            std::tie(op,opstream) = tup;
            checkCudaErrors(cudaStreamWaitEvent(opstream,parent_event,0));
            res |= op->step(opstream,parent_event);
            checkCudaErrors(cudaStreamWaitEvent(stream,op->event,0)); // this stream will wait on the opevent
        }
        checkCudaErrors(cudaEventRecord(event,stream));
        return res;
    }
    void add_child(Op *op, cudaStream_t opstream){
        children.emplace_back(op,opstream);
    }
    void set_child(Op *op){
        throw std::runtime_error("Use add_child(Op*,cudaStream_t)!");
    }
    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        util::Log::get().debug_fmt("SkipStep in Collector, %lu children", children.size());
        bool res = false;
        Op* op;
        cudaStream_t opstream;
        for (auto &tup:children){
            std::tie(op,opstream) = tup;
            res |= op->skip_step(opstream,event);
        }
        return res;
    }
private:
    std::vector<std::tuple<Op*,cudaStream_t>> children;
};

class Decollector : public Op{
public:
    Decollector(const Decollector &obj) = delete;
    Decollector(Decollector&&) = default;
    Decollector(uint64_t num_children):Op(),num_children(num_children){}
    /**
     * Opposite of a collector: called multiple times, only calls child once.

         ┌───────┐        ┌──────┐
         │       │        │ D    │
         │ B     │        │ e    ◄─────
         │ a     │        │ c    │
         │ t     │        │ o    │
         │ c     │        │ l    ◄─────
         │ h     │◄───────┤ l    │
         │ IO    │        │ e    │
         │       │        │ c    ◄─────
         │       │        │ t    │
         │       │        │ o    │
         │       │        │ r    │
         │       │        │      │
         │       │        │      │
         │       │        │      │
         └───────┘        └──────┘
     */
    bool step(cudaStream_t stream, cudaEvent_t parent_event){
        if (called % num_children == 0){
            if (child!=nullptr && !child->step(stream,event)) return false;
        }

        util::Log::get().debug_fmt("Step in Decollector, %lu children, called %lu times already", num_children, called);

        called += 1;

        checkCudaErrors(cudaEventRecord(event,stream));
        return true;
    }

    bool skip_step(cudaStream_t stream, cudaEvent_t parent_event){
        if (called % num_children == 0){
            if (child!=nullptr && !child->skip_step(stream,event)) return false;
        }

        util::Log::get().debug_fmt("SkipStep in Decollector, %lu children, called %lu times already", num_children, called);

        called += 1;

        checkCudaErrors(cudaEventRecord(event,stream));
        return true;
    }
private:
    uint64_t called = 0;
    uint64_t num_children;
};


} // end of namespace
