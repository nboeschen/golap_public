#pragma once
#include <atomic>
#include "nvToolsExt.h"

#include "query_common.cuh"
#include "../ssb.hpp"
#include "03c.cuh"

#include "concurrentqueue.h"
#include "util.hpp"
#include "mem.hpp"
#include "host_structs.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*
select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and c_region = 'ASIA'
and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and (c_city='UNITED KI1' or c_city='UNITED KI5')
and (s_city='UNITED KI1' or s_city='UNITED KI5')
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_key
and lo_suppkey = s_key
and lo_orderdate = d_key
and (c_city='UNITED KI1' or c_city='UNITED KI5')
and (s_city='UNITED KI1' or s_city='UNITED KI5')
and d_yearmonth = 'Dec1997'
group by c_city, s_city, d_year
order by d_year asc, revenue desc;

3 --> assume all dimension tables / HT fit in GPU memory
3a --> Customer decompression + filter on GPU, HT on CPU
3b --> Customer decompression on GPU, filter + HT on CPU
3c --> Customer decompression + filter + CPU_HT_INSERT on GPU
3d --> Customer decompression + filter + HT on CPU

*/

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using CUSTKEY_TYPE = decltype(Lineorder::lo_custkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);

using CKEY_TYPE = decltype(Customer::c_key);
using CCITY_TYPE = decltype(Customer::c_city);
using CNATION_TYPE = decltype(Customer::c_nation);
using CREGION_TYPE = decltype(Customer::c_region);


template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP, typename BLOCK_ENV, typename SUPP_PRED, typename DATE_PRED>
bool query3c(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel, Q3SUPP_COLUMNS &supplier_dev,
                          Q3DATE_COLUMNS &date_dev, SUPP_PRED supp_pred, DATE_PRED date_pred){
    cudaSetDevice(ssbobj.var.cuda_device);


    /**
     * Generally needed:
     * Lineorder: lo_orderdate, lo_custkey, lo_suppkey, lo_revenue
     * Supplier: s_key, s_nation, s_region, s_city
     * Date: d_key, d_year
     * Customer: c_key, c_nation, c_region, c_city
     */
    uint64_t lo_tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(ORDERDATE_TYPE),sizeof(CUSTKEY_TYPE),
                                                           sizeof(SUPPKEY_TYPE),sizeof(REVENUE_TYPE)});
    uint64_t cust_tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(CKEY_TYPE),sizeof(CCITY_TYPE),sizeof(CNATION_TYPE),
                                                           sizeof(CREGION_TYPE)});

    util::Log::get().info_fmt("lo_tuples_per_chunk=%lu", lo_tuples_per_chunk);
    util::Log::get().info_fmt("cust_tuples_per_chunk=%lu", cust_tuples_per_chunk);

    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = 0;

    /**
    * - Write the chunk-compressed lineorder table to disk. 
    */

    ssbobj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,uint64_t> columns{
        {"lo_orderdate",lo_tuples_per_chunk},     // Lineorder::ORDERDATE,
        {"lo_custkey",lo_tuples_per_chunk},       // Lineorder::CUSTKEY,
        {"lo_suppkey",lo_tuples_per_chunk},       // Lineorder::SUPPKEY,
        {"lo_revenue",lo_tuples_per_chunk},       // Lineorder::REVENUE
        {"c_key",cust_tuples_per_chunk},             // Customer::KEY,
        {"c_city",cust_tuples_per_chunk},            // Customer::CITY,
        {"c_nation",cust_tuples_per_chunk},          // Customer::NATION,
        {"c_region",cust_tuples_per_chunk},          // Customer::REGION,
    };

    std::unordered_map<std::string, golap::CompInfo> compinfos;

    auto compress_columns_fun = [&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        auto entry = columns.find(a_col.attr_name);
        if (entry == columns.end()) return;
        auto algo = ssbobj.var.comp_algo;
        if(ssbobj.var.comp_algo == "BEST_BW_COMP"){
            if (BEST_BW_COMP.find(a_col.attr_name) == BEST_BW_COMP.end()) algo = "Gdeflate";
            else algo = BEST_BW_COMP.at(a_col.attr_name);
        }else if(ssbobj.var.comp_algo == "BEST_RATIO_COMP"){
            if (BEST_RATIO_COMP.find(a_col.attr_name) == BEST_RATIO_COMP.end()) algo = "Gdeflate";
            else algo = BEST_RATIO_COMP.at(a_col.attr_name);
        }

        compinfos[a_col.attr_name] = golap::CompInfo{entry->second*a_col.value_size,
                                                     num_tuples*a_col.value_size,
                                                     algo, ssbobj.var.nvchunk};
        util::Log::get().debug_fmt("Compressing col %lu=>%s to disk, algo=%s",col_idx,a_col.attr_name.c_str(),algo.c_str());
        if constexpr (std::is_same_v<BLOCK_ENV,golap::DecompressEnv>){
            ssbobj.var.comp_ms += golap::prepare_compressed_device(a_col, num_tuples, compinfos[a_col.attr_name]);
        }else{
            ssbobj.var.comp_ms += golap::prepare_uncompressed(a_col, num_tuples, compinfos[a_col.attr_name]);
        }
        ssbobj.var.comp_bytes += compinfos[a_col.attr_name].get_comp_bytes();
        ssbobj.var.uncomp_bytes += compinfos[a_col.attr_name].uncomp_bytes;
    };

    ssbobj.tables.lineorder.apply(compress_columns_fun);
    ssbobj.tables.customer.apply(compress_columns_fun);


    uint64_t lo_column_bytes = 0,cust_column_bytes=0;
    for (auto& comp_info : compinfos){
        if (comp_info.first.find("lo_")==0) lo_column_bytes += comp_info.second.get_comp_bytes();
        else if (comp_info.first.find("c_")==0) cust_column_bytes += comp_info.second.get_comp_bytes();
    }
    util::Log::get().info_fmt("Lineorder columns bytes: %lu\t(%lu tuples)",lo_column_bytes,ssbobj.tables.lineorder.num_tuples);
    util::Log::get().info_fmt("Customer  columns bytes: %lu\t(%lu tuples)",cust_column_bytes,ssbobj.tables.customer.num_tuples);
    util::Log::get().info_fmt("Supplier  columns bytes: %lu\t(%lu tuples)",supplier_dev.size_bytes(), supplier_dev.num_tuples);
    util::Log::get().info_fmt("Date      columns bytes: %lu\t\t(%lu tuples)",date_dev.size_bytes(), date_dev.num_tuples);


    // create hashtable of cpu resident join
    golap::HostMem customer_host(golap::Tag<Q3Customer>{}, ssbobj.tables.customer.num_tuples);
    std::atomic_uint64_t customer_host_counter{0};

    golap::DeviceMem filled{golap::Tag<uint64_t>{}, 1};
    filled.set(0);
    golap::UnifiedMem RIDChains{golap::Tag<uint64_t>{}, ssbobj.tables.customer.num_tuples};
    golap::UnifiedMem lastRIDs{golap::Tag<uint64_t>{}, ssbobj.tables.customer.num_tuples};
    checkCudaErrors(cudaMemPrefetchAsync(RIDChains.data,RIDChains.size_bytes(),cudaCpuDeviceId));
    checkCudaErrors(cudaMemPrefetchAsync(lastRIDs.data,lastRIDs.size_bytes(),cudaCpuDeviceId));
    // checkCudaErrors(cudaMemAdvise(RIDChains.data,RIDChains.size_bytes(),cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId));
    // checkCudaErrors(cudaMemAdvise(lastRIDs.data,lastRIDs.size_bytes(),cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId));
    RIDChains.set(0); lastRIDs.set(0);
    golap::HashMap customer_hj(ssbobj.tables.customer.num_tuples, customer_host.ptr<Q3Customer>(),
                               RIDChains.ptr<uint64_t>(),lastRIDs.ptr<uint64_t>(), filled.ptr<uint64_t>());


    // prepare hashmaps of the two normal joins
    golap::HashMap supplier_hj(supplier_dev.num_tuples,supplier_dev.col<0>().data());
    golap::HashMap date_hj(date_dev.num_tuples,date_dev.col<0>().data());

    uint64_t num_groups = 2000;
    // golap::MirrorMem groups(golap::Tag<GROUP>{}, num_groups);
    // golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<GROUP>(), aggs.dev.ptr<uint64_t>());
    golap::HostMem groups(golap::Tag<GROUP>{}, num_groups);
    auto aggs = new std::atomic<uint64_t>[num_groups]{};
    golap::HostHashAggregate hash_agg(num_groups, groups.ptr<GROUP>(), aggs);

    golap::MirrorMem debug_aggs(golap::Tag<double>{},5);
    checkCudaErrors(cudaMemset(debug_aggs.dev.ptr<uint8_t>(),0,debug_aggs.dev.size_bytes()));

    struct LO_BUFFER{
        LO_BUFFER(const LO_BUFFER &obj) = delete;
        LO_BUFFER(LO_BUFFER&&) = default;
        LO_BUFFER(uint64_t lo_tuples_per_chunk):
                joined_buffer(golap::Tag<LO_GROUP>{}, lo_tuples_per_chunk),
                joined_counter(golap::Tag<uint64_t>{}, 1){}

        // buffer and counter
        golap::MirrorMem joined_buffer;
        golap::MirrorMem joined_counter;
    };
    struct CUST_BUFFER{
        CUST_BUFFER(const CUST_BUFFER &obj) = delete;
        CUST_BUFFER(CUST_BUFFER&&) = default;
        CUST_BUFFER(uint64_t cust_tuples_per_chunk):
                customer_buffer(golap::Tag<Q3Customer>{}, cust_tuples_per_chunk),
                customer_counter(golap::Tag<uint64_t>{}, 1){}

        // necessary for co-execution: buffer and counter
        golap::DeviceMem customer_buffer;
        golap::MirrorMem customer_counter;
    };

    std::vector<golap::TableLoader<BLOCK_ENV>> lo_envs;
    std::vector<LO_BUFFER> lo_add_envs;
    lo_add_envs.reserve(ssbobj.var.workers);
    lo_envs.reserve(ssbobj.var.workers);

    util::SliceSeq workslice(compinfos["lo_orderdate"].blocks.size(), ssbobj.var.workers);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_orderdate"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        // lo_envs.emplace_back(startblock, endblock, compinfos["lo_orderdate"], compinfos["lo_custkey"],
        //                      compinfos["lo_suppkey"], compinfos["lo_revenue"], lo_tuples_per_chunk);
        lo_envs.emplace_back(4);
        lo_add_envs.emplace_back(lo_tuples_per_chunk);
        lo_envs[pipeline_idx].add("lo_orderdate", all_blocks_idxs, startblock, endblock, compinfos["lo_orderdate"], nvcomp::TypeOf<ORDERDATE_TYPE>());
        lo_envs[pipeline_idx].add("lo_custkey", all_blocks_idxs, startblock, endblock, compinfos["lo_custkey"], nvcomp::TypeOf<CUSTKEY_TYPE>());
        lo_envs[pipeline_idx].add("lo_suppkey", all_blocks_idxs, startblock, endblock, compinfos["lo_suppkey"], nvcomp::TypeOf<SUPPKEY_TYPE>());
        lo_envs[pipeline_idx].add("lo_revenue", all_blocks_idxs, startblock, endblock, compinfos["lo_revenue"], nvcomp::TypeOf<REVENUE_TYPE>());
    }


    std::vector<golap::TableLoader<BLOCK_ENV>> cust_envs;
    std::vector<CUST_BUFFER> cust_add_envs;
    cust_envs.reserve(ssbobj.var.extra_workers);
    cust_add_envs.reserve(ssbobj.var.extra_workers);


    util::SliceSeq cworkslice(compinfos["c_nation"].blocks.size(), ssbobj.var.extra_workers);
    all_blocks_idxs.resize(compinfos["c_nation"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.extra_workers; ++pipeline_idx){
        // prepare environment for each thread
        cworkslice.get(startblock,endblock);

        // cust_envs.emplace_back(startblock, endblock, compinfos["c_key"], compinfos["c_city"],
        //                            compinfos["c_nation"], compinfos["c_region"], cust_tuples_per_chunk);
        cust_envs.emplace_back(4);
        cust_add_envs.emplace_back(cust_tuples_per_chunk);
        cust_envs[pipeline_idx].add("c_key", all_blocks_idxs, startblock, endblock, compinfos["c_key"], nvcomp::TypeOf<CKEY_TYPE>());
        cust_envs[pipeline_idx].add("c_city", all_blocks_idxs, startblock, endblock, compinfos["c_city"], nvcomp::TypeOf<uint8_t>());
        cust_envs[pipeline_idx].add("c_nation", all_blocks_idxs, startblock, endblock, compinfos["c_nation"], nvcomp::TypeOf<uint8_t>());
        cust_envs[pipeline_idx].add("c_region", all_blocks_idxs, startblock, endblock, compinfos["c_region"], nvcomp::TypeOf<uint8_t>());
    }

    uint64_t freemem,totalmem;
    checkCudaErrors(cudaMemGetInfo(&freemem,&totalmem));
    golap::DeviceMem dummy_mem;
    if(ssbobj.var.max_gpu_um_memory != 0 && freemem > ssbobj.var.max_gpu_um_memory){
        dummy_mem.resize_num<char>(freemem-ssbobj.var.max_gpu_um_memory);
        dummy_mem.set(0);
    }
    util::Log::get().info_fmt("Allocated %.2f GB as dummy memory, leaving %.2f GB", (1.0*dummy_mem.size_bytes())/(1<<30),
                              (1.0*ssbobj.var.max_gpu_um_memory)/(1<<30));

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    std::vector<std::thread> extra_p_threads;
    extra_p_threads.reserve(ssbobj.var.extra_workers);


    // misuse some streams ...
    auto& cust_build_stream = lo_envs[0].blockenvs.at("lo_custkey").gstream.stream;
    auto& supp_build_stream = lo_envs[1%ssbobj.var.workers].blockenvs.at("lo_suppkey").gstream.stream;
    auto& date_build_stream = lo_envs[2%ssbobj.var.workers].blockenvs.at("lo_orderdate").gstream.stream;

    /**
     * Start the timer
     */
    util::Timer timer;

    ssbobj.tables.date.col<Date::KEY>().transfer(date_dev.col<0>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    ssbobj.tables.date.col<Date::YEAR>().transfer(date_dev.col<1>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    ssbobj.tables.date.col<Date::YEARMONTH>().transfer(date_dev.col<2>().data(), ssbobj.tables.date.num_tuples, date_build_stream);

    ssbobj.tables.supplier.col<Supplier::KEY>().transfer(supplier_dev.col<0>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::NATION>().transfer(supplier_dev.col<1>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::REGION>().transfer(supplier_dev.col<2>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::CITY>().transfer(supplier_dev.col<3>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);

    golap::hash_map_build_pred_other<<<util::div_ceil(supplier_dev.num_tuples,512),512,0,supp_build_stream>>>(
                                            supplier_hj,supplier_dev.num_tuples,
                                            golap::DirectKey<decltype(Supplier::s_key)>(), supp_pred);

    golap::hash_map_build_pred_other<<<util::div_ceil(date_dev.num_tuples,512),512,0,date_build_stream>>>(
                                            date_hj,date_dev.num_tuples,
                                            golap::DirectKey<decltype(Date::d_key)>(), date_pred);

    // checkCudaErrors(cudaStreamSynchronize(cust_build_stream));
    checkCudaErrors(cudaStreamSynchronize(supp_build_stream));
    checkCudaErrors(cudaStreamSynchronize(date_build_stream));

    util::Timer another_timer;
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.extra_workers; ++pipeline_idx){
        extra_p_threads.emplace_back([&,pipeline_idx]{
            cudaSetDevice(ssbobj.var.cuda_device);
            auto& env = cust_envs[pipeline_idx];
            auto& add_env = cust_add_envs[pipeline_idx];


            // for query3, supplier and customer pred is the same, so just use that information
            auto cust_pred = [&env]{
                if constexpr(std::is_same_v<decltype(supp_pred.pred), RegionAsiaPred>){
                    return golap::PredInfo<CREGION_TYPE,RegionAsiaPred>{env.blockenvs.at("c_region").decomp_buf.template ptr<CREGION_TYPE>(),
                                                                                RegionAsiaPred()};
                }else if constexpr(std::is_same_v<decltype(supp_pred.pred), NationUSPred>) {
                    return golap::PredInfo<CNATION_TYPE,NationUSPred>{env.blockenvs.at("c_nation").decomp_buf.template ptr<CNATION_TYPE>(),
                                                                                 NationUSPred()};
                }else if constexpr(std::is_same_v<decltype(supp_pred.pred), CityKIPred>) {
                    return golap::PredInfo<CCITY_TYPE,CityKIPred>{env.blockenvs.at("c_city").decomp_buf.template ptr<CCITY_TYPE>(),
                                                                                 CityKIPred()};
                }else if constexpr(std::is_same_v<decltype(supp_pred.pred), RegionAPred>) {
                    return golap::PredInfo<CREGION_TYPE,RegionAPred>{env.blockenvs.at("c_region").decomp_buf.template ptr<CREGION_TYPE>(),
                                                                                RegionAPred()};
                }
                // https://forums.developer.nvidia.com/t/nvc-22-1-nonsensical-warning-about-missing-return-statement/202358/7
                __builtin_unreachable();
            }();

            uint64_t round = 0;
            uint32_t num_blocks;
            uint64_t tuples_this_round,cust_tuples,cust_host_insert=0;
            while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
                cudaMemsetAsync(add_env.customer_counter.dev.template ptr<uint64_t>(), 0, sizeof(uint64_t), env.rootstream.stream);
                
                tuples_this_round = env.blockenvs.at("c_key").myblocks[round].tuples;
                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));

                pipeline_customer_q3c_prerun<<<num_blocks,512,0,env.rootstream.stream>>>(
                                                                            // customer_hj,
                                                                                  env.blockenvs.at("c_key").decomp_buf.template ptr<CKEY_TYPE>(),
                                                                                  env.blockenvs.at("c_city").decomp_buf.template ptr<CCITY_TYPE>(),
                                                                                  env.blockenvs.at("c_nation").decomp_buf.template ptr<CNATION_TYPE>(),
                                                                                  env.blockenvs.at("c_region").decomp_buf.template ptr<CREGION_TYPE>(),
                                                                                  cust_pred,
                                                                                  add_env.customer_buffer.template ptr<Q3Customer>(),
                                                                                  add_env.customer_counter.dev.template ptr<uint64_t>(),
                                                                                  tuples_this_round
                                                                      );

                add_env.customer_counter.sync_to_host(env.rootstream.stream);
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
                cust_tuples = add_env.customer_counter.hst.template ptr<uint64_t>()[0];

                // atomically add to customer host counter
                cust_host_insert = customer_host_counter.fetch_add(cust_tuples,std::memory_order_relaxed);

                add_env.customer_counter.hst.template ptr<uint64_t>()[0] = cust_host_insert;
                add_env.customer_counter.sync_to_device(env.rootstream.stream);

                customer_kernel<<<num_blocks,512,0,env.rootstream.stream>>>(    customer_hj,
                                                                                  env.blockenvs.at("c_key").decomp_buf.template ptr<CKEY_TYPE>(),
                                                                                  env.blockenvs.at("c_city").decomp_buf.template ptr<CCITY_TYPE>(),
                                                                                  env.blockenvs.at("c_nation").decomp_buf.template ptr<CNATION_TYPE>(),
                                                                                  env.blockenvs.at("c_region").decomp_buf.template ptr<CREGION_TYPE>(),
                                                                                  cust_pred,
                                                                                  add_env.customer_buffer.template ptr<Q3Customer>(),
                                                                                  add_env.customer_counter.dev.template ptr<uint64_t>(),
                                                                                  cust_host_insert,
                                                                                  tuples_this_round
                                                                      );
                

                // util::Log::get().info_fmt("pipeline[%lu]: Tuple this round: %lu, matching #: %lu, cust_host_insert= %lu", pipeline_idx, tuples_this_round, cust_tuples, cust_host_insert);
                checkCudaErrors(cudaMemcpyAsync(customer_host.ptr<Q3Customer>() + cust_host_insert,
                                                add_env.customer_buffer.data,
                                                cust_tuples * sizeof(Q3Customer),
                                                cudaMemcpyDefault, env.rootstream.stream));
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream)); // not necessary?

                checkCudaErrors(cudaMemPrefetchAsync(customer_hj.RIDChains+cust_host_insert,cust_tuples*sizeof(uint64_t),
                                                     cudaCpuDeviceId,env.rootstream.stream));


                round += 1;
            } // end of while
            checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

        });
    }
    for(auto &thread: extra_p_threads) thread.join();
    // checkCudaErrors(cudaMemPrefetchAsync(RIDChains.data,RIDChains.size_bytes(),cudaCpuDeviceId));
    // checkCudaErrors(cudaMemPrefetchAsync(lastRIDs.data,lastRIDs.size_bytes(),cudaCpuDeviceId));
    debug_aggs.hst.ptr<double>()[0] = another_timer.elapsed();


    another_timer.reset();
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx]{
            
            cudaSetDevice(ssbobj.var.cuda_device);
            auto& env = lo_envs[pipeline_idx];
            auto& add_env = lo_add_envs[pipeline_idx];

            uint64_t tuples_this_round,joined_tuples,customer_match;
            uint64_t round = 0;
            uint32_t num_blocks;
            while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
                cudaMemsetAsync(add_env.joined_counter.dev.template ptr<uint64_t>(), 0, sizeof(uint64_t), env.rootstream.stream);

                tuples_this_round = env.blockenvs.at("lo_orderdate").myblocks[round].tuples;

                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                kernel<<<num_blocks,512,0,env.rootstream.stream>>>( 
                                                                   // customer_hj,
                                                                    supplier_hj,
                                                                      date_hj,
                                                                      // hash_agg,
                                                                      env.blockenvs.at("lo_custkey").decomp_buf.template ptr<CUSTKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_suppkey").decomp_buf.template ptr<SUPPKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_orderdate").decomp_buf.template ptr<ORDERDATE_TYPE>(),
                                                                      env.blockenvs.at("lo_revenue").decomp_buf.template ptr<REVENUE_TYPE>(),
                                                                      // customer_host.ptr<Q3Customer>(),
                                                                      supplier_dev.col<1>().data(), // supplier nation
                                                                      supplier_dev.col<3>().data(), // supplier city
                                                                      date_dev.col<1>().data(), // date year
                                                                      SumAgg(),
                                                                      add_env.joined_buffer.dev.template ptr<LO_GROUP>(),
                                                                      add_env.joined_counter.dev.template ptr<uint64_t>(),
                                                                      tuples_this_round
                                                                      ,&debug_aggs.dev.ptr<double>()[0],
                                                                      &debug_aggs.dev.ptr<double>()[1]
                                                                      );

                // find out how many lineorders qualified
                add_env.joined_counter.sync_to_host(env.rootstream.stream);
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
                joined_tuples = add_env.joined_counter.hst.template ptr<uint64_t>()[0];
                checkCudaErrors(cudaMemcpyAsync(add_env.joined_buffer.hst.data,add_env.joined_buffer.dev.data,
                                                joined_tuples * sizeof(LO_GROUP),
                                                cudaMemcpyDefault, env.rootstream.stream));
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

                // util::Log::get().info_fmt("Pipeline/Thread[%lu]: Qualified %lu / %lu lineorder tuples (%.2f %%, %lu bytes transferred)!",pipeline_idx,
                //                           joined_tuples,tuples_this_round,100.0*joined_tuples/tuples_this_round, joined_tuples * sizeof(LO_GROUP));
                // util::Timer probe_timer,agg_timer;
                // double probe_ms_total=0.0,agg_ms_total=0.0;
                for (uint64_t tid = 0; tid<joined_tuples;++tid){
                     // take lineorder tuple from buffer, probe customer ht, insert into hash_agg
                    auto& the_lo = add_env.joined_buffer.hst.template ptr<LO_GROUP>()[tid];
                    // probe_timer.reset();
                    customer_match = customer_hj.probe(the_lo.lo_custkey, Q3CustomerKey());

                    if (customer_match == (uint64_t) -1) continue;
                    // only add for qualified tuples?
                    // probe_ms_total += probe_timer.elapsed();

                    // agg_timer.reset();
                    if constexpr(std::is_same_v<GROUP,Q3NATION_GROUP>){
                        hash_agg.add(Q3NATION_GROUP{customer_host.ptr<Q3Customer>()[customer_match].c_nation,
                                                the_lo.s_nation,
                                                the_lo.d_year},
                                                the_lo.lo_revenue,
                                        HostSum());

                    }else if constexpr(std::is_same_v<GROUP,Q3CITY_GROUP>){
                        hash_agg.add(Q3CITY_GROUP{customer_host.ptr<Q3Customer>()[customer_match].c_city,
                                                the_lo.s_city,
                                                the_lo.d_year},
                                                the_lo.lo_revenue,
                                        HostSum());
                    }
                    // agg_ms_total += agg_timer.elapsed();
                }

                // util::Log::get().info_fmt("Pipeline/Thread[%lu]: probe_ms_total=%.2f,agg_ms_total=%.2f!",pipeline_idx,
                //                           probe_ms_total,agg_ms_total);

                round += 1;

            } // end of while
            checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
        });
    }

    for(auto &thread: threads) thread.join();


    checkCudaErrors(cudaStreamSynchronize(lo_envs[0].rootstream.stream));
    

    ssbobj.var.time_ms = timer.elapsed();
    debug_aggs.hst.ptr<double>()[1] = another_timer.elapsed();
    /**
     * Stopped timer
     */

    // debug_aggs.sync_to_host();

    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if ( hash_agg.wrote_group[i] == 0) continue;
        num_actual_groups += 1;

        auto& group = groups.ptr<GROUP>()[i];
        complete_sum += aggs[i];
        // std::cout << group <<"," << aggs[i] <<"\n";
    }

    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();

    ssbobj.var.debug_0 = std::to_string(debug_aggs.hst.ptr<double>()[0]);
    ssbobj.var.debug_1 = std::to_string(debug_aggs.hst.ptr<double>()[1]);

    // util::Log::get().info_fmt("Added to hash agg         =%lu",hash_agg.added.load());
    util::Log::get().info_fmt("CUST vs LO timer          =CUST %.2f, LO %.2f",debug_aggs.hst.ptr<double>()[0],debug_aggs.hst.ptr<double>()[1]);
    util::Log::get().info_fmt("CPU hashtable (/wo Agg)   =%.2fMB",1.0f*(RIDChains.size_bytes()+lastRIDs.size_bytes())/(1<<20));
    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);
    return true;
}


template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP>
bool query3_1c_gen(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_region),RegionAsiaPred> supp_pred{supplier_dev->col<2>().data(), RegionAsiaPred()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data(), DATE_9297()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::LoadEnv,
                            golap::PredInfo<decltype(Supplier::s_region),RegionAsiaPred>,
                            golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::DecompressEnv,
                    golap::PredInfo<decltype(Supplier::s_region),RegionAsiaPred>,
                    golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP>
bool query3_2c_gen(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_nation),NationUSPred> supp_pred{supplier_dev->col<1>().data(), NationUSPred()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data(), DATE_9297()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::LoadEnv,
                            golap::PredInfo<decltype(Supplier::s_nation),NationUSPred>,
                            golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::DecompressEnv,
                    golap::PredInfo<decltype(Supplier::s_nation),NationUSPred>,
                    golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP>
bool query3_3c_gen(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_city),CityKIPred> supp_pred{supplier_dev->col<3>().data(), CityKIPred()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data(), DATE_9297()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::LoadEnv,
                            golap::PredInfo<decltype(Supplier::s_city),CityKIPred>,
                            golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::DecompressEnv,
                    golap::PredInfo<decltype(Supplier::s_city),CityKIPred>,
                    golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP>
bool query3_4c_gen(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_city),CityKIPred> supp_pred{supplier_dev->col<3>().data(), CityKIPred()};
    golap::PredInfo<decltype(Date::d_yearmonth),YEARMONTHDec1997> date_pred{date_dev->col<2>().data(), YEARMONTHDec1997()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::LoadEnv,
                            golap::PredInfo<decltype(Supplier::s_city),CityKIPred>,
                            golap::PredInfo<decltype(Date::d_yearmonth),YEARMONTHDec1997>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::DecompressEnv,
                    golap::PredInfo<decltype(Supplier::s_city),CityKIPred>,
                    golap::PredInfo<decltype(Date::d_yearmonth),YEARMONTHDec1997>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}


template <typename KERNEL, typename CUSTOMER_KERNEL, typename LO_GROUP, typename GROUP>
bool query3_5c_gen(SSBColLayout &ssbobj, KERNEL kernel, CUSTOMER_KERNEL customer_kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_region),RegionAPred> supp_pred{supplier_dev->col<2>().data(), RegionAPred()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data(), DATE_9297()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::LoadEnv,
                            golap::PredInfo<decltype(Supplier::s_region),RegionAPred>,
                            golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3c<KERNEL,CUSTOMER_KERNEL,LO_GROUP,GROUP,golap::DecompressEnv,
                    golap::PredInfo<decltype(Supplier::s_region),RegionAPred>,
                    golap::PredInfo<decltype(Date::d_year),DATE_9297>>(ssbobj, kernel, customer_kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}



bool SSBColLayout::query3_1c(){
    return query3_1c_gen<decltype(pipeline_q3a<Q3PreJoinedNation>),decltype(pipeline_customer_q3c<decltype(Customer::c_region),RegionAsiaPred,golap::PredInfo>),Q3PreJoinedNation,Q3NATION_GROUP>(*this,pipeline_q3a<Q3PreJoinedNation>,pipeline_customer_q3c<decltype(Customer::c_region),RegionAsiaPred,golap::PredInfo>);
}

bool SSBColLayout::query3_2c(){
    return query3_2c_gen<decltype(pipeline_q3a<Q3PreJoinedCity>),decltype(pipeline_customer_q3c<decltype(Customer::c_nation),NationUSPred,golap::PredInfo>),Q3PreJoinedCity,Q3CITY_GROUP>(*this,pipeline_q3a<Q3PreJoinedCity>,pipeline_customer_q3c<decltype(Customer::c_nation),NationUSPred,golap::PredInfo>);
}

bool SSBColLayout::query3_3c(){
    return query3_3c_gen<decltype(pipeline_q3a<Q3PreJoinedCity>),decltype(pipeline_customer_q3c<decltype(Customer::c_city),CityKIPred,golap::PredInfo>),Q3PreJoinedCity,Q3CITY_GROUP>(*this,pipeline_q3a<Q3PreJoinedCity>,pipeline_customer_q3c<decltype(Customer::c_city),CityKIPred,golap::PredInfo>);
}

bool SSBColLayout::query3_4c(){
    return query3_4c_gen<decltype(pipeline_q3a<Q3PreJoinedCity>),decltype(pipeline_customer_q3c<decltype(Customer::c_city),CityKIPred,golap::PredInfo>),Q3PreJoinedCity,Q3CITY_GROUP>(*this,pipeline_q3a<Q3PreJoinedCity>,pipeline_customer_q3c<decltype(Customer::c_city),CityKIPred,golap::PredInfo>);
}

bool SSBColLayout::query3_5c(){
    return query3_5c_gen<decltype(pipeline_q3a<Q3PreJoinedNation>),decltype(pipeline_customer_q3c<decltype(Customer::c_region),RegionAPred,golap::PredInfo>),Q3PreJoinedNation,Q3NATION_GROUP>(*this,pipeline_q3a<Q3PreJoinedNation>,pipeline_customer_q3c<decltype(Customer::c_region),RegionAPred,golap::PredInfo>);
}
