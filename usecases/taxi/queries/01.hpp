#pragma once
#include <any>

#include "mem.hpp"
#include "host_structs.hpp"
#include "dev_structs.cuh"
#include "apps.cuh"
#include "metadata.cuh"

#include "../taxi.hpp"
#include "common.cuh"
#include "01.cuh"

/*
-- 

SELECT EXTRACT_MONTH(tpep_pickup_datetime), COUNT(*) trips 
FROM trips 
WHERE distance >= 5 
GROUP BY month;

SELECT extract('month' FROM tpep_pickup_datetime),COUNT(*) FROM '/mnt/labstore/nboeschen/taxi/*.parquet' 
WHERE trip_distance >= 5 GROUP BY extract('month' FROM tpep_pickup_datetime);


trip_distance >= 2 and trip_distance <= 5
trip_distance >= 5 and trip_distance <= 15
trip_distance >= 15

 */


using PICKUP_TIME_TYPE = decltype(Trips::tpep_pickup_datetime);
using DISTANCE_TYPE = decltype(Trips::Trip_distance);


template <typename BLOCK_ENV>
bool query1gen(TaxiColLayout &obj, DISTANCE_TYPE dist_lo, DISTANCE_TYPE dist_hi){

    cudaSetDevice(obj.var.cuda_device);
    uint64_t tuples_per_chunk = obj.var.chunk_bytes / std::max({sizeof(PICKUP_TIME_TYPE),sizeof(DISTANCE_TYPE)});
    obj.var.comp_bytes = 0;
    obj.var.uncomp_bytes = 0;
    /**
     * - Write the chunk-compressed table to disk.
     */
    obj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,std::tuple<uint64_t,golap::MetaFlags>> columns{
        {"tpep_pickup_datetime",{tuples_per_chunk,golap::MetaFlags::DATA}},
        {"Trip_distance",{tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}},
    };

    std::unordered_map<std::string, golap::CompInfo> compinfos;
    std::unordered_map<std::string, std::any> minmaxmeta;
    std::unordered_map<std::string, std::any> histmeta;
    std::unordered_map<std::string, std::any> bloommeta;
    compinfos.reserve(columns.size());
    minmaxmeta.reserve(columns.size());
    histmeta.reserve(columns.size());
    bloommeta.reserve(columns.size());


    auto compress_columns_fun = [&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        auto entry = columns.find(a_col.attr_name);
        if (entry == columns.end()) return;

        auto& [tuples_per_chunk,usage] = entry->second;

        using COL_TYPE = typename std::remove_reference<decltype(a_col)>::type::value_t;

        auto algo = obj.var.comp_algo;
        if(obj.var.comp_algo == "BEST_BW_COMP"){
            if (BEST_BW_COMP.find(a_col.attr_name) == BEST_BW_COMP.end()) algo = "Gdeflate";
            else algo = BEST_BW_COMP.at(a_col.attr_name);
        }

        compinfos[a_col.attr_name] = golap::CompInfo{tuples_per_chunk*a_col.value_size,
                                                     num_tuples*a_col.value_size,
                                                     algo, obj.var.nvchunk};

        if (obj.var.chunk_bytes == (uint64_t)-1) compinfos[a_col.attr_name].chunk_bytes = (uint64_t) -1;
        for (auto &tup_count : obj.var.chunk_size_vec) compinfos[a_col.attr_name].chunk_size_vec.push_back(tup_count*a_col.value_size);

        golap::MinMaxMeta<COL_TYPE> *minmaxptr = nullptr;
        golap::EqHistogram<COL_TYPE> *histptr = nullptr;
        golap::BloomMeta<COL_TYPE> *bloomptr = nullptr;

        if (usage & golap::MetaFlags::META){
            minmaxmeta.try_emplace(a_col.attr_name, std::in_place_type<golap::MinMaxMeta<COL_TYPE>>);
            minmaxptr = &std::any_cast<golap::MinMaxMeta<COL_TYPE>&>(minmaxmeta[a_col.attr_name]);

            histmeta.try_emplace(a_col.attr_name, std::in_place_type<golap::EqHistogram<COL_TYPE>>, obj.var.pruning_param);
            histptr = &std::any_cast<golap::EqHistogram<COL_TYPE>&>(histmeta[a_col.attr_name]);

            bloommeta.try_emplace(a_col.attr_name, std::in_place_type<golap::BloomMeta<COL_TYPE>>, obj.var.pruning_p, obj.var.pruning_m);
            bloomptr = &std::any_cast<golap::BloomMeta<COL_TYPE>&>(bloommeta[a_col.attr_name]);
        }

        util::Log::get().debug_fmt("Compressing col %lu=>%s to disk, algo=%s",col_idx,a_col.attr_name.c_str(),algo.c_str());
        if constexpr (std::is_same_v<BLOCK_ENV,golap::DecompressEnv>){
            obj.var.comp_ms += golap::prepare_compressed_device(a_col, num_tuples, compinfos[a_col.attr_name],minmaxptr,histptr,bloomptr);
        }else{
            obj.var.comp_ms += golap::prepare_uncompressed(a_col, num_tuples, compinfos[a_col.attr_name],minmaxptr,histptr,bloomptr);
        }

        // copy metadata to host, so that we can copy it back to device if needed
        if (usage & golap::MetaFlags::META){
            std::any_cast<golap::MinMaxMeta<COL_TYPE>&>(minmaxmeta[a_col.attr_name]).to_host();
            std::any_cast<golap::EqHistogram<COL_TYPE>&>(histmeta[a_col.attr_name]).to_host();
            std::any_cast<golap::BloomMeta<COL_TYPE>&>(bloommeta[a_col.attr_name]).to_host();
        }
        // if this was a column we only used to gather metadata, but we actually wont use it otherwise (e.g. load it from disk),
        // return here. The next column will overwrite this one on disk
        if (!(usage & golap::MetaFlags::DATA)){
            golap::StorageManager::get().set_offset(compinfos[a_col.attr_name].start_offset());
            compinfos.erase(a_col.attr_name);
            return;
        }
        obj.var.comp_bytes += compinfos[a_col.attr_name].get_comp_bytes();
        obj.var.uncomp_bytes += compinfos[a_col.attr_name].uncomp_bytes;
    };

    obj.tables.trips.apply(compress_columns_fun);


    uint64_t chunk_num = compinfos["tpep_pickup_datetime"].blocks.size();
    uint64_t num_groups = 50;
    golap::MirrorMem groups(golap::Tag<MONTH_GROUP>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    checkCudaErrors(cudaMemset(aggs.dev.ptr<uint8_t>(),0,aggs.dev.size_bytes()));
    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<MONTH_GROUP>(), aggs.dev.ptr<uint64_t>());
    golap::MirrorMem combined_check{golap::Tag<uint16_t>{}, chunk_num};
    combined_check.dev.set(0);
    std::atomic<uint64_t> pruned_bytes{0};
    std::atomic<uint64_t> pruned_chunks{0};


    std::vector<golap::TableLoader<BLOCK_ENV>> envs;
    envs.reserve(obj.var.workers);

    util::SliceSeq workslice(compinfos["tpep_pickup_datetime"].blocks.size(), obj.var.workers);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["tpep_pickup_datetime"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    std::random_shuffle(all_blocks_idxs.begin(),all_blocks_idxs.end());

    for (uint32_t pipeline_idx=0; pipeline_idx<obj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        envs.emplace_back(4);
        envs[pipeline_idx].add("tpep_pickup_datetime", all_blocks_idxs, startblock, endblock, compinfos["tpep_pickup_datetime"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("Trip_distance", all_blocks_idxs, startblock, endblock, compinfos["Trip_distance"], nvcomp::TypeOf<uint8_t>());
    }

    std::vector<std::thread> threads;
    threads.reserve(obj.var.workers);
    auto& meta_stream = envs[1%obj.var.workers].rootstream.stream;


    /**
     * Start timer
     */
    uint64_t block_num = std::min((long long)obj.var.block_limit, util::div_ceil(chunk_num,512));

    util::Timer timer;

    if (obj.var.pruning == "MINMAX"){
        std::any_cast<golap::MinMaxMeta<DISTANCE_TYPE>&>(minmaxmeta["Trip_distance"]).to_device(meta_stream);
        golap::check_mmmeta<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::MinMaxMeta<DISTANCE_TYPE>&>(minmaxmeta["Trip_distance"]), combined_check.dev.ptr<uint16_t>(), (DISTANCE_TYPE)dist_lo,(DISTANCE_TYPE)dist_hi);

    }else if (obj.var.pruning == "HIST"){
        std::any_cast<golap::EqHistogram<DISTANCE_TYPE>&>(histmeta["Trip_distance"]).to_device(meta_stream);
        golap::check_hist<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::EqHistogram<DISTANCE_TYPE>&>(histmeta["Trip_distance"]), combined_check.dev.ptr<uint16_t>(), (DISTANCE_TYPE)dist_lo, (DISTANCE_TYPE) dist_hi);
    }

    if (obj.var.pruning != "DONTPRUNE") combined_check.sync_to_host(meta_stream);

    checkCudaErrors(cudaStreamSynchronize(meta_stream));
    obj.var.prune_ms = timer.elapsed();


    for (uint32_t pipeline_idx=0; pipeline_idx<obj.var.workers; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx]{

            cudaSetDevice(obj.var.cuda_device);
            auto &env = envs[pipeline_idx];

            uint64_t tuples_this_round,global_block_idx;
            uint64_t round = 0;
            uint32_t num_blocks;

            // while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
            while(round< env.blockenvs.at("tpep_pickup_datetime").myblocks.size()){

                global_block_idx = env.blockenvs.at("tpep_pickup_datetime").myblock_idxs[round];

                // check predicate:
                if (obj.var.pruning != "DONTPRUNE" && combined_check.hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    util::Log::get().debug_fmt("Thread[%lu, round%lu] would skip the next chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);
                    pruned_bytes.fetch_add(env.blockenvs.at("tpep_pickup_datetime").myblocks[round].size
                                          +env.blockenvs.at("Trip_distance").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }

                if (!env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));


                tuples_this_round = env.blockenvs.at("tpep_pickup_datetime").myblocks[round].tuples;
                util::Log::get().debug_fmt("Thread[%lu, round%lu] Block %lu, %lu tuples", pipeline_idx, round, global_block_idx, tuples_this_round);

                num_blocks = std::min((long long)obj.var.block_limit, util::div_ceil(tuples_this_round,512));

                pipeline_q1<<<num_blocks,512,0,env.rootstream.stream>>>(hash_agg,
                                env.blockenvs.at("tpep_pickup_datetime").decomp_buf.template ptr<PICKUP_TIME_TYPE>(),
                                env.blockenvs.at("Trip_distance").decomp_buf.template ptr<DISTANCE_TYPE>(),
                                dist_lo,
                                dist_hi,
                                SumAgg(),
                                tuples_this_round);

                checkCudaErrors(cudaEventRecord(env.rootevent.event, env.rootstream.stream));
                round += 1;

            } // end of while
            checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
        });
    }

    for(auto &thread: threads) thread.join();
    aggs.sync_to_host(envs[0].rootstream.stream);
    groups.sync_to_host(envs[0].rootstream.stream);
    checkCudaErrors(cudaStreamSynchronize(envs[0].rootstream.stream));

    obj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, num_groups};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, num_groups*sizeof(uint32_t), cudaMemcpyDefault));

    obj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    obj.var.host_mem_used = golap::HOST_ALLOCATED.load();
    obj.var.pruned_bytes = pruned_bytes.load();

    for(uint32_t i = 0; i<num_groups; ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;

        auto& group = groups.hst.ptr<MONTH_GROUP>()[i];
        util::Log::get().info_fmt("Month: %d, Sum: %lu",group.month,aggs.hst.ptr<uint64_t>()[i]);
    }

    util::Log::get().info_fmt("Pruned: %lu of %lu chunks (%.2f), %lu of %lu bytes (%.2f)",
                                pruned_chunks.load(),chunk_num,(double)pruned_chunks.load()/chunk_num,
                                pruned_bytes.load(),obj.var.comp_bytes,(double)pruned_bytes.load()/obj.var.comp_bytes);

    return true;
}

bool TaxiColLayout::query1_1(){
    if(var.comp_algo == "UNCOMPRESSED") return query1gen<golap::LoadEnv>(*this, 2, 5);
    else return query1gen<golap::DecompressEnv>(*this, 2, 5);
}

bool TaxiColLayout::query1_2(){
    if(var.comp_algo == "UNCOMPRESSED") return query1gen<golap::LoadEnv>(*this, 5, 15);
    else return query1gen<golap::DecompressEnv>(*this, 5, 15);
}

bool TaxiColLayout::query1_3(){
    if(var.comp_algo == "UNCOMPRESSED") return query1gen<golap::LoadEnv>(*this, 15, 5000);
    else return query1gen<golap::DecompressEnv>(*this, 15, 5000);
}