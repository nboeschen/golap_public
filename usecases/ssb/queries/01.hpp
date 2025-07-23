#pragma once
#include <any>
#include "query_common.cuh"
#include "../ssb.hpp"
#include "01.cuh"

#include "util.hpp"
#include "join.cuh"
#include "dev_structs.cuh"
#include "apps.cuh"
#include "metadata.cuh"
#include "../data/SSB_def.hpp"
#include "../helper.hpp"

/*
select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key
and d_year = 1993 
and lo_discount between 1 and 3
and lo_quantity < 25;

select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_yearmonthnum = 199401 
and lo_discount between 4 and 6
and lo_quantity between 26 and 35;

select sum(lo_extendedprice*lo_discount) as revenue 
from lineorder, date 
where lo_orderdate = d_key 
and d_weeknuminyear = 6 
and d_year = 1994 
and lo_discount between 5 and 7 
and lo_quantity between 26 and 35;

 */

using Q1DATE_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Date::d_key),
                                        decltype(Date::d_year),decltype(Date::d_yearmonthnum),
                                        decltype(Date::d_weeknuminyear)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using EXTENDEDPRICE_TYPE = decltype(Lineorder::lo_extendedprice);
using DISCOUNT_TYPE = decltype(Lineorder::lo_discount);
using QUANTITY_TYPE = decltype(Lineorder::lo_quantity);
using YEAR_TYPE = decltype(Date::d_year);
using YEARMONTHNUM_TYPE = decltype(Date::d_yearmonthnum);
using WEEKNUMINYEAR_TYPE = decltype(Date::d_weeknuminyear);


template <template <typename> class TABLE_LOADER, typename BLOCK_ENV, uint8_t DISCOUNT_LO, uint8_t DISCOUNT_HI, uint8_t QUANTITY_LO, uint8_t QUANTITY_HI, typename... DATE_PREDS>
bool query1(SSBColLayout &ssbobj, Q1DATE_COLUMNS& date_dev, DATE_PREDS... date_preds){
    /**
     * Date: d_key, d_year, d_yearmonthnum, d_weeknuminyear (Keep full columns on device)
     * Lineorder: lo_orderdate, lo_extendedprice, lo_discount, lo_quantity
     * (streamed through)
     *
     */
    cudaSetDevice(ssbobj.var.cuda_device);
    uint64_t tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(ORDERDATE_TYPE),sizeof(EXTENDEDPRICE_TYPE),
                                                            sizeof(DISCOUNT_TYPE),sizeof(QUANTITY_TYPE)});
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = 0;
    /**
     * - Write the chunk-compressed lineorder table to disk.
     */
    ssbobj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,std::tuple<uint64_t,golap::MetaFlags>> columns{
        {"lo_orderdate",{tuples_per_chunk,golap::MetaFlags::DATA}}, // Lineorder::ORDERDATE,
        {"lo_extendedprice",{tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}}, // Lineorder::EXTENDEDPRICE,
        {"lo_discount",{tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}}, // Lineorder::DISCOUNT,
        {"lo_quantity",{tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}}, // Lineorder::QUANTITY
        {"meta_d_year",{tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_d_yearmonthnum",{tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_d_weeknuminyear",{tuples_per_chunk,golap::MetaFlags::META}}
    };

    std::unordered_map<std::string, golap::CompInfo> compinfos;
    std::unordered_map<std::string, std::any> minmaxmeta;
    std::unordered_map<std::string, std::any> histmeta;
    std::unordered_map<std::string, std::any> bloommeta;

    compinfos.reserve(columns.size());
    minmaxmeta.reserve(columns.size());
    histmeta.reserve(columns.size());
    bloommeta.reserve(columns.size());

    // This function is used for both the fact table, as well as the table of prejoined results (for metadata collection)
    auto compress_columns_fun = [&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
        auto entry = columns.find(a_col.attr_name);
        if (entry == columns.end()) return;

        auto& [tuples_per_chunk,usage] = entry->second;

        using COL_TYPE = typename std::remove_reference<decltype(a_col)>::type::value_t;

        auto algo = ssbobj.var.comp_algo;
        if(ssbobj.var.comp_algo == "BEST_BW_COMP"){
            if (BEST_BW_COMP.find(a_col.attr_name) == BEST_BW_COMP.end()) algo = "Gdeflate";
            else algo = BEST_BW_COMP.at(a_col.attr_name);
        }else if(ssbobj.var.comp_algo == "BEST_RATIO_COMP"){
            if (BEST_RATIO_COMP.find(a_col.attr_name) == BEST_RATIO_COMP.end()) algo = "Gdeflate";
            else algo = BEST_RATIO_COMP.at(a_col.attr_name);
        }

        compinfos[a_col.attr_name] = golap::CompInfo{tuples_per_chunk*a_col.value_size,
                                                     num_tuples*a_col.value_size,
                                                     algo, ssbobj.var.nvchunk};

        if (ssbobj.var.chunk_bytes == (uint64_t)-1) compinfos[a_col.attr_name].chunk_bytes = (uint64_t) -1;
        for (auto &tup_count : ssbobj.var.chunk_size_vec) compinfos[a_col.attr_name].chunk_size_vec.push_back(tup_count*a_col.value_size);

        golap::MinMaxMeta<COL_TYPE> *minmaxptr = nullptr;
        golap::EqHistogram<COL_TYPE> *histptr = nullptr;
        golap::BloomMeta<COL_TYPE> *bloomptr = nullptr;

        if (usage & golap::MetaFlags::META){
            minmaxmeta.try_emplace(a_col.attr_name, std::in_place_type<golap::MinMaxMeta<COL_TYPE>>);
            minmaxptr = &std::any_cast<golap::MinMaxMeta<COL_TYPE>&>(minmaxmeta[a_col.attr_name]);

            histmeta.try_emplace(a_col.attr_name, std::in_place_type<golap::EqHistogram<COL_TYPE>>, ssbobj.var.pruning_param);
            histptr = &std::any_cast<golap::EqHistogram<COL_TYPE>&>(histmeta[a_col.attr_name]);

            bloommeta.try_emplace(a_col.attr_name, std::in_place_type<golap::BloomMeta<COL_TYPE>>, ssbobj.var.pruning_p, ssbobj.var.pruning_m);
            bloomptr = &std::any_cast<golap::BloomMeta<COL_TYPE>&>(bloommeta[a_col.attr_name]);
        }


        util::Log::get().debug_fmt("Compressing col %lu=>%s to disk, algo=%s",col_idx,a_col.attr_name.c_str(),algo.c_str());
        if constexpr (!std::is_same_v<BLOCK_ENV,golap::LoadEnv>){
            ssbobj.var.comp_ms += golap::prepare_compressed_device(a_col, num_tuples, compinfos[a_col.attr_name],minmaxptr,histptr,bloomptr);
        }else{
            ssbobj.var.comp_ms += golap::prepare_uncompressed(a_col, num_tuples, compinfos[a_col.attr_name],minmaxptr,histptr,bloomptr);
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
        ssbobj.var.comp_bytes += compinfos[a_col.attr_name].get_comp_bytes();
        ssbobj.var.uncomp_bytes += compinfos[a_col.attr_name].uncomp_bytes;
    };

    ssbobj.tables.lineorder.apply(compress_columns_fun);

    SortHelper::get().prejoined->apply(compress_columns_fun);

    // empty enterprise ssd cache by writing to different offset
    // golap::HostMem OneGB_empty{golap::Tag<char>{},(1<<30)};
    // uint64_t write_dummy_offset = 500l*(1<<30);
    // for (int write_dummy = 0; write_dummy<20;++write_dummy){
    //     golap::StorageManager::get().host_write_bytes(OneGB_empty.ptr<char>(), OneGB_empty.size_bytes(), write_dummy_offset);
    //     write_dummy_offset += OneGB_empty.size_bytes();
    // }
    /**
     * - Prepare the workers.
     * - The actual timed execution should include:
     *      o The build phase on the date table (including filter)
     *      o The pipelined join per chunk (+ filter and aggregation)
     */

    golap::HashMap hash_map(date_dev.col<0>().size(),date_dev.col<0>().data());

    golap::MirrorMem agg_results{golap::Tag<uint64_t>{},8};
    checkCudaErrors(cudaMemset(agg_results.dev.ptr<char>(),0,agg_results.size_bytes()));
    uint64_t chunk_num = compinfos["lo_orderdate"].blocks.size();

    // join prune related:
    golap::MirrorMem build_side_keys{golap::Tag<uint64_t>{},date_dev.col<0>().size()};
    golap::MirrorMem build_side_keys_num{golap::Tag<uint64_t>{},1};
    build_side_keys_num.dev.set(0);

    golap::MirrorMem quantity_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem discount_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem orderdate_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem extended_price_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem dim_date_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem combined_check{golap::Tag<uint16_t>{}, chunk_num};
    quantity_check.dev.set(0);
    discount_check.dev.set(0);
    orderdate_check.dev.set(0);
    extended_price_check.dev.set(0);
    dim_date_check.dev.set(0);
    combined_check.dev.set(0);
    std::atomic<uint64_t> pruned_bytes{0};
    std::atomic<uint64_t> pruned_chunks{0};

    std::vector<TABLE_LOADER<BLOCK_ENV>> envs;
    envs.reserve(ssbobj.var.workers);

    util::SliceSeq workslice(compinfos["lo_orderdate"].blocks.size(), ssbobj.var.workers);

    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_orderdate"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);
    std::random_shuffle(all_blocks_idxs.begin(),all_blocks_idxs.end());

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        envs.emplace_back(4);
        envs[pipeline_idx].add("lo_orderdate", all_blocks_idxs, startblock, endblock, compinfos["lo_orderdate"], nvcomp::TypeOf<ORDERDATE_TYPE>());
        envs[pipeline_idx].add("lo_extendedprice", all_blocks_idxs, startblock, endblock, compinfos["lo_extendedprice"], nvcomp::TypeOf<EXTENDEDPRICE_TYPE>());
        envs[pipeline_idx].add("lo_discount", all_blocks_idxs, startblock, endblock, compinfos["lo_discount"], nvcomp::TypeOf<DISCOUNT_TYPE>());
        envs[pipeline_idx].add("lo_quantity", all_blocks_idxs, startblock, endblock, compinfos["lo_quantity"], nvcomp::TypeOf<QUANTITY_TYPE>());
    }

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    EXTENDEDPRICE_TYPE ext_price_lo = 0;
    EXTENDEDPRICE_TYPE ext_price_hi = (EXTENDEDPRICE_TYPE) -1;
    if (ssbobj.var.extended_price_pred){
        ext_price_lo = 3990000;
        ext_price_hi = 4000000;
    }

    auto& date_pred_stream = envs[0].rootstream.stream;
    auto& meta_stream = envs[1%ssbobj.var.workers].rootstream.stream;

    /**
     * Start timer
     */
    util::Timer timer;
    ssbobj.tables.date.col<Date::KEY>().transfer(date_dev.col<0>().data(), ssbobj.tables.date.num_tuples, date_pred_stream);
    ssbobj.tables.date.col<Date::YEAR>().transfer(date_dev.col<1>().data(), ssbobj.tables.date.num_tuples, date_pred_stream);
    ssbobj.tables.date.col<Date::YEARMONTHNUM>().transfer(date_dev.col<2>().data(), ssbobj.tables.date.num_tuples, date_pred_stream);
    ssbobj.tables.date.col<Date::WEEKNUMINYEAR>().transfer(date_dev.col<3>().data(), ssbobj.tables.date.num_tuples, date_pred_stream);

    golap::hash_map_build_pred_other<<<util::div_ceil(date_dev.col<0>().size(),512),512,0,date_pred_stream>>>(
                                            hash_map,date_dev.num_tuples,golap::DirectKey<decltype(Date::d_key)>(),
                                            date_preds...
                                            );

    if (ssbobj.var.join_prune){
        golap::collect_hashmap_keys<<<util::div_ceil(date_dev.col<0>().size(),512),512,0,date_pred_stream>>>(hash_map, build_side_keys.dev.ptr<uint64_t>(), build_side_keys_num.dev.ptr<uint64_t>());
    }
    // debug:
    // build_side_keys.sync_to_host(date_pred_stream);
    // build_side_keys_num.sync_to_host(date_pred_stream);
    // checkCudaErrors(cudaStreamSynchronize(date_pred_stream));

    // printf("In date hashmap: %lu\n", build_side_keys_num.hst.ptr<uint64_t>()[0]);
    // for(uint64_t i = 0; i< build_side_keys_num.hst.ptr<uint64_t>()[0]; ++i){
    //     printf("Datekey: %lu \n",build_side_keys.hst.ptr<uint64_t>()[i]);
    // }

    uint64_t block_num = std::min((long long)ssbobj.var.block_limit, util::div_ceil(chunk_num,512));
    golap::MetaChecker checker{block_num, chunk_num, combined_check.dev.ptr<uint16_t>(), quantity_check.dev.ptr<uint16_t>(), meta_stream};
    util::Timer prune_timer;

    if (ssbobj.var.pruning == "MINMAX"){
        checker.add_minmax<QUANTITY_TYPE>(minmaxmeta["lo_quantity"], QUANTITY_LO, QUANTITY_HI);
        checker.add_minmax<DISCOUNT_TYPE>(minmaxmeta["lo_discount"], DISCOUNT_LO, DISCOUNT_HI);
        if (ssbobj.var.join_prune){
            checkCudaErrors(cudaStreamSynchronize(date_pred_stream));
            golap::check_mmmeta_multiple_pts<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::MinMaxMeta<ORDERDATE_TYPE>&>(minmaxmeta["lo_orderdate"]), orderdate_check.dev.ptr<uint16_t>(), build_side_keys.dev.ptr<uint64_t>(), build_side_keys_num.dev.ptr<uint64_t>());
            golap::combine_and<<<block_num,512,0,meta_stream>>>(combined_check.dev.ptr<uint16_t>(), combined_check.dev.ptr<uint16_t>(), orderdate_check.dev.ptr<uint16_t>(), chunk_num);
        }
        if (ssbobj.var.query == "query1.1"){
            checker.add_minmax<YEAR_TYPE>(minmaxmeta["meta_d_year"], (YEAR_TYPE)1993, (YEAR_TYPE)1993);
        }else if (ssbobj.var.query == "query1.2"){
            checker.add_minmax<YEARMONTHNUM_TYPE>(minmaxmeta["meta_d_yearmonthnum"], (YEARMONTHNUM_TYPE)199401, (YEARMONTHNUM_TYPE)199401);
        }else if (ssbobj.var.query == "query1.3"){
            checker.add_minmax<YEAR_TYPE>(minmaxmeta["meta_d_year"], (YEAR_TYPE)1994, (YEAR_TYPE)1994);
            checker.add_minmax<WEEKNUMINYEAR_TYPE>(minmaxmeta["meta_d_weeknuminyear"], (WEEKNUMINYEAR_TYPE)6, (WEEKNUMINYEAR_TYPE)6);
        }
        if (ssbobj.var.extended_price_pred){
            checker.add_minmax<EXTENDEDPRICE_TYPE>(minmaxmeta["lo_extendedprice"], ext_price_lo, ext_price_hi);
        }
    }else if (ssbobj.var.pruning == "HIST"){
        checker.add_hist<QUANTITY_TYPE>(histmeta["lo_quantity"], QUANTITY_LO, QUANTITY_HI);
        checker.add_hist<DISCOUNT_TYPE>(histmeta["lo_discount"], DISCOUNT_LO, DISCOUNT_HI);

        if (ssbobj.var.join_prune){
            checkCudaErrors(cudaStreamSynchronize(date_pred_stream));
            golap::check_hist_multiple_pts<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::EqHistogram<ORDERDATE_TYPE>&>(histmeta["lo_orderdate"]), orderdate_check.dev.ptr<uint16_t>(), build_side_keys.dev.ptr<uint64_t>(), build_side_keys_num.dev.ptr<uint64_t>());
            golap::combine_and<<<block_num,512,0,meta_stream>>>(combined_check.dev.ptr<uint16_t>(), combined_check.dev.ptr<uint16_t>(), orderdate_check.dev.ptr<uint16_t>(), chunk_num);
        }
        if (ssbobj.var.query == "query1.1"){
            checker.add_hist<YEAR_TYPE>(histmeta["meta_d_year"], (YEAR_TYPE)1993, (YEAR_TYPE)1993);
        }else if (ssbobj.var.query == "query1.2"){
            checker.add_hist<YEARMONTHNUM_TYPE>(histmeta["meta_d_yearmonthnum"], (YEARMONTHNUM_TYPE)199401, (YEARMONTHNUM_TYPE)199401);
        }else if (ssbobj.var.query == "query1.3"){
            checker.add_hist<YEAR_TYPE>(histmeta["meta_d_year"], (YEAR_TYPE)1994, (YEAR_TYPE)1994);
            checker.add_hist<WEEKNUMINYEAR_TYPE>(histmeta["meta_d_weeknuminyear"], (WEEKNUMINYEAR_TYPE)6, (WEEKNUMINYEAR_TYPE)6);
        }
        if (ssbobj.var.extended_price_pred){
            checker.add_hist<EXTENDEDPRICE_TYPE>(histmeta["lo_extendedprice"], ext_price_lo, ext_price_hi);
        }
    }else if (ssbobj.var.pruning == "BLOOM"){
        // for a very short range like discount, check the bloom filter point-wise
        checker.add_bloom<DISCOUNT_TYPE>(bloommeta["lo_discount"], DISCOUNT_LO);
        golap::check_bloom<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::BloomMeta<DISCOUNT_TYPE>&>(bloommeta["lo_discount"]), combined_check.dev.ptr<uint16_t>(), (DISCOUNT_TYPE)(DISCOUNT_LO+1));
        golap::check_bloom<<<block_num,512,0,meta_stream>>>(std::any_cast<golap::BloomMeta<DISCOUNT_TYPE>&>(bloommeta["lo_discount"]), combined_check.dev.ptr<uint16_t>(), (DISCOUNT_TYPE)(DISCOUNT_LO+2));

        if (ssbobj.var.query == "query1.1"){
            checker.add_bloom<YEAR_TYPE>(bloommeta["meta_d_year"], (YEAR_TYPE)1993);
        }else if (ssbobj.var.query == "query1.2"){
            checker.add_bloom<YEARMONTHNUM_TYPE>(bloommeta["meta_d_yearmonthnum"], (YEARMONTHNUM_TYPE)199401);
        }else if (ssbobj.var.query == "query1.3"){
            checker.add_bloom<YEAR_TYPE>(bloommeta["meta_d_year"], (YEAR_TYPE)1994);
            checker.add_bloom<WEEKNUMINYEAR_TYPE>(bloommeta["meta_d_weeknuminyear"], (WEEKNUMINYEAR_TYPE)6);
        }
    }else if (ssbobj.var.pruning == "COMBINED"){
        checker.add_hist<QUANTITY_TYPE>(histmeta["lo_quantity"], QUANTITY_LO, QUANTITY_HI);
        checker.add_hist<DISCOUNT_TYPE>(histmeta["lo_discount"], DISCOUNT_LO, DISCOUNT_HI);

        if (ssbobj.var.query == "query1.1"){
            checker.add_bloom<YEAR_TYPE>(bloommeta["meta_d_year"], (YEAR_TYPE)1993);
        }else if (ssbobj.var.query == "query1.2"){
            checker.add_bloom<YEARMONTHNUM_TYPE>(bloommeta["meta_d_yearmonthnum"], (YEARMONTHNUM_TYPE)199401);
        }else if (ssbobj.var.query == "query1.3"){
            checker.add_bloom<YEAR_TYPE>(bloommeta["meta_d_year"], (YEAR_TYPE)1994);
            checker.add_bloom<WEEKNUMINYEAR_TYPE>(bloommeta["meta_d_weeknuminyear"], (WEEKNUMINYEAR_TYPE)6);
        }
    }

    if (ssbobj.var.pruning != "DONTPRUNE" && ssbobj.var.pruning != "MINMAXHOST") combined_check.sync_to_host(meta_stream);

    checkCudaErrors(cudaStreamSynchronize(meta_stream));
    ssbobj.var.prune_ms = prune_timer.elapsed();

    checkCudaErrors(cudaStreamSynchronize(date_pred_stream));

    util::Log::get().info_fmt("chunk_num: %lu, tuples_per_chunk: %lu, Tuples total %lu", chunk_num, tuples_per_chunk, ssbobj.tables.lineorder.num_tuples);

    // if (ssbobj.var.extended_price_pred){
    //     for(uint64_t i = 0; i< chunk_num; ++i){
    //         printf("%d, ", extended_price_check.hst.ptr<uint16_t>()[i]);
    //     }
    //     printf("\n");
    // }
    // for(uint64_t i = 0; i< chunk_num; ++i){
    //     printf("%d, ", combined_check.hst.ptr<uint16_t>()[i]);
    // }
    // printf("\n");

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx]{

            cudaSetDevice(ssbobj.var.cuda_device);
            auto &env = envs[pipeline_idx];

            uint64_t tuples_this_round,global_block_idx;
            uint64_t round = 0;
            uint32_t num_blocks;

            // while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
            while(round< env.blockenvs.at("lo_orderdate").myblocks.size()){

                global_block_idx = env.blockenvs.at("lo_orderdate").myblock_idxs[round];
                // auto& discount_meta = std::any_cast<golap::MinMaxMeta<DISCOUNT_TYPE>&>(minmaxmeta["lo_discount"])[global_block_idx];
                // auto& quantity_meta = std::any_cast<golap::MinMaxMeta<QUANTITY_TYPE>&>(minmaxmeta["lo_quantity"])[global_block_idx];

                // check predicate:
                if (ssbobj.var.pruning != "DONTPRUNE" && ssbobj.var.pruning != "MINMAXHOST" &&
                    combined_check.hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    util::Log::get().debug_fmt("Thread[%lu, round%lu] would skip the next chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);
                    pruned_bytes.fetch_add(env.blockenvs.at("lo_orderdate").myblocks[round].size
                                          +env.blockenvs.at("lo_quantity").myblocks[round].size
                                          +env.blockenvs.at("lo_extendedprice").myblocks[round].size
                                          +env.blockenvs.at("lo_discount").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }else if (ssbobj.var.pruning == "MINMAXHOST" &&
                           (! std::any_cast<golap::MinMaxMeta<QUANTITY_TYPE>&>(minmaxmeta["lo_quantity"]).check_pred_host(global_block_idx, QUANTITY_LO, QUANTITY_HI) ||
        ! std::any_cast<golap::MinMaxMeta<DISCOUNT_TYPE>&>(minmaxmeta["lo_discount"]).check_pred_host(global_block_idx, DISCOUNT_LO, DISCOUNT_HI))){

                    util::Log::get().debug_fmt("Thread[%lu, round%lu] would host prune / skip the next chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);
                    pruned_bytes.fetch_add(env.blockenvs.at("lo_orderdate").myblocks[round].size
                                          +env.blockenvs.at("lo_quantity").myblocks[round].size
                                          +env.blockenvs.at("lo_extendedprice").myblocks[round].size
                                          +env.blockenvs.at("lo_discount").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }


                if (!env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));


                tuples_this_round = env.blockenvs.at("lo_orderdate").myblocks[round].tuples;
                util::Log::get().debug_fmt("Thread[%lu, round%lu] Block %lu, %lu tuples", pipeline_idx, round, global_block_idx, tuples_this_round);

                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                pipeline_q1_lineorder<<<num_blocks,512,0,env.rootstream.stream>>>(hash_map,
                                                                                 env.blockenvs.at("lo_orderdate").decomp_buf.template ptr<ORDERDATE_TYPE>(),
                                                                                 env.blockenvs.at("lo_discount").decomp_buf.template ptr<DISCOUNT_TYPE>(),
                                                                                 DiscountPred<DISCOUNT_LO,DISCOUNT_HI>(),
                                                                                 env.blockenvs.at("lo_quantity").decomp_buf.template ptr<QUANTITY_TYPE>(),
                                                                                 QuantityPred<QUANTITY_LO,QUANTITY_HI>(),
                                                                                 env.blockenvs.at("lo_extendedprice").decomp_buf.template ptr<EXTENDEDPRICE_TYPE>(),
                                                                                 ext_price_lo,ext_price_hi,
                                                                                 SumAgg(), agg_results.dev.template ptr<uint64_t>(),
                                                                                 tuples_this_round
                                                                                 //,agg_results.dev.template ptr<uint64_t>()+1,
                                                                                 // agg_results.dev.template ptr<uint64_t>()+2
                                                                                 );
                checkCudaErrors(cudaEventRecord(env.rootevent.event, env.rootstream.stream));
                round += 1;

            } // end of while
            checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
        });
    }

    for(auto &thread: threads) thread.join();
    agg_results.sync_to_host(envs[0].rootstream.stream);
    checkCudaErrors(cudaStreamSynchronize(envs[0].rootstream.stream));

    ssbobj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */
    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();
    ssbobj.var.pruned_bytes = pruned_bytes.load();

    // util::Log::get().info_fmt("Filtered lineorders : %lu of %lu",agg_results.hst.template ptr<uint64_t>()[1], ssbobj.tables.lineorder.num_tuples);
    // util::Log::get().info_fmt("Results after join : %lu",agg_results.hst.template ptr<uint64_t>()[2]);
    util::Log::get().info_fmt("The result is: %lu",agg_results.hst.template ptr<uint64_t>()[0]);
    util::Log::get().info_fmt("Pruned: %lu of %lu chunks (%.2f), %lu of %lu bytes (%.2f), took %.2f ms",
                                pruned_chunks.load(),chunk_num,(double)pruned_chunks.load()/chunk_num,
                                pruned_bytes.load(),ssbobj.var.comp_bytes,(double)pruned_bytes.load()/ssbobj.var.comp_bytes,
                                ssbobj.var.prune_ms);



    return true;
}

bool SSBColLayout::query1_1(){
    /**
     * - Prepare the necessary date columns.
     */
    Q1DATE_COLUMNS date_dev("d_key,d_year,d_yearmonthnum,d_weeknuminyear",tables.date.num_tuples);
    date_dev.num_tuples = tables.date.num_tuples;

    auto pred = golap::PredInfo<decltype(Date::d_year),golap::ConstPred<decltype(Date::d_year),1993>>{date_dev.col<1>().data()};

    if(var.comp_algo == "UNCOMPRESSED") return query1<golap::TableLoader, golap::LoadEnv,1,3,0,24>(*this, date_dev, pred);
    else if(var.dataflow == "SSD2GPU_BATCH") return query1<golap::BatchTableLoader, golap::DecompressEnvWOLoad,1,3,0,24>(*this, date_dev, pred);
    else return query1<golap::TableLoader, golap::DecompressEnv,1,3,0,24>(*this, date_dev, pred);
}
bool SSBColLayout::query1_2(){
    /**
     * - Prepare the necessary date columns.
     */
    Q1DATE_COLUMNS date_dev("d_key,d_year,d_yearmonthnum,d_weeknuminyear",tables.date.num_tuples);
    date_dev.num_tuples = tables.date.num_tuples;

    auto pred = golap::PredInfo<decltype(Date::d_yearmonthnum),golap::ConstPred<decltype(Date::d_yearmonthnum),199401>>{date_dev.col<2>().data()};

    if(var.comp_algo == "UNCOMPRESSED") return query1<golap::TableLoader, golap::LoadEnv,4,6,26,35>(*this,  date_dev, pred);
    else if(var.dataflow == "SSD2GPU_BATCH") return query1<golap::BatchTableLoader, golap::DecompressEnvWOLoad,4,6,26,35>(*this,  date_dev, pred);
    else return query1<golap::TableLoader, golap::DecompressEnv,4,6,26,35>(*this,  date_dev, pred);
}
bool SSBColLayout::query1_3(){
    /**
     * - Prepare the necessary date columns.
     */
    Q1DATE_COLUMNS date_dev("d_key,d_year,d_yearmonthnum,d_weeknuminyear",tables.date.num_tuples);
    date_dev.num_tuples = tables.date.num_tuples;

    auto predA = golap::PredInfo<decltype(Date::d_weeknuminyear),golap::ConstPred<decltype(Date::d_weeknuminyear),6>>{date_dev.col<3>().data()};
    auto predB = golap::PredInfo<decltype(Date::d_year),golap::ConstPred<decltype(Date::d_year),1994>>{date_dev.col<1>().data()};

    if(var.comp_algo == "UNCOMPRESSED") return query1<golap::TableLoader, golap::LoadEnv,5,7,26,35>(*this, date_dev, predA, predB);
    else if(var.dataflow == "SSD2GPU_BATCH") return query1<golap::BatchTableLoader, golap::DecompressEnvWOLoad,5,7,26,35>(*this, date_dev, predA, predB);
    else return query1<golap::TableLoader, golap::DecompressEnv,5,7,26,35>(*this, date_dev, predA, predB);
}

