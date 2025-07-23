#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "02.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*

*/

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using PARTKEY_TYPE = decltype(Lineorder::lo_partkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);
using YEAR_TYPE = decltype(Date::d_year);
using MONTHNUMINYEAR_TYPE = decltype(Date::d_monthnuminyear);

using Q2preaggrSUPP_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Supplier::s_key), decltype(Supplier::s_region), decltype(Supplier::s_nation)>;

template <typename BLOCK_ENV, typename SUPP_PRED>
bool query2pre_aggr(SSBColLayout &ssbobj, Q2preaggrSUPP_COLUMNS &supplier_dev, SUPP_PRED supp_pred){
    cudaSetDevice(ssbobj.var.cuda_device);

    uint64_t tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(ORDERDATE_TYPE),sizeof(PARTKEY_TYPE),
                                                           sizeof(SUPPKEY_TYPE),sizeof(REVENUE_TYPE)});
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = 0;
    /**
    * - Write the chunk-compressed lineorder table to disk. 
    */
    ssbobj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,uint64_t> columns{
        {"lo_suppkey",tuples_per_chunk},
        {"rev_per_week",tuples_per_chunk},
        {"d_year",tuples_per_chunk},
        {"d_monthnuminyear",tuples_per_chunk},
        {"d_weeknuminyear",tuples_per_chunk},
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

        if (ssbobj.var.chunk_bytes == (uint64_t)-1) compinfos[a_col.attr_name].chunk_bytes = (uint64_t) -1;
        for (auto &tup_count : ssbobj.var.chunk_size_vec) compinfos[a_col.attr_name].chunk_size_vec.push_back(tup_count*a_col.value_size);

        util::Log::get().debug_fmt("Compressing col %lu=>%s to disk, algo=%s",col_idx,a_col.attr_name.c_str(),algo.c_str());
        if constexpr (std::is_same_v<BLOCK_ENV,golap::DecompressEnv>){
            ssbobj.var.comp_ms += golap::prepare_compressed_device(a_col, num_tuples, compinfos[a_col.attr_name]);
        }else{
            ssbobj.var.comp_ms += golap::prepare_uncompressed(a_col, num_tuples, compinfos[a_col.attr_name]);
        }
        ssbobj.var.comp_bytes += compinfos[a_col.attr_name].get_comp_bytes();
        ssbobj.var.uncomp_bytes += compinfos[a_col.attr_name].uncomp_bytes;
    };

    std::cout << "# Preaggr tuples: " << SortHelper::get().preaggrdb->pre_aggr.num_tuples << "\n";
    SortHelper::get().preaggrdb->pre_aggr.apply(compress_columns_fun);


    uint64_t lo_column_bytes = 0;
    for (auto& comp_info : compinfos){
        lo_column_bytes += comp_info.second.get_comp_bytes();
    }

    // prepare hashmaps of the three joins
    golap::HashMap supplier_hj(supplier_dev.num_tuples,supplier_dev.col<0>().data());

    uint64_t num_groups = 500;
    golap::MirrorMem groups(golap::Tag<YEARMONTH_GROUP>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    checkCudaErrors(cudaMemset(aggs.dev.ptr<uint8_t>(),0,aggs.dev.size_bytes()));
    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<YEARMONTH_GROUP>(), aggs.dev.ptr<uint64_t>()); 

    golap::MirrorMem debug_aggs(golap::Tag<uint64_t>{},5);
    checkCudaErrors(cudaMemset(debug_aggs.dev.ptr<uint8_t>(),0,debug_aggs.dev.size_bytes()));

    std::vector<golap::TableLoader<BLOCK_ENV>> envs;
    envs.reserve(ssbobj.var.workers);

    util::SliceSeq workslice(compinfos["lo_suppkey"].blocks.size(), ssbobj.var.workers);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_suppkey"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        // envs.emplace_back(startblock, endblock, compinfos["lo_suppkey"], compinfos["lo_partkey"],
        //                   compinfos["lo_suppkey"], compinfos["lo_revenue"]);
        envs.emplace_back(5);
        envs[pipeline_idx].add("lo_suppkey", all_blocks_idxs, startblock, endblock, compinfos["lo_suppkey"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("rev_per_week", all_blocks_idxs, startblock, endblock, compinfos["rev_per_week"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("d_year", all_blocks_idxs, startblock, endblock, compinfos["d_year"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("d_monthnuminyear", all_blocks_idxs, startblock, endblock, compinfos["d_monthnuminyear"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("d_weeknuminyear", all_blocks_idxs, startblock, endblock, compinfos["d_weeknuminyear"], nvcomp::TypeOf<uint8_t>());
    }

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    // misuse some streams ...
    auto& supp_build_stream = envs[1%ssbobj.var.workers].blockenvs.at("lo_suppkey").gstream.stream;

    /**
     * Start the timer
     */
    util::Timer timer;

    ssbobj.tables.supplier.col<Supplier::KEY>().transfer(supplier_dev.col<0>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::REGION>().transfer(supplier_dev.col<1>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::NATION>().transfer(supplier_dev.col<2>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);


    golap::hash_map_build_pred_other<<<util::div_ceil(supplier_dev.num_tuples,512),512,0,supp_build_stream>>>(
                                            supplier_hj,supplier_dev.num_tuples,golap::DirectKey<decltype(Supplier::s_key)>(),
                                            supp_pred);

    checkCudaErrors(cudaStreamSynchronize(supp_build_stream));

    // debug:

    // util::Log::get().info_fmt("Supp tuples=%lu",supplier_dev.num_tuples);
    // golap::HostMem supplier_filled_hst(golap::Tag<uint64_t>(),1);
    // checkCudaErrors(cudaMemcpy(supplier_filled_hst.ptr<uint64_t>(), supplier_hj.filled, sizeof(uint64_t), cudaMemcpyDefault));
    // util::Log::get().info_fmt("In supplier hashmap %lu",supplier_filled_hst.ptr<uint64_t>()[0]);

    // end of debug


    // start the main pipeline
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx]{

            cudaSetDevice(ssbobj.var.cuda_device);
            auto &env = envs[pipeline_idx];

            uint64_t tuples_this_round;
            uint64_t round = 0;
            uint32_t num_blocks;
            while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

                tuples_this_round = env.blockenvs.at("lo_suppkey").myblocks[round].tuples;

                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                pipeline_q2_pre_aggr<<<num_blocks,512,0,env.rootstream.stream>>>(
                                                                                supplier_hj,
                                                                                hash_agg,
                                                                                env.blockenvs.at("lo_suppkey").decomp_buf.template ptr<SUPPKEY_TYPE>(),
                                                                                env.blockenvs.at("rev_per_week").decomp_buf.template ptr<uint64_t>(),
                                                                                env.blockenvs.at("d_year").decomp_buf.template ptr<YEAR_TYPE>(),
                                                                                env.blockenvs.at("d_monthnuminyear").decomp_buf.template ptr<MONTHNUMINYEAR_TYPE>(),
                                                                                SumAgg(),
                                                                                tuples_this_round
                );
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

    ssbobj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */

    debug_aggs.sync_to_host();
    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, num_groups};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, num_groups*sizeof(uint32_t), cudaMemcpyDefault));


    // util::Log::get().info_fmt("Sum(lo_revenue), d_year, p_brand1");
    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;
        num_actual_groups += 1;

        auto& group = groups.hst.ptr<YEARMONTH_GROUP>()[i];
        complete_sum += aggs.hst.ptr<uint64_t>()[i];;
        util::Log::get().info_fmt("%lu\t%d\t%d",aggs.hst.ptr<uint64_t>()[i],group.d_year,group.d_monthnuminyear);
    }
    // util::Log::get().info_fmt("raw_lineorder=%lu, after_part=%lu, after_supp=%lu, after_date=%lu, actual_groups=%lu",
    //                           debug_aggs.hst.ptr<uint64_t>()[0],
    //                           debug_aggs.hst.ptr<uint64_t>()[1],
    //                           debug_aggs.hst.ptr<uint64_t>()[2],
    //                           debug_aggs.hst.ptr<uint64_t>()[3],
    //                           num_actual_groups);
    // util::Log::get().info_fmt("After parts hashmap (groups)=%lu",debug_aggs.hst.ptr<uint64_t>()[0]); //debug
    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);
    return true;
}


std::shared_ptr<Q2preaggrSUPP_COLUMNS> prepareQ2preaggrSUPP(SSBColLayout &ssbobj){
    // copy supplier columns to device (uncompressed for now)
    auto supplier_dev = std::make_shared<Q2preaggrSUPP_COLUMNS>("s_key,s_region,s_nation",ssbobj.tables.supplier.num_tuples);
    supplier_dev->num_tuples = ssbobj.tables.supplier.num_tuples;

    return supplier_dev;
}

bool query2_1_pre_aggr_gen(SSBColLayout &ssbobj){
    auto supplier_dev = prepareQ2preaggrSUPP(ssbobj);

    golap::PredInfo<decltype(Supplier::s_region),RegionAmericaPred> supp_pred{supplier_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED") return query2pre_aggr<golap::LoadEnv>(ssbobj, *supplier_dev, supp_pred);
    else return query2pre_aggr<golap::DecompressEnv>(ssbobj, *supplier_dev, supp_pred);
}


bool query2_2_pre_aggr_gen(SSBColLayout &ssbobj){
    auto supplier_dev = prepareQ2preaggrSUPP(ssbobj);

    auto supp_pred = golap::PredInfo<decltype(Supplier::s_nation), NationChinaPred>{supplier_dev->col<2>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED") return query2pre_aggr<golap::LoadEnv>(ssbobj, *supplier_dev, supp_pred);
    else return query2pre_aggr<golap::DecompressEnv>(ssbobj, *supplier_dev, supp_pred);
}


bool SSBColLayout::query2_1_pre_aggr(){return query2_1_pre_aggr_gen(*this);}
bool SSBColLayout::query2_2_pre_aggr(){return query2_2_pre_aggr_gen(*this);}