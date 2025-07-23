#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "02.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*
select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1

select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 between 'MFGR#2221'
and 'MFGR#2228'
and s_region = 'ASIA'
group by d_year, p_brand1
order by d_year, p_brand1;

select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1= 'MFGR#2239'
and s_region = 'EUROPE'
group by d_year, p_brand1
order by d_year, p_brand1;

*/

using Q2PART_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Part::p_key),
                                    decltype(Part::p_brand1),decltype(Part::p_category)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using PARTKEY_TYPE = decltype(Lineorder::lo_partkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);


template <typename KERNEL, typename BLOCK_ENV, typename REGION_PRED, typename... PART_PREDS>
bool query2(SSBColLayout &ssbobj, KERNEL kernel, Q2PART_COLUMNS &part_dev, REGION_PRED region_pred, PART_PREDS... part_preds){
    cudaSetDevice(ssbobj.var.cuda_device);

    /**
     * Generally needed:
     * Lineorder: lo_orderdate, lo_partkey, lo_suppkey, lo_revenue
     * Supplier: s_key, s_region
     * Date: d_key, d_year
     * Parts: p_key, p_category, p_brand1
     */
    uint64_t tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(ORDERDATE_TYPE),sizeof(PARTKEY_TYPE),
                                                           sizeof(SUPPKEY_TYPE),sizeof(REVENUE_TYPE)});
    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = 0;
    /**
    * - Write the chunk-compressed lineorder table to disk. 
    */
    ssbobj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,uint64_t> columns{
        {"lo_orderdate",tuples_per_chunk},// Lineorder::ORDERDATE,
        {"lo_partkey",tuples_per_chunk},// Lineorder::PARTKEY,
        {"lo_suppkey",tuples_per_chunk},// Lineorder::SUPPKEY,
        {"lo_revenue",tuples_per_chunk},// Lineorder::REVENUE
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

    ssbobj.tables.lineorder.apply(compress_columns_fun);

    // empty enterprise ssd cache by writing to different offset
    // golap::HostMem OneGB_empty{golap::Tag<char>{},(1<<30)};
    // uint64_t write_dummy_offset = 500l*(1<<30);
    // for (int write_dummy = 0; write_dummy<20;++write_dummy){
    //     golap::StorageManager::get().host_write_bytes(OneGB_empty.ptr<char>(), OneGB_empty.size_bytes(), write_dummy_offset);
    //     write_dummy_offset += OneGB_empty.size_bytes();
    // }

    // copy date columns to device (uncompressed for now)
    golap::ColumnTable<golap::DeviceMem,decltype(Date::d_key), decltype(Date::d_year)> date_dev("d_key,d_year",ssbobj.tables.date.num_tuples);
    date_dev.num_tuples = ssbobj.tables.date.num_tuples;

    // copy supplier columns to device (uncompressed for now)
    golap::ColumnTable<golap::DeviceMem,decltype(Supplier::s_key), decltype(Supplier::s_region)> supplier_dev("s_key,s_region",ssbobj.tables.supplier.num_tuples);
    supplier_dev.num_tuples = ssbobj.tables.supplier.num_tuples;

    uint64_t lo_column_bytes = 0;
    for (auto& comp_info : compinfos){
        lo_column_bytes += comp_info.second.get_comp_bytes();
    }
    util::Log::get().info_fmt("Lineorder columns bytes: %lu\t(%lu tuples)",lo_column_bytes,ssbobj.tables.lineorder.num_tuples);
    util::Log::get().info_fmt("Parts     columns bytes: %lu\t(%lu tuples)",part_dev.size_bytes(), part_dev.num_tuples);
    util::Log::get().info_fmt("Supplier  columns bytes: %lu\t(%lu tuples)",supplier_dev.size_bytes(), supplier_dev.num_tuples);
    util::Log::get().info_fmt("Date      columns bytes: %lu\t\t(%lu tuples)",date_dev.size_bytes(), date_dev.num_tuples);

    // prepare hashmaps of the three joins
    golap::HashMap part_hj(part_dev.num_tuples,part_dev.col<0>().data());
    golap::HashMap supplier_hj(supplier_dev.num_tuples,supplier_dev.col<0>().data());
    golap::HashMap date_hj(date_dev.num_tuples,date_dev.col<0>().data());

    uint64_t num_groups = 500;
    golap::MirrorMem groups(golap::Tag<YEARBRAND_GROUP>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    checkCudaErrors(cudaMemset(aggs.dev.ptr<uint8_t>(),0,aggs.dev.size_bytes()));
    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<YEARBRAND_GROUP>(), aggs.dev.ptr<uint64_t>()); 

    golap::MirrorMem debug_aggs(golap::Tag<uint64_t>{},5);
    checkCudaErrors(cudaMemset(debug_aggs.dev.ptr<uint8_t>(),0,debug_aggs.dev.size_bytes()));

    std::vector<golap::TableLoader<BLOCK_ENV>> envs;
    envs.reserve(ssbobj.var.workers);

    util::SliceSeq workslice(compinfos["lo_orderdate"].blocks.size(), ssbobj.var.workers);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_orderdate"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        // envs.emplace_back(startblock, endblock, compinfos["lo_orderdate"], compinfos["lo_partkey"],
        //                   compinfos["lo_suppkey"], compinfos["lo_revenue"]);
        envs.emplace_back(4);
        envs[pipeline_idx].add("lo_orderdate", all_blocks_idxs, startblock, endblock, compinfos["lo_orderdate"], nvcomp::TypeOf<ORDERDATE_TYPE>());
        envs[pipeline_idx].add("lo_partkey", all_blocks_idxs, startblock, endblock, compinfos["lo_partkey"], nvcomp::TypeOf<PARTKEY_TYPE>());
        envs[pipeline_idx].add("lo_suppkey", all_blocks_idxs, startblock, endblock, compinfos["lo_suppkey"], nvcomp::TypeOf<SUPPKEY_TYPE>());
        envs[pipeline_idx].add("lo_revenue", all_blocks_idxs, startblock, endblock, compinfos["lo_revenue"], nvcomp::TypeOf<REVENUE_TYPE>());
    }

    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    // misuse some streams ...
    auto& part_build_stream = envs[0].blockenvs.at("lo_partkey").gstream.stream;
    auto& supp_build_stream = envs[1%ssbobj.var.workers].blockenvs.at("lo_suppkey").gstream.stream;
    auto& date_build_stream = envs[2%ssbobj.var.workers].blockenvs.at("lo_orderdate").gstream.stream;

    /**
     * Start the timer
     */
    util::Timer timer;

    ssbobj.tables.part.col<Part::KEY>().transfer(part_dev.col<0>().data(),ssbobj.tables.part.num_tuples, part_build_stream);
    ssbobj.tables.part.col<Part::BRAND1>().transfer(part_dev.col<1>().data(),ssbobj.tables.part.num_tuples, part_build_stream);
    ssbobj.tables.part.col<Part::CATEGORY>().transfer(part_dev.col<2>().data(),ssbobj.tables.part.num_tuples, part_build_stream);

    ssbobj.tables.supplier.col<Supplier::KEY>().transfer(supplier_dev.col<0>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::REGION>().transfer(supplier_dev.col<1>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);

    ssbobj.tables.date.col<Date::KEY>().transfer(date_dev.col<0>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    ssbobj.tables.date.col<Date::YEAR>().transfer(date_dev.col<1>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    

    golap::hash_map_build_pred_other<<<util::div_ceil(part_dev.num_tuples,512),512,0,part_build_stream>>>(
                                            part_hj, part_dev.num_tuples, golap::DirectKey<decltype(Part::p_key)>(),
                                            part_preds...);

    golap::hash_map_build_pred_other<<<util::div_ceil(supplier_dev.num_tuples,512),512,0,supp_build_stream>>>(
                                            supplier_hj,supplier_dev.num_tuples,golap::DirectKey<decltype(Supplier::s_key)>(),
                                            golap::PredInfo<decltype(Supplier::s_region), REGION_PRED>{supplier_dev.col<1>().data(),region_pred});

    golap::hash_map_build<<<util::div_ceil(date_dev.num_tuples,512),512,0,date_build_stream>>>(date_hj,date_dev.num_tuples,golap::DirectKey<decltype(Date::d_key)>());
    checkCudaErrors(cudaStreamSynchronize(part_build_stream));
    checkCudaErrors(cudaStreamSynchronize(supp_build_stream));
    checkCudaErrors(cudaStreamSynchronize(date_build_stream));

    // debug:
    // util::Log::get().info_fmt("Part tuples=%lu",part_dev.num_tuples);
    // golap::HostMem part_filled_hst(golap::Tag<uint64_t>(),1);
    // checkCudaErrors(cudaMemcpy(part_filled_hst.ptr<uint64_t>(), part_hj.filled, sizeof(uint64_t), cudaMemcpyDefault));
    // util::Log::get().info_fmt("In part hashmap %lu",part_filled_hst.ptr<uint64_t>()[0]);

    // util::Log::get().info_fmt("Supp tuples=%lu",supplier_dev.num_tuples);
    // golap::HostMem supplier_filled_hst(golap::Tag<uint64_t>(),1);
    // checkCudaErrors(cudaMemcpy(supplier_filled_hst.ptr<uint64_t>(), supplier_hj.filled, sizeof(uint64_t), cudaMemcpyDefault));
    // util::Log::get().info_fmt("In supplier hashmap %lu",supplier_filled_hst.ptr<uint64_t>()[0]);

    // util::Log::get().info_fmt("Date tuples=%lu",date_dev.num_tuples);
    // golap::HostMem date_filled_hst(golap::Tag<uint64_t>(),1);
    // checkCudaErrors(cudaMemcpy(date_filled_hst.ptr<uint64_t>(), date_hj.filled, sizeof(uint64_t), cudaMemcpyDefault));
    // util::Log::get().info_fmt("In date hashmap %lu",date_filled_hst.ptr<uint64_t>()[0]);
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

                tuples_this_round = env.blockenvs.at("lo_orderdate").myblocks[round].tuples;

                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                kernel<<<num_blocks,512,0,env.rootstream.stream>>>(part_hj,
                                                                                  supplier_hj,
                                                                                  date_hj,
                                                                                  hash_agg,
                                                                                  env.blockenvs.at("lo_partkey").decomp_buf.template ptr<PARTKEY_TYPE>(),
                                                                                  env.blockenvs.at("lo_suppkey").decomp_buf.template ptr<SUPPKEY_TYPE>(),
                                                                                  env.blockenvs.at("lo_orderdate").decomp_buf.template ptr<ORDERDATE_TYPE>(),
                                                                                  env.blockenvs.at("lo_revenue").decomp_buf.template ptr<REVENUE_TYPE>(),
                                                                                  part_dev.col<1>().data(),
                                                                                  date_dev.col<1>().data(),
                                                                                  SumAgg(),
                                                                                  tuples_this_round
                                                                                  ,&debug_aggs.dev.ptr<uint64_t>()[0]
                                                                                  // &debug_aggs.dev.ptr<uint64_t>()[1],
                                                                                  // &debug_aggs.dev.ptr<uint64_t>()[2],
                                                                                  // &debug_aggs.dev.ptr<uint64_t>()[3]
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

        // auto& group = groups.hst.ptr<YEARBRAND_GROUP>()[i];
        complete_sum += aggs.hst.ptr<uint64_t>()[i];;
        // util::Log::get().info_fmt("%lu\t%d\t%s",aggs.hst.ptr<uint64_t>()[i],group.d_year,group.p_brand1.d);
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

template <typename KERNEL>
bool query2_1_gen(SSBColLayout &ssbobj, KERNEL kernel){
    Q2PART_COLUMNS part_dev("p_key,p_brand1,p_category",ssbobj.tables.part.num_tuples);
    part_dev.num_tuples = ssbobj.tables.part.num_tuples;

    auto pred = golap::PredInfo<decltype(Part::p_category),PartPred1>{part_dev.col<2>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED") return query2<KERNEL,golap::LoadEnv,RegionAmericaPred>(ssbobj, kernel, part_dev, RegionAmericaPred(), pred);
    else return query2<KERNEL,golap::DecompressEnv,RegionAmericaPred>(ssbobj, kernel, part_dev, RegionAmericaPred(), pred);
}


template <typename KERNEL>
bool query2_2_gen(SSBColLayout &ssbobj, KERNEL kernel){
    Q2PART_COLUMNS part_dev("p_key,p_brand1,p_category",ssbobj.tables.part.num_tuples);
    part_dev.num_tuples = ssbobj.tables.part.num_tuples;

    auto pred = golap::PredInfo<decltype(Part::p_brand1),PartPred2>{part_dev.col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED") return query2<KERNEL,golap::LoadEnv,RegionAsiaPred>(ssbobj, kernel, part_dev, RegionAsiaPred(), pred);
    else return query2<KERNEL,golap::DecompressEnv,RegionAsiaPred>(ssbobj, kernel, part_dev, RegionAsiaPred(), pred);
}


template <typename KERNEL>
bool query2_3_gen(SSBColLayout &ssbobj, KERNEL kernel){
    Q2PART_COLUMNS part_dev("p_key,p_brand1,p_category",ssbobj.tables.part.num_tuples);
    part_dev.num_tuples = ssbobj.tables.part.num_tuples;

    auto pred = golap::PredInfo<decltype(Part::p_brand1),PartPred3>{part_dev.col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED") return query2<KERNEL,golap::LoadEnv,RegionEuropePred>(ssbobj, kernel, part_dev, RegionEuropePred(), pred);
    else return query2<KERNEL,golap::DecompressEnv,RegionEuropePred>(ssbobj, kernel, part_dev, RegionEuropePred(), pred);
}

bool SSBColLayout::query2_1(){return query2_1_gen(*this,pipeline_q2<Q2JO::PSD>);}
bool SSBColLayout::query2_2(){return query2_2_gen(*this,pipeline_q2<Q2JO::PSD>);}
bool SSBColLayout::query2_3(){return query2_3_gen(*this,pipeline_q2<Q2JO::PSD>);}
bool SSBColLayout::query2_1spd(){return query2_1_gen(*this,pipeline_q2<Q2JO::SPD>);}
bool SSBColLayout::query2_2spd(){return query2_2_gen(*this,pipeline_q2<Q2JO::SPD>);}
bool SSBColLayout::query2_3spd(){return query2_3_gen(*this,pipeline_q2<Q2JO::SPD>);}
bool SSBColLayout::query2_1sdp(){return query2_1_gen(*this,pipeline_q2<Q2JO::SDP>);}
bool SSBColLayout::query2_2sdp(){return query2_2_gen(*this,pipeline_q2<Q2JO::SDP>);}
bool SSBColLayout::query2_3sdp(){return query2_3_gen(*this,pipeline_q2<Q2JO::SDP>);}
bool SSBColLayout::query2_1dps(){return query2_1_gen(*this,pipeline_q2<Q2JO::DPS>);}
bool SSBColLayout::query2_2dps(){return query2_2_gen(*this,pipeline_q2<Q2JO::DPS>);}
bool SSBColLayout::query2_3dps(){return query2_3_gen(*this,pipeline_q2<Q2JO::DPS>);}
bool SSBColLayout::query2_1dsp(){return query2_1_gen(*this,pipeline_q2<Q2JO::DSP>);}
bool SSBColLayout::query2_2dsp(){return query2_2_gen(*this,pipeline_q2<Q2JO::DSP>);}
bool SSBColLayout::query2_3dsp(){return query2_3_gen(*this,pipeline_q2<Q2JO::DSP>);}