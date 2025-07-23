#pragma once
#include "query_common.cuh"
#include "../ssb.hpp"
#include "04.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "../data/SSB_def.hpp"

/*
select d_year, c_nation,
sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and (p_mfgr = 'MFGR#1'
or p_mfgr = 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation;

*/

using Q4DATE_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Date::d_key), decltype(Date::d_year)>;
using Q4SUPP_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Supplier::s_key), decltype(Supplier::s_nation), decltype(Supplier::s_region),decltype(Supplier::s_city)>;
using Q4CUST_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Customer::c_key), decltype(Customer::c_nation), decltype(Customer::c_region)>;
using Q4PART_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Part::p_key), decltype(Part::p_mfgr),
                                            decltype(Part::p_brand1),decltype(Part::p_category)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using CUSTKEY_TYPE = decltype(Lineorder::lo_custkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using PARTKEY_TYPE = decltype(Lineorder::lo_partkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);
using SUPPLYCOST_TYPE = decltype(Lineorder::lo_supplycost);


template <typename KERNEL, typename GROUP, typename BLOCK_ENV, typename CUST_PRED, typename SUPP_PRED, typename DATE_PRED, typename PART_PRED>
bool query_4(SSBColLayout &ssbobj, KERNEL kernel, Q4CUST_COLUMNS &customer_dev, Q4SUPP_COLUMNS &supplier_dev,
                          Q4DATE_COLUMNS &date_dev, Q4PART_COLUMNS &part_dev, CUST_PRED cust_pred, SUPP_PRED supp_pred, DATE_PRED date_pred,
                          PART_PRED part_pred){
    cudaSetDevice(ssbobj.var.cuda_device);

    uint64_t tuples_per_chunk = ssbobj.var.chunk_bytes / std::max({sizeof(ORDERDATE_TYPE),sizeof(CUSTKEY_TYPE),
                                                           sizeof(SUPPKEY_TYPE),sizeof(PARTKEY_TYPE),
                                                           sizeof(REVENUE_TYPE), sizeof(SUPPLYCOST_TYPE)});

    ssbobj.var.comp_bytes = 0;
    ssbobj.var.uncomp_bytes = 0;

    /**
    * - Write the chunk-compressed lineorder table to disk.
    */
    ssbobj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,uint64_t> columns{
        {"lo_orderdate", tuples_per_chunk},
        {"lo_partkey", tuples_per_chunk},
        {"lo_custkey", tuples_per_chunk},
        {"lo_suppkey", tuples_per_chunk},
        {"lo_revenue", tuples_per_chunk},
        {"lo_supplycost", tuples_per_chunk},
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

    uint64_t lo_column_bytes = 0;
    for (auto& comp_info : compinfos){
        lo_column_bytes += comp_info.second.get_comp_bytes();
    }
    util::Log::get().info_fmt("Lineorder columns bytes: %lu\t(%lu tuples)",lo_column_bytes,ssbobj.tables.lineorder.num_tuples);
    util::Log::get().info_fmt("Part      columns bytes: %lu\t(%lu tuples)",part_dev.size_bytes(), part_dev.num_tuples);
    util::Log::get().info_fmt("Customer  columns bytes: %lu\t(%lu tuples)",customer_dev.size_bytes(), customer_dev.num_tuples);
    util::Log::get().info_fmt("Supplier  columns bytes: %lu\t(%lu tuples)",supplier_dev.size_bytes(), supplier_dev.num_tuples);
    util::Log::get().info_fmt("Date      columns bytes: %lu\t\t(%lu tuples)",date_dev.size_bytes(), date_dev.num_tuples);


    // prepare hashmaps of the four joins
    golap::HashMap customer_hj(customer_dev.num_tuples,customer_dev.col<0>().data());
    golap::HashMap supplier_hj(supplier_dev.num_tuples,supplier_dev.col<0>().data());
    golap::HashMap date_hj(date_dev.num_tuples,date_dev.col<0>().data());
    golap::HashMap part_hj(part_dev.num_tuples,part_dev.col<0>().data());

    uint64_t num_groups = 1000;
    golap::MirrorMem groups(golap::Tag<GROUP>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    checkCudaErrors(cudaMemset(aggs.dev.ptr<uint8_t>(),0,aggs.dev.size_bytes()));
    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<GROUP>(), aggs.dev.ptr<uint64_t>());

    std::vector<golap::TableLoader<BLOCK_ENV>> envs;
    envs.reserve(ssbobj.var.workers);

    util::SliceSeq workslice(compinfos["lo_orderdate"].blocks.size(),ssbobj.var.workers);
    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_orderdate"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        // envs.emplace_back(startblock, endblock, compinfos["lo_orderdate"], compinfos["lo_custkey"], compinfos["lo_suppkey"],
                          // compinfos["lo_partkey"], compinfos["lo_revenue"], compinfos["lo_supplycost"]);
        envs.emplace_back(6);
        envs[pipeline_idx].add("lo_orderdate", all_blocks_idxs, startblock, endblock, compinfos["lo_orderdate"], nvcomp::TypeOf<ORDERDATE_TYPE>());
        envs[pipeline_idx].add("lo_custkey", all_blocks_idxs, startblock, endblock, compinfos["lo_custkey"], nvcomp::TypeOf<CUSTKEY_TYPE>());
        envs[pipeline_idx].add("lo_suppkey", all_blocks_idxs, startblock, endblock, compinfos["lo_suppkey"], nvcomp::TypeOf<SUPPKEY_TYPE>());
        envs[pipeline_idx].add("lo_partkey", all_blocks_idxs, startblock, endblock, compinfos["lo_partkey"], nvcomp::TypeOf<PARTKEY_TYPE>());
        envs[pipeline_idx].add("lo_revenue", all_blocks_idxs, startblock, endblock, compinfos["lo_revenue"], nvcomp::TypeOf<REVENUE_TYPE>());
        envs[pipeline_idx].add("lo_supplycost", all_blocks_idxs, startblock, endblock, compinfos["lo_supplycost"], nvcomp::TypeOf<SUPPLYCOST_TYPE>());
    }


    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    // misuse some streams ...
    auto& cust_build_stream = envs[0].blockenvs.at("lo_custkey").gstream.stream;
    auto& supp_build_stream = envs[1%ssbobj.var.workers].blockenvs.at("lo_suppkey").gstream.stream;
    auto& part_build_stream = envs[2%ssbobj.var.workers].blockenvs.at("lo_partkey").gstream.stream;
    auto& date_build_stream = envs[3%ssbobj.var.workers].blockenvs.at("lo_orderdate").gstream.stream;

    /**
     * Start the timer
     */
    util::Timer timer;

    ssbobj.tables.date.col<Date::KEY>().transfer(date_dev.col<0>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    ssbobj.tables.date.col<Date::YEAR>().transfer(date_dev.col<1>().data(), ssbobj.tables.date.num_tuples, date_build_stream);
    ssbobj.tables.supplier.col<Supplier::KEY>().transfer(supplier_dev.col<0>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::NATION>().transfer(supplier_dev.col<1>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::REGION>().transfer(supplier_dev.col<2>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.supplier.col<Supplier::CITY>().transfer(supplier_dev.col<3>().data(), ssbobj.tables.supplier.num_tuples, supp_build_stream);
    ssbobj.tables.customer.col<Customer::KEY>().transfer(customer_dev.col<0>().data(), ssbobj.tables.customer.num_tuples, cust_build_stream);
    ssbobj.tables.customer.col<Customer::NATION>().transfer(customer_dev.col<1>().data(), ssbobj.tables.customer.num_tuples, cust_build_stream);
    ssbobj.tables.customer.col<Customer::REGION>().transfer(customer_dev.col<2>().data(), ssbobj.tables.customer.num_tuples, cust_build_stream);
    ssbobj.tables.part.col<Part::KEY>().transfer(part_dev.col<0>().data(), ssbobj.tables.part.num_tuples, part_build_stream);
    ssbobj.tables.part.col<Part::MFGR>().transfer(part_dev.col<1>().data(), ssbobj.tables.part.num_tuples, part_build_stream);
    ssbobj.tables.part.col<Part::BRAND1>().transfer(part_dev.col<2>().data(), ssbobj.tables.part.num_tuples, part_build_stream);
    ssbobj.tables.part.col<Part::CATEGORY>().transfer(part_dev.col<3>().data(), ssbobj.tables.part.num_tuples, part_build_stream);

    golap::hash_map_build_pred_other<<<util::div_ceil(customer_dev.num_tuples,512),512,0,cust_build_stream>>>(
                                            customer_hj, customer_dev.num_tuples,
                                            golap::DirectKey<decltype(Customer::c_key)>(), cust_pred);

    golap::hash_map_build_pred_other<<<util::div_ceil(supplier_dev.num_tuples,512),512,0,supp_build_stream>>>(
                                            supplier_hj,supplier_dev.num_tuples,
                                            golap::DirectKey<decltype(Supplier::s_key)>(), supp_pred);

    golap::hash_map_build_pred_other<<<util::div_ceil(part_dev.num_tuples,512),512,0,part_build_stream>>>(
                                            part_hj,part_dev.num_tuples,
                                            golap::DirectKey<decltype(Part::p_key)>(), part_pred);

    golap::hash_map_build_pred_other<<<util::div_ceil(date_dev.num_tuples,512),512,0,date_build_stream>>>(
                                            date_hj,date_dev.num_tuples,
                                            golap::DirectKey<decltype(Date::d_key)>(), date_pred);

    checkCudaErrors(cudaStreamSynchronize(cust_build_stream));
    checkCudaErrors(cudaStreamSynchronize(supp_build_stream));
    checkCudaErrors(cudaStreamSynchronize(part_build_stream));
    checkCudaErrors(cudaStreamSynchronize(date_build_stream));


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
                kernel<<<num_blocks,512,0,env.rootstream.stream>>>(customer_hj,
                                                                      supplier_hj,
                                                                      date_hj,
                                                                      part_hj,
                                                                      hash_agg,
                                                                      env.blockenvs.at("lo_custkey").decomp_buf.template ptr<CUSTKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_suppkey").decomp_buf.template ptr<SUPPKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_orderdate").decomp_buf.template ptr<ORDERDATE_TYPE>(),
                                                                      env.blockenvs.at("lo_partkey").decomp_buf.template ptr<PARTKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_revenue").decomp_buf.template ptr<REVENUE_TYPE>(),
                                                                      env.blockenvs.at("lo_supplycost").decomp_buf.template ptr<SUPPLYCOST_TYPE>(),
                                                                      customer_dev.col<1>().data(), // customer nation
                                                                      customer_dev.col<2>().data(), // customer region
                                                                      supplier_dev.col<1>().data(), // supplier nation
                                                                      supplier_dev.col<2>().data(), // supplier region
                                                                      supplier_dev.col<3>().data(), // supplier city
                                                                      date_dev.col<1>().data(), // date year
                                                                      part_dev.col<2>().data(), // part brand1
                                                                      part_dev.col<3>().data(), // part category
                                                                      SumAgg(),
                                                                      tuples_this_round
                                                                      // ,&debug_aggs.dev.ptr<double>()[0],
                                                                      // &debug_aggs.dev.ptr<double>()[1]
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

    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, num_groups};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, num_groups*sizeof(uint32_t), cudaMemcpyDefault));

    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;
        num_actual_groups += 1;

        // auto& group = groups.hst.ptr<GROUP>()[i];
        complete_sum += aggs.hst.ptr<uint64_t>()[i];
        // std::cout << group <<"," << aggs.hst.ptr<uint64_t>()[i] <<"\n";
    }

    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);

    return true;
}

std::shared_ptr<Q4DATE_COLUMNS> prepareQ4DATE(SSBColLayout &ssbobj){
    // prepare date columns
    auto date_dev = std::make_shared<Q4DATE_COLUMNS>("d_key,d_year",ssbobj.tables.date.num_tuples);
    date_dev->num_tuples = ssbobj.tables.date.num_tuples;
    return date_dev;
}

std::shared_ptr<Q4SUPP_COLUMNS> prepareQ4SUPP(SSBColLayout &ssbobj){
    // prepare supplier columns
    auto supplier_dev = std::make_shared<Q4SUPP_COLUMNS>("s_key,s_nation,s_region,s_city",ssbobj.tables.supplier.num_tuples);
    supplier_dev->num_tuples = ssbobj.tables.supplier.num_tuples;
    return supplier_dev;
}

std::shared_ptr<Q4CUST_COLUMNS> prepareQ4CUST(SSBColLayout &ssbobj){
    // prepare customer columns
    auto customer_dev = std::make_shared<Q4CUST_COLUMNS>("c_key,c_nation,c_region", ssbobj.tables.customer.num_tuples);
    customer_dev->num_tuples = ssbobj.tables.customer.num_tuples;
    return customer_dev;
}

std::shared_ptr<Q4PART_COLUMNS> prepareQ4PART(SSBColLayout &ssbobj){
    // prepare part columns
    auto part_dev = std::make_shared<Q4PART_COLUMNS>("p_key,p_mfgr,p_brand1,p_category", ssbobj.tables.part.num_tuples);
    part_dev->num_tuples = ssbobj.tables.part.num_tuples;
    return part_dev;

}

template <typename KERNEL, typename GROUP>
bool query4_1_gen(SSBColLayout &ssbobj, KERNEL kernel){

    auto date_dev = prepareQ4DATE(ssbobj);
    auto supplier_dev = prepareQ4SUPP(ssbobj);
    auto customer_dev = prepareQ4CUST(ssbobj);
    auto part_dev = prepareQ4PART(ssbobj);

    golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred> cust_pred{customer_dev->col<2>().data()};
    golap::PredInfo<decltype(Supplier::s_region),RegionAmericaPred> supp_pred{supplier_dev->col<2>().data()};
    golap::PredInfo<char,TruePred> date_pred{nullptr};
    golap::PredInfo<decltype(Part::p_mfgr),MFGRPred> part_pred{part_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query_4<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);
    }else return query_4<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);

}

template <typename KERNEL, typename GROUP>
bool query4_2_gen(SSBColLayout &ssbobj, KERNEL kernel){

    auto date_dev = prepareQ4DATE(ssbobj);
    auto supplier_dev = prepareQ4SUPP(ssbobj);
    auto customer_dev = prepareQ4CUST(ssbobj);
    auto part_dev = prepareQ4PART(ssbobj);

    golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred> cust_pred{customer_dev->col<2>().data()};
    golap::PredInfo<decltype(Supplier::s_region),RegionAmericaPred> supp_pred{supplier_dev->col<2>().data()};
    golap::PredInfo<decltype(Date::d_year),DATE_9798> date_pred{date_dev->col<1>().data()};
    golap::PredInfo<decltype(Part::p_mfgr),MFGRPred> part_pred{part_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query_4<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);
    }else return query_4<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);


}

template <typename KERNEL, typename GROUP>
bool query4_3_gen(SSBColLayout &ssbobj, KERNEL kernel){

    auto date_dev = prepareQ4DATE(ssbobj);
    auto supplier_dev = prepareQ4SUPP(ssbobj);
    auto customer_dev = prepareQ4CUST(ssbobj);
    auto part_dev = prepareQ4PART(ssbobj);

    golap::PredInfo<decltype(Customer::c_region),RegionAmericaPred> cust_pred{customer_dev->col<2>().data()};
    golap::PredInfo<decltype(Supplier::s_nation),NationUSPred> supp_pred{supplier_dev->col<1>().data()};
    golap::PredInfo<decltype(Date::d_year),DATE_9798> date_pred{date_dev->col<1>().data()};
    golap::PredInfo<decltype(Part::p_category),Category14Pred> part_pred{part_dev->col<3>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query_4<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);
    }else return query_4<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *customer_dev, *supplier_dev, *date_dev, *part_dev, cust_pred, supp_pred, date_pred, part_pred);

}

bool SSBColLayout::query4_1(){
    return query4_1_gen<decltype(pipeline_q4<Q41_GROUP>),Q41_GROUP>(*this,pipeline_q4<Q41_GROUP>);
}
bool SSBColLayout::query4_2(){
    return query4_2_gen<decltype(pipeline_q4<Q42_GROUP>),Q42_GROUP>(*this,pipeline_q4<Q42_GROUP>);
}
bool SSBColLayout::query4_3(){
    return query4_3_gen<decltype(pipeline_q4<Q43_GROUP>),Q43_GROUP>(*this,pipeline_q4<Q43_GROUP>);
}


