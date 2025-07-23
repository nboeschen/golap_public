#pragma once
#include <any>
#include "query_common.cuh"
#include "../ssb.hpp"
#include "03.cuh"

#include "util.hpp"
#include "join.cuh"
#include "apps.cuh"
#include "metadata.cuh"
#include "../data/SSB_def.hpp"
#include "../helper.hpp"

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


*/

using Q3DATE_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Date::d_key), decltype(Date::d_year), decltype(Date::d_yearmonth)>;
using Q3SUPP_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Supplier::s_key), decltype(Supplier::s_nation), decltype(Supplier::s_region),decltype(Supplier::s_city)>;
using Q3CUST_COLUMNS = golap::ColumnTable<golap::DeviceMem,decltype(Customer::c_key), decltype(Customer::c_nation), decltype(Customer::c_region),decltype(Customer::c_city)>;

using ORDERDATE_TYPE = decltype(Lineorder::lo_orderdate);
using CUSTKEY_TYPE = decltype(Lineorder::lo_custkey);
using SUPPKEY_TYPE = decltype(Lineorder::lo_suppkey);
using REVENUE_TYPE = decltype(Lineorder::lo_revenue);

using DYEAR_TYPE = decltype(Date::d_year);
using DYEARMONTH_TYPE = decltype(Date::d_yearmonth);

using CKEY_TYPE = decltype(Customer::c_key);
using CCITY_TYPE = decltype(Customer::c_city);
using CNATION_TYPE = decltype(Customer::c_nation);
using CREGION_TYPE = decltype(Customer::c_region);

template <typename KERNEL, typename GROUP, typename BLOCK_ENV, typename SUPP_PRED, typename DATE_PRED>
bool query3(SSBColLayout &ssbobj, KERNEL kernel, Q3SUPP_COLUMNS &supplier_dev,
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
    // colname,tuples_per_chunk, metadataonly
    const std::unordered_map<std::string,std::tuple<uint64_t,golap::MetaFlags>> columns{
        {"lo_orderdate",{lo_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"lo_custkey",{lo_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"lo_suppkey",{lo_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"lo_revenue",{lo_tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}},
        {"c_key",{cust_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"c_city",{cust_tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}},
        {"c_nation",{cust_tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}},
        {"c_region",{cust_tuples_per_chunk,golap::MetaFlags::DATA | golap::MetaFlags::META}},
        {"meta_d_year",{lo_tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_d_yearmonth",{lo_tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_c_city",{lo_tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_c_nation",{lo_tuples_per_chunk,golap::MetaFlags::META}},
        {"meta_c_region",{lo_tuples_per_chunk,golap::MetaFlags::META}},
    };

    std::unordered_map<std::string, golap::CompInfo> compinfos;
    std::unordered_map<std::string, std::any> minmaxmeta;
    // std::unordered_map<std::string, std::any> histmeta;
    std::unordered_map<std::string, std::any> bloommeta;

    compinfos.reserve(columns.size());
    minmaxmeta.reserve(columns.size());
    // histmeta.reserve(columns.size());
    bloommeta.reserve(columns.size());

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

            // histmeta.try_emplace(a_col.attr_name, std::in_place_type<golap::EqHistogram<COL_TYPE>>, ssbobj.var.pruning_param);
            // histptr = &std::any_cast<golap::EqHistogram<COL_TYPE>&>(histmeta[a_col.attr_name]);

            bloommeta.try_emplace(a_col.attr_name, std::in_place_type<golap::BloomMeta<COL_TYPE>>, ssbobj.var.pruning_p, ssbobj.var.pruning_m);
            bloomptr = &std::any_cast<golap::BloomMeta<COL_TYPE>&>(bloommeta[a_col.attr_name]);
        }

        util::Log::get().debug_fmt("Compressing col %lu=>%s to disk, algo=%s, offset=%lu",col_idx,a_col.attr_name.c_str(),algo.c_str());
        if constexpr (std::is_same_v<BLOCK_ENV,golap::DecompressEnv>){
            ssbobj.var.comp_ms += golap::prepare_compressed_device(a_col, num_tuples, compinfos[a_col.attr_name], minmaxptr, histptr, bloomptr);
        }else{
            ssbobj.var.comp_ms += golap::prepare_uncompressed(a_col, num_tuples, compinfos[a_col.attr_name], minmaxptr, histptr, bloomptr);
        }

        // copy metadata to host, so that we can copy it back to device if needed
        if (usage & golap::MetaFlags::META){
            std::any_cast<golap::MinMaxMeta<COL_TYPE>&>(minmaxmeta[a_col.attr_name]).to_host();
            // std::any_cast<golap::EqHistogram<COL_TYPE>&>(histmeta[a_col.attr_name]).to_host();
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
    ssbobj.tables.customer.apply(compress_columns_fun);
    SortHelper::get().prejoined->apply(compress_columns_fun);

    // empty enterprise ssd cache by writing to different offset
    // golap::HostMem OneGB_empty{golap::Tag<char>{},(1<<30)};
    // uint64_t write_dummy_offset = 500l*(1<<30);
    // for (int write_dummy = 0; write_dummy<20;++write_dummy){
    //     golap::StorageManager::get().host_write_bytes(OneGB_empty.ptr<char>(), OneGB_empty.size_bytes(), write_dummy_offset);
    //     write_dummy_offset += OneGB_empty.size_bytes();
    // }

    uint64_t lo_column_bytes = 0,cust_column_bytes=0;
    for (auto& comp_info : compinfos){
        if (comp_info.first.find("lo_")==0) lo_column_bytes += comp_info.second.get_comp_bytes();
        else if (comp_info.first.find("c_")==0) cust_column_bytes += comp_info.second.get_comp_bytes();
    }
    util::Log::get().info_fmt("Lineorder columns bytes: %lu\t(%lu tuples)",lo_column_bytes,ssbobj.tables.lineorder.num_tuples);
    util::Log::get().info_fmt("Customer  columns bytes: %lu\t(%lu tuples)",cust_column_bytes,ssbobj.tables.customer.num_tuples);
    util::Log::get().info_fmt("Supplier  columns bytes: %lu\t(%lu tuples)",supplier_dev.size_bytes(), supplier_dev.num_tuples);
    util::Log::get().info_fmt("Date      columns bytes: %lu\t\t(%lu tuples)",date_dev.size_bytes(), date_dev.num_tuples);


    golap::DeviceMem customer_dev{golap::Tag<Q3Customer>{}, ssbobj.tables.customer.num_tuples};
    golap::DeviceMem customer_count{golap::Tag<uint64_t>{}, 1};
    customer_count.set(0);

    // prepare hashmaps of the three joins
    golap::HashMap customer_hj(ssbobj.tables.customer.num_tuples,customer_dev.ptr<Q3Customer>());
    golap::HashMap supplier_hj(supplier_dev.num_tuples,supplier_dev.col<0>().data());
    golap::HashMap date_hj(date_dev.num_tuples,date_dev.col<0>().data());

    uint64_t num_groups = 800;
    golap::MirrorMem groups(golap::Tag<GROUP>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    checkCudaErrors(cudaMemset(aggs.dev.ptr<uint8_t>(),0,aggs.dev.size_bytes()));
    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<GROUP>(), aggs.dev.ptr<uint64_t>());

    golap::MirrorMem debug_aggs(golap::Tag<double>{},5);
    checkCudaErrors(cudaMemset(debug_aggs.dev.ptr<uint8_t>(),0,debug_aggs.dev.size_bytes()));

    // pruning related
    golap::MirrorMem tmp_lineorder_check{golap::Tag<uint16_t>{}, compinfos["lo_orderdate"].blocks.size()};
    golap::MirrorMem tmp_customer_check{golap::Tag<uint16_t>{}, compinfos["c_key"].blocks.size()};
    golap::MirrorMem combined_check_lineorder{golap::Tag<uint16_t>{}, compinfos["lo_orderdate"].blocks.size()};
    golap::MirrorMem combined_check_customer{golap::Tag<uint16_t>{}, compinfos["c_key"].blocks.size()};
    tmp_lineorder_check.dev.set(0);
    tmp_customer_check.dev.set(0);
    combined_check_lineorder.dev.set(0);
    combined_check_customer.dev.set(0);
    std::atomic<uint64_t> pruned_bytes{0};
    std::atomic<uint64_t> pruned_chunks{0};

    std::vector<golap::TableLoader<BLOCK_ENV>> lo_envs;
    lo_envs.reserve(ssbobj.var.workers);

    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["lo_orderdate"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    util::SliceSeq workslice(compinfos["lo_orderdate"].blocks.size(), ssbobj.var.workers);
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        lo_envs.emplace_back(4);
        lo_envs[pipeline_idx].add("lo_orderdate", all_blocks_idxs, startblock, endblock, compinfos["lo_orderdate"], nvcomp::TypeOf<ORDERDATE_TYPE>());
        lo_envs[pipeline_idx].add("lo_custkey", all_blocks_idxs, startblock, endblock, compinfos["lo_custkey"], nvcomp::TypeOf<CUSTKEY_TYPE>());
        lo_envs[pipeline_idx].add("lo_suppkey", all_blocks_idxs, startblock, endblock, compinfos["lo_suppkey"], nvcomp::TypeOf<SUPPKEY_TYPE>());
        lo_envs[pipeline_idx].add("lo_revenue", all_blocks_idxs, startblock, endblock, compinfos["lo_revenue"], nvcomp::TypeOf<REVENUE_TYPE>());
    }


    std::vector<golap::TableLoader<BLOCK_ENV>> customer_envs;
    customer_envs.reserve(ssbobj.var.extra_workers);
    all_blocks_idxs.resize(compinfos["c_nation"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    util::SliceSeq cworkslice(compinfos["c_nation"].blocks.size(), ssbobj.var.extra_workers);
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.extra_workers; ++pipeline_idx){
        // prepare environment for each thread
        cworkslice.get(startblock,endblock);

        customer_envs.emplace_back(4);
        customer_envs[pipeline_idx].add("c_key", all_blocks_idxs, startblock, endblock, compinfos["c_key"], nvcomp::TypeOf<CKEY_TYPE>());
        customer_envs[pipeline_idx].add("c_city", all_blocks_idxs, startblock, endblock, compinfos["c_city"], nvcomp::TypeOf<uint8_t>());
        customer_envs[pipeline_idx].add("c_nation", all_blocks_idxs, startblock, endblock, compinfos["c_nation"], nvcomp::TypeOf<uint8_t>());
        customer_envs[pipeline_idx].add("c_region", all_blocks_idxs, startblock, endblock, compinfos["c_region"], nvcomp::TypeOf<uint8_t>());
    }


    std::vector<std::thread> threads;
    threads.reserve(ssbobj.var.workers);
    std::vector<std::thread> extra_p_threads;
    extra_p_threads.reserve(ssbobj.var.extra_workers);

    double customer_time,lo_time;
    // misuse some streams ...
    auto& cust_build_stream = lo_envs[0].blockenvs.at("lo_custkey").gstream.stream; // not used
    auto& meta_stream = lo_envs[0].blockenvs.at("lo_custkey").gstream.stream;
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

    // golap::hash_map_build_pred_other<<<util::div_ceil(customer_dev.num_tuples,512),512,0,cust_build_stream>>>(
    //                                         customer_hj, customer_dev.num_tuples,
    //                                         golap::DirectKey<decltype(Customer::c_key)>(), cust_pred);

    uint64_t block_num = std::min((long long)ssbobj.var.block_limit, util::div_ceil(compinfos["lo_orderdate"].blocks.size(),512));
    golap::MetaChecker lineorderchecker{block_num, compinfos["lo_orderdate"].blocks.size(), combined_check_lineorder.dev.ptr<uint16_t>(), tmp_lineorder_check.dev.ptr<uint16_t>(), meta_stream};
    golap::MetaChecker customerchecker{block_num, compinfos["c_key"].blocks.size(), combined_check_customer.dev.ptr<uint16_t>(), tmp_customer_check.dev.ptr<uint16_t>(), meta_stream};
    util::Timer prune_timer;

    if (ssbobj.var.pruning == "MINMAX"){
        if (ssbobj.var.query == "query3.1"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_minmax<CREGION_TYPE>(minmaxmeta["meta_c_region"], CREGION_TYPE("ASIA"), CREGION_TYPE("ASIA"));
            // customer
            customerchecker.add_minmax<CREGION_TYPE>(minmaxmeta["c_region"], CREGION_TYPE("ASIA"), CREGION_TYPE("ASIA"));
        }else if(ssbobj.var.query == "query3.2"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_minmax<CNATION_TYPE>(minmaxmeta["meta_c_nation"], CNATION_TYPE("UNITED STATES"), CNATION_TYPE("UNITED STATES"));
            // customer
            customerchecker.add_minmax<CNATION_TYPE>(minmaxmeta["c_nation"], CNATION_TYPE("UNITED STATES"), CNATION_TYPE("UNITED STATES"));
        }else if(ssbobj.var.query == "query3.3"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_minmax<CCITY_TYPE>(minmaxmeta["meta_c_city"], CCITY_TYPE("UNITED KI1"), CCITY_TYPE("UNITED KI5"));
            // customer
            customerchecker.add_minmax<CCITY_TYPE>(minmaxmeta["c_city"], CCITY_TYPE("UNITED KI1"), CCITY_TYPE("UNITED KI5"));
        }else if(ssbobj.var.query == "query3.4"){
            lineorderchecker.add_minmax<DYEARMONTH_TYPE>(minmaxmeta["meta_d_yearmonth"], DYEARMONTH_TYPE("Dec1997"), DYEARMONTH_TYPE("Dec1997"));
            lineorderchecker.add_minmax<CCITY_TYPE>(minmaxmeta["meta_c_city"], CCITY_TYPE("UNITED KI1"), CCITY_TYPE("UNITED KI5"));
            // customer
            customerchecker.add_minmax<CCITY_TYPE>(minmaxmeta["c_city"], CCITY_TYPE("UNITED KI1"), CCITY_TYPE("UNITED KI5"));
        }
    }else if (ssbobj.var.pruning == "HIST"){
        combined_check_lineorder.dev.set(1, meta_stream);
        combined_check_customer.dev.set(1, meta_stream);
    }else if (ssbobj.var.pruning == "COMBINED"){
        if (ssbobj.var.query == "query3.1"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_bloom<CREGION_TYPE>(bloommeta["meta_c_region"], CREGION_TYPE("ASIA"));
            // customer
            customerchecker.add_bloom<CREGION_TYPE>(bloommeta["c_region"], CREGION_TYPE("ASIA"));
        }else if(ssbobj.var.query == "query3.2"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_bloom<CNATION_TYPE>(bloommeta["meta_c_nation"], CNATION_TYPE("UNITED STATES"));
            // customer
            customerchecker.add_bloom<CNATION_TYPE>(bloommeta["c_nation"], CNATION_TYPE("UNITED STATES"));
        }else if(ssbobj.var.query == "query3.3"){
            lineorderchecker.add_minmax<DYEAR_TYPE>(minmaxmeta["meta_d_year"], DYEAR_TYPE(1992), DYEAR_TYPE(1997));
            lineorderchecker.add_bloom<CCITY_TYPE>(bloommeta["meta_c_city"], CCITY_TYPE("UNITED KI1"));
            // customer
            customerchecker.add_bloom<CCITY_TYPE>(bloommeta["c_city"], CCITY_TYPE("UNITED KI1"));
        }else if(ssbobj.var.query == "query3.4"){
            lineorderchecker.add_minmax<DYEARMONTH_TYPE>(minmaxmeta["meta_d_yearmonth"], DYEARMONTH_TYPE("Dec1997"), DYEARMONTH_TYPE("Dec1997"));
            lineorderchecker.add_bloom<CCITY_TYPE>(bloommeta["meta_c_city"], CCITY_TYPE("UNITED KI1"));
            // customer
            customerchecker.add_bloom<CCITY_TYPE>(bloommeta["c_city"], CCITY_TYPE("UNITED KI1"));
        }
    }else if (ssbobj.var.pruning == "BLOOM"){
        if (ssbobj.var.query == "query3.1"){
            lineorderchecker.add_bloom<CREGION_TYPE>(bloommeta["meta_c_region"], CREGION_TYPE("ASIA"));
            // customer
            customerchecker.add_bloom<CREGION_TYPE>(bloommeta["c_region"], CREGION_TYPE("ASIA"));
        }else if(ssbobj.var.query == "query3.2"){
            lineorderchecker.add_bloom<CNATION_TYPE>(bloommeta["meta_c_nation"], CNATION_TYPE("UNITED STATES"));
            // customer
            customerchecker.add_bloom<CNATION_TYPE>(bloommeta["c_nation"], CNATION_TYPE("UNITED STATES"));
        }else if(ssbobj.var.query == "query3.3"){
            lineorderchecker.add_bloom<CCITY_TYPE>(bloommeta["meta_c_city"], CCITY_TYPE("UNITED KI1"));
            // customer
            customerchecker.add_bloom<CCITY_TYPE>(bloommeta["c_city"], CCITY_TYPE("UNITED KI1"));
        }else if(ssbobj.var.query == "query3.4"){
            lineorderchecker.add_bloom<DYEARMONTH_TYPE>(bloommeta["meta_d_yearmonth"], DYEARMONTH_TYPE("Dec1997"));
            lineorderchecker.add_bloom<CCITY_TYPE>(bloommeta["meta_c_city"], CCITY_TYPE("UNITED KI1"));
            // customer
            customerchecker.add_bloom<CCITY_TYPE>(bloommeta["c_city"], CCITY_TYPE("UNITED KI1"));
        }
    }

    if (ssbobj.var.pruning != "DONTPRUNE"){
        combined_check_lineorder.sync_to_host(meta_stream);
        combined_check_customer.sync_to_host(meta_stream);
    }
    checkCudaErrors(cudaStreamSynchronize(meta_stream));
    ssbobj.var.prune_ms = prune_timer.elapsed();


    util::Timer customer_timer;
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.extra_workers; ++pipeline_idx){
        extra_p_threads.emplace_back([&,pipeline_idx]{
            cudaSetDevice(ssbobj.var.cuda_device);

            auto &env = customer_envs[pipeline_idx];

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
                }
                // https://forums.developer.nvidia.com/t/nvc-22-1-nonsensical-warning-about-missing-return-statement/202358/7
                __builtin_unreachable();
            }();

            uint64_t round = 0;
            uint64_t tuples_this_round,global_block_idx;
            uint32_t num_blocks;
            while(round< env.blockenvs.at("c_key").myblocks.size()){
            // while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){

                global_block_idx = env.blockenvs.at("c_key").myblock_idxs[round];

                if (ssbobj.var.pruning != "DONTPRUNE" && combined_check_customer.hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    // util::Log::get().info_fmt("Thread[%lu, round%lu] would skip the next customer chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);
                    pruned_bytes.fetch_add(env.blockenvs.at("c_key").myblocks[round].size
                                          +env.blockenvs.at("c_city").myblocks[round].size
                                          +env.blockenvs.at("c_nation").myblocks[round].size
                                          +env.blockenvs.at("c_region").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }

                if (!env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

                tuples_this_round = env.blockenvs.at("c_key").myblocks[round].tuples;


                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                pipeline_customer_q3<<<num_blocks,512,0,env.rootstream.stream>>>(customer_hj,
                                                                            env.blockenvs.at("c_key").decomp_buf.template ptr<CKEY_TYPE>(),
                                                                            env.blockenvs.at("c_city").decomp_buf.template ptr<CCITY_TYPE>(),
                                                                            env.blockenvs.at("c_nation").decomp_buf.template ptr<CNATION_TYPE>(),
                                                                            env.blockenvs.at("c_region").decomp_buf.template ptr<CREGION_TYPE>(),
                                                                            cust_pred,
                                                                            customer_dev.ptr<Q3Customer>(),
                                                                            customer_count.ptr<uint64_t>(),
                                                                            tuples_this_round
                                                                      );
                // util::Log::get().info_fmt("pipeline[%lu]: Tuple this round: %lu", pipeline_idx, tuples_this_round);
                round += 1;
                checkCudaErrors(cudaEventRecord(env.rootevent.event, env.rootstream.stream));
            } // end of while

        });
    }
    for(auto &thread: extra_p_threads) thread.join();
    // util::Log::get().info_fmt("Customer HT on device took %2.fms",customer_timer.elapsed());
    customer_time = customer_timer.elapsed();


    checkCudaErrors(cudaStreamSynchronize(date_build_stream));
    checkCudaErrors(cudaStreamSynchronize(supp_build_stream));
    // checkCudaErrors(cudaStreamSynchronize(cust_build_stream));

    util::Timer lo_timer;
    // start the main pipeline
    for (uint32_t pipeline_idx=0; pipeline_idx<ssbobj.var.workers; ++pipeline_idx){
        threads.emplace_back([&,pipeline_idx]{

            cudaSetDevice(ssbobj.var.cuda_device);
            auto &env = lo_envs[pipeline_idx];

            uint64_t round = 0;
            uint64_t tuples_this_round,global_block_idx;
            uint32_t num_blocks;
            // while(env.rootop.step(env.rootstream.stream,env.rootevent.event)){
            while(round< env.blockenvs.at("lo_orderdate").myblocks.size()){

                global_block_idx = env.blockenvs.at("lo_orderdate").myblock_idxs[round];

                if (ssbobj.var.pruning != "DONTPRUNE" && combined_check_lineorder.hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    // util::Log::get().info_fmt("Thread[%lu, round%lu] would skip the next lineorder chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);
                    pruned_bytes.fetch_add(env.blockenvs.at("lo_orderdate").myblocks[round].size
                                          +env.blockenvs.at("lo_custkey").myblocks[round].size
                                          +env.blockenvs.at("lo_suppkey").myblocks[round].size
                                          +env.blockenvs.at("lo_revenue").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }

                if (!env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));

                tuples_this_round = env.blockenvs.at("lo_orderdate").myblocks[round].tuples;

                num_blocks = std::min((long long)ssbobj.var.block_limit, util::div_ceil(tuples_this_round,512));
                kernel<<<num_blocks,512,0,env.rootstream.stream>>>(customer_hj,
                                                                      supplier_hj,
                                                                      date_hj,
                                                                      hash_agg,
                                                                      env.blockenvs.at("lo_custkey").decomp_buf.template ptr<CUSTKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_suppkey").decomp_buf.template ptr<SUPPKEY_TYPE>(),
                                                                      env.blockenvs.at("lo_orderdate").decomp_buf.template ptr<ORDERDATE_TYPE>(),
                                                                      env.blockenvs.at("lo_revenue").decomp_buf.template ptr<REVENUE_TYPE>(),
                                                                      // customer_dev.col<1>().data(), // customer nation
                                                                      // customer_dev.col<3>().data(), // customer city
                                                                      supplier_dev.col<1>().data(), // supplier nation
                                                                      supplier_dev.col<3>().data(), // supplier city
                                                                      date_dev.col<1>().data(), // date year
                                                                      customer_dev.ptr<Q3Customer>(),
                                                                      SumAgg(),
                                                                      tuples_this_round
                                                                      ,&debug_aggs.dev.ptr<double>()[0],
                                                                      &debug_aggs.dev.ptr<double>()[1]
                                                                      // &debug_aggs.dev.ptr<uint64_t>()[2],
                                                                      // &debug_aggs.dev.ptr<uint64_t>()[3]
                                                                      );
                round += 1;
                checkCudaErrors(cudaEventRecord(env.rootevent.event, env.rootstream.stream));
            } // end of while
            checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));
        });
    }

    for(auto &thread: threads) thread.join();
    aggs.sync_to_host(lo_envs[0].rootstream.stream);
    groups.sync_to_host(lo_envs[0].rootstream.stream);
    checkCudaErrors(cudaStreamSynchronize(lo_envs[0].rootstream.stream));
    lo_time = lo_timer.elapsed();
    ssbobj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */

    debug_aggs.sync_to_host();
    ssbobj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    ssbobj.var.host_mem_used = golap::HOST_ALLOCATED.load();
    ssbobj.var.pruned_bytes = pruned_bytes.load();

    ssbobj.var.debug_0 = std::to_string(customer_time);
    ssbobj.var.debug_1 = std::to_string(lo_time);

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, num_groups};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, num_groups*sizeof(uint32_t), cudaMemcpyDefault));

    uint64_t num_actual_groups = 0;
    uint64_t complete_sum = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;
        num_actual_groups += 1;

        auto& group = groups.hst.ptr<GROUP>()[i];
        complete_sum += aggs.hst.ptr<uint64_t>()[i];
        // std::cout << group <<"," << aggs.hst.ptr<uint64_t>()[i] <<"\n";
    }


    // util::Log::get().info_fmt("Join cycles: %f",debug_aggs.hst.ptr<double>()[0]);
    // util::Log::get().info_fmt("Agg  cycles: %f",debug_aggs.hst.ptr<double>()[1]);

    util::Log::get().info_fmt("Sum of results            =%lu",complete_sum);
    util::Log::get().info_fmt("Number of results (groups)=%lu",num_actual_groups);
    util::Log::get().info_fmt("Pruned: %lu of %lu chunks (%.2f), %lu of %lu bytes (%.2f)",
                                pruned_chunks.load(),(compinfos["lo_orderdate"].blocks.size()+compinfos["c_key"].blocks.size()),
                                (double)pruned_chunks.load()/(compinfos["lo_orderdate"].blocks.size()+compinfos["c_key"].blocks.size()),
                                pruned_bytes.load(),ssbobj.var.comp_bytes,(double)pruned_bytes.load()/ssbobj.var.comp_bytes);
    return true;
}


std::shared_ptr<Q3DATE_COLUMNS> prepareQ3DATE(SSBColLayout &ssbobj){
    // copy date columns to device (uncompressed for now)
    auto date_dev = std::make_shared<Q3DATE_COLUMNS>("d_key,d_year,d_yearmonth",ssbobj.tables.date.num_tuples);
    date_dev->num_tuples = ssbobj.tables.date.num_tuples;
    return date_dev;
}

std::shared_ptr<Q3SUPP_COLUMNS> prepareQ3SUPP(SSBColLayout &ssbobj){
    // copy supplier columns to device (uncompressed for now)
    auto supplier_dev = std::make_shared<Q3SUPP_COLUMNS>("s_key,s_nation,s_region,s_city",ssbobj.tables.supplier.num_tuples);
    supplier_dev->num_tuples = ssbobj.tables.supplier.num_tuples;
    return supplier_dev;
}


template <typename KERNEL, typename GROUP>
bool query3_1_gen(SSBColLayout &ssbobj, KERNEL kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    // golap::PredInfo<decltype(Customer::c_region),RegionAsiaPred> cust_pred{customer_dev->col<2>().data()};
    golap::PredInfo<decltype(Supplier::s_region),RegionAsiaPred> supp_pred{supplier_dev->col<2>().data()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename GROUP>
bool query3_2_gen(SSBColLayout &ssbobj, KERNEL kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    // golap::PredInfo<decltype(Customer::c_nation),NationUSPred> cust_pred{customer_dev->col<1>().data()};
    golap::PredInfo<decltype(Supplier::s_nation),NationUSPred> supp_pred{supplier_dev->col<1>().data()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename GROUP>
bool query3_3_gen(SSBColLayout &ssbobj, KERNEL kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    // golap::PredInfo<decltype(Customer::c_city),CityKIPred> cust_pred{customer_dev->col<3>().data()};
    golap::PredInfo<decltype(Supplier::s_city),CityKIPred> supp_pred{supplier_dev->col<3>().data()};
    golap::PredInfo<decltype(Date::d_year),DATE_9297> date_pred{date_dev->col<1>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

template <typename KERNEL, typename GROUP>
bool query3_4_gen(SSBColLayout &ssbobj, KERNEL kernel){
    auto date_dev = prepareQ3DATE(ssbobj);
    auto supplier_dev = prepareQ3SUPP(ssbobj);

    // golap::PredInfo<decltype(Customer::c_city),CityKIPred> cust_pred{customer_dev->col<3>().data(), CityKIPred()};
    golap::PredInfo<decltype(Supplier::s_city),CityKIPred> supp_pred{supplier_dev->col<3>().data()};
    golap::PredInfo<decltype(Date::d_yearmonth),YEARMONTHDec1997> date_pred{date_dev->col<2>().data()};

    if(ssbobj.var.comp_algo == "UNCOMPRESSED"){
        return query3<KERNEL,GROUP,golap::LoadEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
    }else return query3<KERNEL,GROUP,golap::DecompressEnv>(ssbobj, kernel, *supplier_dev, *date_dev, supp_pred, date_pred);
}

bool SSBColLayout::query3_1(){
    return query3_1_gen<decltype(pipeline_q3<Q3NATION_GROUP>),Q3NATION_GROUP>(*this,pipeline_q3<Q3NATION_GROUP>);
}
bool SSBColLayout::query3_2(){
    return query3_2_gen<decltype(pipeline_q3<Q3CITY_GROUP>),Q3CITY_GROUP>(*this,pipeline_q3<Q3CITY_GROUP>);
}
bool SSBColLayout::query3_3(){
    return query3_3_gen<decltype(pipeline_q3<Q3CITY_GROUP>),Q3CITY_GROUP>(*this,pipeline_q3<Q3CITY_GROUP>);
}
bool SSBColLayout::query3_4(){
    return query3_4_gen<decltype(pipeline_q3<Q3CITY_GROUP>),Q3CITY_GROUP>(*this,pipeline_q3<Q3CITY_GROUP>);
}
