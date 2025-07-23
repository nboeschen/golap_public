#pragma once
#include <any>

#include "common.cuh"


using ORDERKEY_TYPE = decltype(Lineitem::l_orderkey);
using EXTENDEDPRICE_TYPE = decltype(Lineitem::l_extendedprice);
using SHIPDATE_TYPE = decltype(Lineitem::l_shipdate);
using DISCOUNT_TYPE = decltype(Lineitem::l_discount);
using ORDERDATE_TYPE = decltype(Order::o_orderdate);
using SHIPPRIORITY_TYPE = decltype(Order::o_shippriority);
using CUSTKEY_TYPE = decltype(Order::o_custkey);
using MKTSEGMENT_TYPE = decltype(Customer::c_mktsegment);

using Q3CUSTOMER_COLUMNS = golap::ColumnTable<golap::DeviceMem,CUSTKEY_TYPE,MKTSEGMENT_TYPE>;
using Q3ORDER_COLUMNS = golap::ColumnTable<golap::DeviceMem,ORDERKEY_TYPE,ORDERDATE_TYPE,SHIPPRIORITY_TYPE,CUSTKEY_TYPE>;

#include "03.cuh"

template <typename BLOCK_ENV>
bool query3gen(TPCHColLayout &obj){
    cudaSetDevice(obj.var.cuda_device);
    uint64_t l_tuples_per_chunk = obj.var.chunk_bytes / 8;
    obj.var.comp_bytes = 0;
    obj.var.uncomp_bytes = 0;
    /**
     * - Write the chunk-compressed table to disk.
     */
    obj.var.comp_ms = 0.0;
    const std::unordered_map<std::string,std::tuple<uint64_t,golap::MetaFlags>> columns{
        {"l_orderkey",{l_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"l_extendedprice",{l_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"l_shipdate",{l_tuples_per_chunk,golap::MetaFlags::DATA|golap::MetaFlags::META}},
        {"l_discount",{l_tuples_per_chunk,golap::MetaFlags::DATA}},
        {"meta_o_orderdate",{l_tuples_per_chunk,golap::MetaFlags::META}},
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

    obj.tables.lineitem.apply(compress_columns_fun);
    SortHelper::get().prejoined->apply(compress_columns_fun);

    uint64_t chunk_num = compinfos["l_orderkey"].blocks.size();
    uint64_t num_groups = 15000*obj.var.scale_factor;
    golap::MirrorMem groups(golap::Tag<KEY_DATE_PRIO>{}, num_groups);
    golap::MirrorMem aggs(golap::Tag<uint64_t>{}, num_groups);
    aggs.dev.set(0);
    util::Date odate_hi;
    util::Date ldate_lo;

    std::stringstream{"1995-03-15"} >> odate_hi;
    std::stringstream{"1995-03-16"} >> ldate_lo;

    Q3CUSTOMER_COLUMNS customer_dev("c_custkey,c_mktsegment",obj.tables.customer.num_tuples);
    golap::PredInfo<decltype(Customer::c_mktsegment),MktSegmentPred> customer_pred{customer_dev.col<1>().data(),MktSegmentPred()};
    customer_dev.num_tuples = obj.tables.customer.num_tuples;

    Q3ORDER_COLUMNS order_dev("o_orderkey,o_orderdate,o_shippriority,o_custkey",obj.tables.orders.num_tuples);
    order_dev.num_tuples = obj.tables.orders.num_tuples;

    golap::HashMap custorder_map(customer_dev.col<0>().size(),customer_dev.col<0>().data());
    golap::HashMap hashmap(order_dev.num_tuples,order_dev.col<0>().data());

    golap::HashAggregate hash_agg(num_groups, groups.dev.ptr<KEY_DATE_PRIO>(), aggs.dev.ptr<uint64_t>());
    golap::MirrorMem combined_check{golap::Tag<uint16_t>{}, chunk_num};
    golap::MirrorMem tmp_check{golap::Tag<uint16_t>{}, chunk_num};
    combined_check.dev.set(0);
    tmp_check.dev.set(0);
    std::atomic<uint64_t> pruned_bytes{0};
    std::atomic<uint64_t> pruned_chunks{0};
    // std::any_cast<golap::MinMaxMeta<SHIPDATE_TYPE>&>(minmaxmeta["l_shipdate"]).print_debug();
    // std::any_cast<golap::MinMaxMeta<SHIPDATE_TYPE>&>(minmaxmeta["l_shipdate"]).to_host();;
    double l_time,o_time;

    std::vector<golap::TableLoader<BLOCK_ENV>> envs;
    envs.reserve(obj.var.workers);

    uint64_t startblock,endblock;
    std::vector<uint64_t> all_blocks_idxs(compinfos["l_orderkey"].blocks.size());
    std::iota(all_blocks_idxs.begin(), all_blocks_idxs.end(), 0);

    util::SliceSeq workslice(compinfos["l_orderkey"].blocks.size(), obj.var.workers);
    for (uint32_t pipeline_idx=0; pipeline_idx<obj.var.workers; ++pipeline_idx){
        // prepare environment for each thread
        workslice.get(startblock,endblock);

        envs.emplace_back(4);
        envs[pipeline_idx].add("l_orderkey", all_blocks_idxs, startblock, endblock, compinfos["l_orderkey"], nvcomp::TypeOf<ORDERKEY_TYPE>());
        envs[pipeline_idx].add("l_extendedprice", all_blocks_idxs, startblock, endblock, compinfos["l_extendedprice"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("l_shipdate", all_blocks_idxs, startblock, endblock, compinfos["l_shipdate"], nvcomp::TypeOf<uint8_t>());
        envs[pipeline_idx].add("l_discount", all_blocks_idxs, startblock, endblock, compinfos["l_discount"], nvcomp::TypeOf<uint8_t>());
    }


    std::vector<std::thread> threads;
    threads.reserve(obj.var.workers);
    auto& meta_stream = envs[0%obj.var.workers].rootstream.stream;
    auto& customer_build_stream = envs[1%obj.var.workers].rootstream.stream;;
    auto& orders_build_stream = envs[2%obj.var.workers].rootstream.stream;;

    /**
     * Start timer
     */
    uint64_t block_num = std::min((long long)obj.var.block_limit, util::div_ceil(chunk_num,512));
    golap::MetaChecker checker{block_num, chunk_num, combined_check.dev.ptr<uint16_t>(), tmp_check.dev.ptr<uint16_t>(), meta_stream};

    util::Timer timer;

    // c_custkey,c_mktsegment
    obj.tables.customer.col<Customer::CUSTKEY>().transfer(customer_dev.col<0>().data(),obj.tables.customer.num_tuples, customer_build_stream);
    obj.tables.customer.col<Customer::MKTSEGMENT>().transfer(customer_dev.col<1>().data(),obj.tables.customer.num_tuples, customer_build_stream);

    golap::hash_map_build_pred_other<<<util::div_ceil(customer_dev.col<0>().size(),512),512,0,customer_build_stream>>>(
                                            custorder_map,customer_dev.num_tuples,golap::DirectKey<decltype(Customer::c_custkey)>(),
                                            customer_pred);

    // "o_orderkey,o_orderdate,o_shippriority,o_custkey"
    obj.tables.orders.col<Order::ORDERKEY>().transfer(order_dev.col<0>().data(),obj.tables.orders.num_tuples, orders_build_stream);
    obj.tables.orders.col<Order::ORDERDATE>().transfer(order_dev.col<1>().data(),obj.tables.orders.num_tuples, orders_build_stream);
    obj.tables.orders.col<Order::SHIPPRIORITY>().transfer(order_dev.col<2>().data(),obj.tables.orders.num_tuples, orders_build_stream);
    obj.tables.orders.col<Order::CUSTKEY>().transfer(order_dev.col<3>().data(),obj.tables.orders.num_tuples, orders_build_stream);

    checkCudaErrors(cudaStreamSynchronize(customer_build_stream));

    // filter orders, join with custorder_map
    pipeline_q3_orders<<<util::div_ceil(order_dev.col<0>().size(),512),512,0,orders_build_stream>>>(hashmap,
                                                                                                      custorder_map,
                                                                                                      order_dev.col<0>().data(),
                                                                                                      order_dev.col<1>().data(),
                                                                                                      order_dev.col<3>().data(),
                                                                                                      odate_hi,
                                                                                                      order_dev.num_tuples);

    checkCudaErrors(cudaStreamSynchronize(orders_build_stream));
    o_time = timer.elapsed();
    timer.reset();

    if (obj.var.pruning == "MINMAX"){
        checker.add_minmax<SHIPDATE_TYPE>(minmaxmeta["l_shipdate"], ldate_lo, util::Date{3123123123l});
        checker.add_minmax<ORDERDATE_TYPE>(minmaxmeta["meta_o_orderdate"], util::Date{0}, odate_hi);
    }else if (obj.var.pruning == "HIST"){
        checker.add_hist<SHIPDATE_TYPE>(histmeta["l_shipdate"], ldate_lo, util::Date{3123123123l});
        checker.add_hist<ORDERDATE_TYPE>(histmeta["meta_o_orderdate"], util::Date{0}, odate_hi);
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
            while(round< env.blockenvs.at("l_orderkey").myblocks.size()){

                global_block_idx = env.blockenvs.at("l_orderkey").myblock_idxs[round];

                // check predicate:
                if (obj.var.pruning != "DONTPRUNE" && combined_check.hst.ptr<uint16_t>()[global_block_idx] == (uint16_t) 0){
                    util::Log::get().debug_fmt("Thread[%lu, round%lu] would skip the next chunk idx %lu...", pipeline_idx, round, global_block_idx);
                    env.rootop.skip_step(env.rootstream.stream,env.rootevent.event);

                    pruned_bytes.fetch_add(env.blockenvs.at("l_orderkey").myblocks[round].size
                                          +env.blockenvs.at("l_extendedprice").myblocks[round].size
                                          +env.blockenvs.at("l_shipdate").myblocks[round].size
                                          +env.blockenvs.at("l_discount").myblocks[round].size, std::memory_order_relaxed);
                    pruned_chunks.fetch_add(1, std::memory_order_relaxed);
                    round += 1;
                    continue;
                }

                if (!env.rootop.step(env.rootstream.stream,env.rootevent.event)){
                    util::Log::get().error_fmt("Shouldnt happen!");
                }
                checkCudaErrors(cudaStreamSynchronize(env.rootstream.stream));


                tuples_this_round = env.blockenvs.at("l_orderkey").myblocks[round].tuples;
                util::Log::get().debug_fmt("Thread[%lu, round%lu] Block %lu, %lu tuples", pipeline_idx, round, global_block_idx, tuples_this_round);

                num_blocks = std::min((long long)obj.var.block_limit, util::div_ceil(tuples_this_round,512));

                pipeline_q3_lineitem<<<num_blocks,512,0,env.rootstream.stream>>>(hash_agg,
                                                                        hashmap,
                                env.blockenvs.at("l_orderkey").decomp_buf.template ptr<ORDERKEY_TYPE>(),
                                env.blockenvs.at("l_extendedprice").decomp_buf.template ptr<EXTENDEDPRICE_TYPE>(),
                                env.blockenvs.at("l_shipdate").decomp_buf.template ptr<SHIPDATE_TYPE>(),
                                env.blockenvs.at("l_discount").decomp_buf.template ptr<DISCOUNT_TYPE>(),
                                order_dev.col<1>().data(),
                                order_dev.col<2>().data(),
                                ldate_lo,
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

    l_time = timer.elapsed();
    obj.var.time_ms = timer.elapsed();
    /**
     * Stopped timer
     */

    golap::HostMem pop_group_slot{golap::Tag<uint32_t>{}, num_groups};
    checkCudaErrors(cudaMemcpy(pop_group_slot.ptr<uint32_t>(), hash_agg.wrote_group, num_groups*sizeof(uint32_t), cudaMemcpyDefault));

    obj.var.device_mem_used = golap::DEVICE_ALLOCATED.load();
    obj.var.host_mem_used = golap::HOST_ALLOCATED.load();
    obj.var.pruned_bytes = pruned_bytes.load();

    uint64_t results_total = 0;
    for(uint32_t i = 0; i<num_groups; ++i){
        if (pop_group_slot.ptr<uint32_t>()[i] == 0) continue;

        results_total += 1;
        if (i < 25){
            auto& group = groups.hst.ptr<KEY_DATE_PRIO>()[i];
            util::Log::get().info_fmt("l_orderkey: %llu, o_orderdate: %llu, o_shippriority: %lld, revenue: %llu",group.l_orderkey,group.o_orderdate.t,
                                      group.o_shippriority, aggs.hst.ptr<uint64_t>()[i]);
        }
    }

    util::Log::get().info_fmt("%llu total results, Order timer: %.2f, Lineitem timer: %.2f,",results_total,o_time,l_time);
    util::Log::get().info_fmt("Pruned: %lu of %lu chunks (%.2f), %lu of %lu bytes (%.2f)",
                                pruned_chunks.load(),chunk_num,(double)pruned_chunks.load()/chunk_num,
                                pruned_bytes.load(),obj.var.comp_bytes,(double)pruned_bytes.load()/obj.var.comp_bytes);

    return true;

}

bool TPCHColLayout::query3(){


    if(var.comp_algo == "UNCOMPRESSED") return query3gen<golap::LoadEnv>(*this);
    else return query3gen<golap::DecompressEnv>(*this);
}