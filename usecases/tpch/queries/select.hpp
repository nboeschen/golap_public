#pragma once

#include "hl/select.cuh"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"

#include "../tpch.hpp"
#include "../data/TPCH_def.hpp"



bool TPCHColLayout::select_l_orderkey() { return golap::select(var, tables.lineitem.col<Lineitem::ORDERKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_partkey() { return golap::select(var, tables.lineitem.col<Lineitem::PARTKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_suppkey() { return golap::select(var, tables.lineitem.col<Lineitem::SUPPKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_linenumber() { return golap::select(var, tables.lineitem.col<Lineitem::LINENUMBER>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_quantity() { return golap::select(var, tables.lineitem.col<Lineitem::QUANTITY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_extendedprice() { return golap::select<decltype(Lineitem::l_extendedprice),uint32_t>(var, tables.lineitem.col<Lineitem::EXTENDEDPRICE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_discount() { return golap::select(var, tables.lineitem.col<Lineitem::DISCOUNT>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_tax() { return golap::select(var, tables.lineitem.col<Lineitem::TAX>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_returnflag() { return golap::select(var, tables.lineitem.col<Lineitem::RETURNFLAG>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_linestatus() { return golap::select(var, tables.lineitem.col<Lineitem::LINESTATUS>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_shipdate() { return golap::select(var, tables.lineitem.col<Lineitem::SHIPDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_commitdate() { return golap::select(var, tables.lineitem.col<Lineitem::COMMITDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_receiptdate() { return golap::select(var, tables.lineitem.col<Lineitem::RECEIPTDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_shipinstruct() { return golap::select(var, tables.lineitem.col<Lineitem::SHIPINSTRUCT>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::select_l_shipmode() { return golap::select(var, tables.lineitem.col<Lineitem::SHIPMODE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };


bool TPCHColLayout::filter_l_orderkey() { return golap::filter(var, tables.lineitem.col<Lineitem::ORDERKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_partkey() { return golap::filter(var, tables.lineitem.col<Lineitem::PARTKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_suppkey() { return golap::filter(var, tables.lineitem.col<Lineitem::SUPPKEY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_linenumber() { return golap::filter(var, tables.lineitem.col<Lineitem::LINENUMBER>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_quantity() { return golap::filter(var, tables.lineitem.col<Lineitem::QUANTITY>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_extendedprice() { return golap::filter(var, tables.lineitem.col<Lineitem::EXTENDEDPRICE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_discount() { return golap::filter(var, tables.lineitem.col<Lineitem::DISCOUNT>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_tax() { return golap::filter(var, tables.lineitem.col<Lineitem::TAX>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_returnflag() { return golap::filter(var, tables.lineitem.col<Lineitem::RETURNFLAG>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_linestatus() { return golap::filter(var, tables.lineitem.col<Lineitem::LINESTATUS>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_shipdate() { return golap::filter(var, tables.lineitem.col<Lineitem::SHIPDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_commitdate() { return golap::filter(var, tables.lineitem.col<Lineitem::COMMITDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_receiptdate() { return golap::filter(var, tables.lineitem.col<Lineitem::RECEIPTDATE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_shipinstruct() { return golap::filter(var, tables.lineitem.col<Lineitem::SHIPINSTRUCT>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };
bool TPCHColLayout::filter_l_shipmode() { return golap::filter(var, tables.lineitem.col<Lineitem::SHIPMODE>(), tables.lineitem.num_tuples, &BEST_BW_COMP); };