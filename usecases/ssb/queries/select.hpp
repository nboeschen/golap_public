#pragma once
#include "../ssb.hpp"

#include "hl/select.cuh"
#include "util.hpp"
#include "comp.cuh"
#include "comp_cpu.hpp"
#include "../data/SSB_def.hpp"


bool SSBColLayout::select_key() { return golap::select(var,tables.lineorder.col<Lineorder::KEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_linenum() { return golap::select(var,tables.lineorder.col<Lineorder::LINENUM>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_custkey() { return golap::select(var,tables.lineorder.col<Lineorder::CUSTKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_partkey() { return golap::select(var,tables.lineorder.col<Lineorder::PARTKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_suppkey() { return golap::select(var,tables.lineorder.col<Lineorder::SUPPKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_orderdate() { return golap::select(var,tables.lineorder.col<Lineorder::ORDERDATE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_linenumber() { return golap::select(var,tables.lineorder.col<Lineorder::LINENUMBER>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_orderpriority() { return golap::select<decltype(Lineorder::lo_orderpriority),uint8_t>(var,tables.lineorder.col<Lineorder::ORDERPRIORITY>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_shippriority() { return golap::select<decltype(Lineorder::lo_shippriority),uint8_t>(var,tables.lineorder.col<Lineorder::SHIPPRIORITY>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_quantity() { return golap::select(var,tables.lineorder.col<Lineorder::QUANTITY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_extendedprice() { return golap::select(var,tables.lineorder.col<Lineorder::EXTENDEDPRICE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_ordtotalprice() { return golap::select(var,tables.lineorder.col<Lineorder::ORDTOTALPRICE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_discount() { return golap::select(var,tables.lineorder.col<Lineorder::DISCOUNT>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_revenue() { return golap::select(var,tables.lineorder.col<Lineorder::REVENUE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_supplycost() { return golap::select(var,tables.lineorder.col<Lineorder::SUPPLYCOST>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_tax() { return golap::select(var,tables.lineorder.col<Lineorder::TAX>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_commitdate() { return golap::select(var,tables.lineorder.col<Lineorder::COMMITDATE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::select_shipmode() { return golap::select<decltype(Lineorder::lo_shipmode),uint8_t>(var,tables.lineorder.col<Lineorder::SHIPMODE>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }


bool SSBColLayout::select_c_key() {return golap::select(var,tables.customer.col<Customer::KEY>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_name() {return golap::select<decltype(Customer::c_name),uint8_t>(var,tables.customer.col<Customer::NAME>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_address() {return golap::select<decltype(Customer::c_address),uint8_t>(var,tables.customer.col<Customer::ADDRESS>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_city() {return golap::select<decltype(Customer::c_city),uint8_t>(var,tables.customer.col<Customer::CITY>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_nation() {return golap::select<decltype(Customer::c_nation),uint8_t>(var,tables.customer.col<Customer::NATION>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_region() {return golap::select<decltype(Customer::c_region),uint8_t>(var,tables.customer.col<Customer::REGION>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_phone() {return golap::select<decltype(Customer::c_phone),uint8_t>(var,tables.customer.col<Customer::PHONE>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::select_c_mktsegment() {return golap::select<decltype(Customer::c_mktsegment),uint8_t>(var,tables.customer.col<Customer::MKTSEGMENT>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}


bool SSBColLayout::filter_key() { return golap::filter(var,tables.lineorder.col<Lineorder::KEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_linenum() { return golap::filter(var,tables.lineorder.col<Lineorder::LINENUM>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_custkey() { return golap::filter(var,tables.lineorder.col<Lineorder::CUSTKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_partkey() { return golap::filter(var,tables.lineorder.col<Lineorder::PARTKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_suppkey() { return golap::filter(var,tables.lineorder.col<Lineorder::SUPPKEY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_orderdate() { return golap::filter(var,tables.lineorder.col<Lineorder::ORDERDATE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_linenumber() { return golap::filter(var,tables.lineorder.col<Lineorder::LINENUMBER>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_orderpriority() { return golap::filter<decltype(Lineorder::lo_orderpriority),uint8_t>(var,tables.lineorder.col<Lineorder::ORDERPRIORITY>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_shippriority() { return golap::filter<decltype(Lineorder::lo_shippriority),uint8_t>(var,tables.lineorder.col<Lineorder::SHIPPRIORITY>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_quantity() { return golap::filter(var,tables.lineorder.col<Lineorder::QUANTITY>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_extendedprice() { return golap::filter(var,tables.lineorder.col<Lineorder::EXTENDEDPRICE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_ordtotalprice() { return golap::filter(var,tables.lineorder.col<Lineorder::ORDTOTALPRICE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_discount() { return golap::filter(var,tables.lineorder.col<Lineorder::DISCOUNT>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_revenue() { return golap::filter(var,tables.lineorder.col<Lineorder::REVENUE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_supplycost() { return golap::filter(var,tables.lineorder.col<Lineorder::SUPPLYCOST>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_tax() { return golap::filter(var,tables.lineorder.col<Lineorder::TAX>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_commitdate() { return golap::filter(var,tables.lineorder.col<Lineorder::COMMITDATE>(),
    tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }
bool SSBColLayout::filter_shipmode() { return golap::filter<decltype(Lineorder::lo_shipmode),uint8_t>(var,tables.lineorder.col<Lineorder::SHIPMODE>(),tables.lineorder.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP); }


bool SSBColLayout::filter_c_key() {return golap::filter(var,tables.customer.col<Customer::KEY>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_name() {return golap::filter<decltype(Customer::c_name),uint8_t>(var,tables.customer.col<Customer::NAME>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_address() {return golap::filter<decltype(Customer::c_address),uint8_t>(var,tables.customer.col<Customer::ADDRESS>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_city() {return golap::filter<decltype(Customer::c_city),uint8_t>(var,tables.customer.col<Customer::CITY>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_nation() {return golap::filter<decltype(Customer::c_nation),uint8_t>(var,tables.customer.col<Customer::NATION>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_region() {return golap::filter<decltype(Customer::c_region),uint8_t>(var,tables.customer.col<Customer::REGION>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_phone() {return golap::filter<decltype(Customer::c_phone),uint8_t>(var,tables.customer.col<Customer::PHONE>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
bool SSBColLayout::filter_c_mktsegment() {return golap::filter<decltype(Customer::c_mktsegment),uint8_t>(var,tables.customer.col<Customer::MKTSEGMENT>(),
                                                                                    tables.customer.num_tuples, &BEST_BW_COMP, &BEST_RATIO_COMP);}
