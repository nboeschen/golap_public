#pragma once

#include <cstdint>
#include <memory>
#include <cstdlib>

#include "TPCH_helper.hpp"
#include "mem.hpp"
#include "util.hpp"
#include "types.hpp"
#include "table.hpp"

struct alignas(8) Order{
    static constexpr char ATTR[] = "o_orderkey,o_custkey,o_orderstatus,o_totalprice,o_orderdate,o_orderpriority,o_clerk,o_shippriority";
    enum : uint64_t {ORDERKEY=0,CUSTKEY=1,ORDERSTATUS=2,TOTALPRICE=3,ORDERDATE=4,ORDERPRIORITY=5,CLERK=6,SHIPPRIORITY=7};

    uint64_t o_orderkey;
    uint64_t o_custkey;
    char o_orderstatus;
    util::Decimal64 o_totalprice;
    util::Date o_orderdate;
    util::Padded<char[21]> o_orderpriority;
    util::Padded<char[16]> o_clerk;
    int64_t o_shippriority;
};

struct alignas(8) Lineitem{
    static constexpr char ATTR[] = "l_orderkey,l_partkey,l_suppkey,l_linenumber,l_quantity,l_extendedprice,l_discount,l_tax,l_returnflag,l_linestatus,l_shipdate,l_commitdate,l_receiptdate,l_shipinstruct,l_shipmode";
    enum : uint64_t {ORDERKEY=0,PARTKEY=1,SUPPKEY=2,LINENUMBER=3,QUANTITY=4,EXTENDEDPRICE=5,DISCOUNT=6,TAX=7,
                    RETURNFLAG=8,LINESTATUS=9,SHIPDATE=10,COMMITDATE=11,RECEIPTDATE=12,SHIPINSTRUCT=13,SHIPMODE=14};

    uint64_t l_orderkey;
    uint64_t l_partkey;
    uint64_t l_suppkey;

    uint8_t l_linenumber;
    uint8_t l_quantity;
    util::Decimal64 l_extendedprice;
    util::Decimal64 l_discount;
    util::Decimal64 l_tax;

    char l_returnflag;
    char l_linestatus;
    util::Date l_shipdate;
    util::Date l_commitdate;
    util::Date l_receiptdate;
    util::Padded<char[21]> l_shipinstruct;
    util::Padded<char[21]> l_shipmode;
};

struct alignas(8) Part{
    static constexpr char ATTR[] = "p_partkey,p_name,p_mfgr,p_brand,p_type,p_size,p_container,p_retailprice";
    enum : uint64_t {PARTKEY=0,NAME=1,MFGR=2,BRAND=3,TYPE=4,SIZE=5,CONTAINER=6,RETAILPRICE=7};

    uint64_t p_partkey;
    util::Padded<char[56]> p_name;
    util::Padded<char[26]> p_mfgr;
    util::Padded<char[11]> p_brand;
    util::Padded<char[26]> p_type;
    int64_t p_size;
    util::Padded<char[11]> p_container;
    util::Decimal64 p_retailprice;
};

struct alignas(8) Supplier{
    static constexpr char ATTR[] = "s_suppkey,s_name,s_address,s_nationkey,s_phone,s_acctbal";
    enum : uint64_t {SUPPKEY=0,NAME=1,ADDRESS=2,NATIONKEY=3,PHONE=4,ACCTBAL=5};

    uint64_t s_suppkey;
    util::Padded<char[26]> s_name;
    util::Padded<char[41]> s_address;
    uint64_t s_nationkey;
    util::Padded<char[16]> s_phone;
    int64_t s_acctbal;
};


struct alignas(8) PartSupp{
    static constexpr char ATTR[] = "ps_partkey,ps_suppkey,ps_availqty,ps_supplycost";
    enum : uint64_t {PARTKEY=0,SUPPKEY=1,AVAILQTY=2,SUPPLYCOST=3};

    uint64_t ps_partkey;
    uint64_t ps_suppkey;
    int64_t ps_availqty;
    util::Decimal64 ps_supplycost;
};

struct alignas(8) Customer{
    static constexpr char ATTR[] = "c_custkey,c_name,c_address,c_nationkey,c_phone,c_acctbal,c_mktsegment";
    enum : uint64_t {CUSTKEY=0,NAME=1,ADDRESS=2,NATIONKEY=3,PHONE=4,ACCTBAL=5,MKTSEGMENT=6};

    uint64_t c_custkey;
    util::Padded<char[26]> c_name;
    util::Padded<char[41]> c_address;
    uint64_t c_nationkey;
    util::Padded<char[16]> c_phone;
    util::Decimal64 c_acctbal;
    util::Padded<char[11]> c_mktsegment;
};

struct alignas(8) Nation{
    static constexpr char ATTR[] = "n_nationkey,n_name,n_regionkey";
    enum : uint64_t {NATIONKEY=0,NAME=1,REGIONKEY=1};

    uint64_t n_nationkey;
    util::Padded<char[16]> n_name;
    uint64_t n_regionkey;
};

struct alignas(8) Region{
    static constexpr char ATTR[] = "r_regionkey,r_name";
    enum : uint64_t {REGIONKEY=0,NAME=1};

    uint64_t r_regionkey;
    util::Padded<char[13]> r_name;
};

template <typename MEM_TYPE>
class TPCH_Tables_col{
public:
    TPCH_Tables_col(uint64_t scale_factor):scale_factor(scale_factor),
            orders("orders", Order::ATTR, 0),
            lineitem("lineitem", Lineitem::ATTR, 0),
            part("part", Part::ATTR, 0),
            supplier("supplier", Supplier::ATTR, 0),
            partsupp("partsupp", PartSupp::ATTR, 0),
            customer("customer", Customer::ATTR, 0),
            nation("nation", Nation::ATTR, 0),
            region("region", Region::ATTR, 0)
        {

        region_num = 5;
        nation_num = 25;
        supplier_num = scale_factor * 10000;
        part_num = scale_factor * 200000;
        partsupp_num = scale_factor * 800000;
        customer_num = scale_factor * 150000;
        order_num = scale_factor * 1500000;
        lineitem_num = order_num * 5;

    }
    void init(){
        init_region();
        init_nation();
        init_customer();
        init_supplier();
        init_part();
        init_partsupp();
        init_order();
        init_lineitem();
    }

    void populate(){
        util::Log::get().warn_fmt("Populate not implemented for TPCH data!");
    }

    void init_region(){ region.resize(region_num); }
    void init_nation(){ nation.resize(nation_num); }
    void init_customer(){ customer.resize(customer_num); }
    void init_supplier(){ supplier.resize(supplier_num); }
    void init_part(){ part.resize(part_num); }
    void init_partsupp(){ partsupp.resize(partsupp_num); }
    void init_order(){ orders.resize(order_num); }
    void init_lineitem(){ lineitem.resize(lineitem_num); }

    void populate_region(){
    }

    void populate_nation(){
    }

    void populate_customer(){
    }

    void populate_supplier(){
    }

    void populate_part(){
    }

    void populate_partsupp(){
    }

    void populate_order_lineitem(){
    }

    uint64_t size_bytes(){
        return orders.size_bytes()+lineitem.size_bytes()+part.size_bytes()
                +supplier.size_bytes()+partsupp.size_bytes()
                +customer.size_bytes()+nation.size_bytes()+region.size_bytes();
    }

    template<typename Fn>
    void apply(Fn&& f){
        f(region);
        f(nation);
        f(customer);
        f(supplier);
        f(part);
        f(partsupp);
        f(orders);
        f(lineitem);
    }

    uint32_t scale_factor;
    uint64_t rand_seed = 0xBEE5BEE5;

    uint64_t order_num;
    uint64_t lineitem_num;
    uint64_t part_num;
    uint64_t supplier_num;
    uint64_t partsupp_num;
    uint64_t customer_num;
    uint64_t nation_num;
    uint64_t region_num;

    golap::ColumnTable<MEM_TYPE,decltype(Order::o_orderkey),decltype(Order::o_custkey),decltype(Order::o_orderstatus),decltype(Order::o_totalprice),decltype(Order::o_orderdate),decltype(Order::o_orderpriority),decltype(Order::o_clerk),decltype(Order::o_shippriority)> orders;
    golap::ColumnTable<MEM_TYPE,decltype(Lineitem::l_orderkey),decltype(Lineitem::l_partkey),decltype(Lineitem::l_suppkey),decltype(Lineitem::l_linenumber),decltype(Lineitem::l_quantity),decltype(Lineitem::l_extendedprice),decltype(Lineitem::l_discount),decltype(Lineitem::l_tax),decltype(Lineitem::l_returnflag),decltype(Lineitem::l_linestatus),decltype(Lineitem::l_shipdate),decltype(Lineitem::l_commitdate),decltype(Lineitem::l_receiptdate),decltype(Lineitem::l_shipinstruct),decltype(Lineitem::l_shipmode)> lineitem;
    golap::ColumnTable<MEM_TYPE,decltype(Part::p_partkey),decltype(Part::p_name),decltype(Part::p_mfgr),decltype(Part::p_brand),decltype(Part::p_type),decltype(Part::p_size),decltype(Part::p_container),decltype(Part::p_retailprice)> part;
    golap::ColumnTable<MEM_TYPE,decltype(PartSupp::ps_partkey),decltype(PartSupp::ps_suppkey),decltype(PartSupp::ps_availqty),decltype(PartSupp::ps_supplycost)> partsupp;
    golap::ColumnTable<MEM_TYPE,decltype(Supplier::s_suppkey),decltype(Supplier::s_name),decltype(Supplier::s_address),decltype(Supplier::s_nationkey),decltype(Supplier::s_phone),decltype(Supplier::s_acctbal)> supplier;
    golap::ColumnTable<MEM_TYPE,decltype(Customer::c_custkey),decltype(Customer::c_name),decltype(Customer::c_address),decltype(Customer::c_nationkey),decltype(Customer::c_phone),decltype(Customer::c_acctbal),decltype(Customer::c_mktsegment)> customer;
    golap::ColumnTable<MEM_TYPE,decltype(Nation::n_nationkey),decltype(Nation::n_name),decltype(Nation::n_regionkey)> nation;
    golap::ColumnTable<MEM_TYPE,decltype(Region::r_regionkey),decltype(Region::r_name)> region;

};


