#pragma once

#include <cstdint>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include "mem.hpp"
#include "util.hpp"
#include "types.hpp"
#include "table.hpp"
#include "SSB_helper.hpp"


struct alignas(8) Lineorder{
    static constexpr char LINEORDER_ATTR[] = "lo_key,lo_linenum,lo_custkey,lo_partkey,lo_suppkey,lo_orderdate,lo_linenumber,lo_orderpriority,lo_shippriority,lo_quantity,lo_extendedprice,lo_ordtotalprice,lo_discount,lo_revenue,lo_supplycost,lo_tax,lo_commitdate,lo_shipmode";
    enum : uint64_t {KEY = 0,LINENUM=1,CUSTKEY=2,PARTKEY=3,SUPPKEY=4,ORDERDATE=5,LINENUMBER=6,ORDERPRIORITY=7,
                SHIPPRIORITY=8,QUANTITY=9,EXTENDEDPRICE=10,ORDTOTALPRICE=11,DISCOUNT=12,REVENUE=13,
                SUPPLYCOST=14,TAX=15,COMMITDATE=16,SHIPMODE=17};

    // from order
    uint64_t lo_key;
    // uint64_t lo_custkey;
    // uint64_t lo_orderdate;
    // uint32_t lo_ordtotalprice;
    // char lo_orderpriority[16];
    // char lo_shippriority;
    uint8_t lo_linenum;

    // from line
    uint64_t lo_custkey;
    uint64_t lo_partkey;
    uint64_t lo_suppkey;
    uint64_t lo_orderdate;
    uint8_t lo_linenumber;
    util::Padded<char[16]> lo_orderpriority;
    char lo_shippriority;
    uint8_t lo_quantity;
    uint32_t lo_extendedprice;
    uint32_t lo_ordtotalprice;
    uint8_t lo_discount;
    uint32_t lo_revenue;
    uint32_t lo_supplycost;
    uint8_t lo_tax;
    uint64_t lo_commitdate;
    // char lo_shipmode[11];
    util::Padded<char[11]> lo_shipmode;
};

struct alignas(8) Part{
    static constexpr char PART_ATTR[] = "p_key,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container";
    enum : uint64_t {KEY=0,NAME=1,MFGR=2,CATEGORY=3,BRAND1=4,COLOR=5,TYPE=6,SIZE=7,CONTAINER=8};
    uint64_t p_key;
    // char p_name[23];
    util::Padded<char[23]> p_name;
    // char p_mfgr[7];
    util::Padded<char[7]> p_mfgr;
    // char p_category[8];
    util::Padded<char[8]> p_category;
    // char p_brand1[10];
    util::Padded<char[10]> p_brand1;
    // char p_color[12];
    util::Padded<char[12]> p_color;
    // char p_type[26];
    util::Padded<char[26]> p_type;
    uint8_t p_size;
    // char p_container[11];
    util::Padded<char[11]> p_container;
};


struct alignas(8) Supplier{
    static constexpr char SUPPLIER_ATTR[] = "s_key,s_name,s_address,s_city,s_nation,s_region,s_phone";
    enum : uint64_t {KEY=0,NAME=1,ADDRESS=2,CITY=3,NATION=4,REGION=5,PHONE=6};
    uint64_t s_key;
    // char s_name[26];
    util::Padded<char[26]> s_name;
    // char s_address[26];
    util::Padded<char[26]> s_address;
    // char s_city[11];
    util::Padded<char[11]> s_city;
    // char s_nation[16];
    util::Padded<char[16]> s_nation;
    // char s_region[13];
    util::Padded<char[13]> s_region;
    // char s_phone[16];
    util::Padded<char[16]> s_phone;
};

struct alignas(8) Customer{
    static constexpr char CUSTOMER_ATTR[] = "c_key,c_name,c_address,c_city,c_nation,c_region,c_phone,c_mktsegment";
    enum : uint64_t {KEY=0,NAME=1,ADDRESS=2,CITY=3,NATION=4,REGION=5,PHONE=6,MKTSEGMENT=7};
    uint64_t c_key;
    // char c_name[26];
    util::Padded<char[26]> c_name;
    // char c_address[26];
    util::Padded<char[26]> c_address;
    // char c_city[11];
    util::Padded<char[11]> c_city;
    // char c_nation[16];
    util::Padded<char[16]> c_nation;
    // char c_region[13];
    util::Padded<char[13]> c_region;
    // char c_phone[16];
    util::Padded<char[16]> c_phone;
    // char c_mktsegment[11];
    util::Padded<char[11]> c_mktsegment;
};

struct alignas(8) Date{
    static constexpr char DATE_ATTR[] = "d_key,d_date,d_dayofweek,d_month,d_year,d_yearmonthnum,d_yearmonth,d_daynuminweek,d_daynuminmonth,d_daynuminyear,d_monthnuminyear,d_weeknuminyear,d_sellingseason,d_lastdayinweekfl,d_lastdayinmonthfl,d_holidayfl,d_weekdayfl";
    enum : uint64_t {KEY=0,DATE=1,DAYOFWEEK=2,MONTH=3,YEAR=4,YEARMONTHNUM=5,YEARMONTH=6,DAYNUMINWEEK=7,
        DAYNUMINMONTH=8,DAYNUMINYEAR=9,MONTHNUMINYEAR=10,WEEKNUMINYEAR=11,SELLINGSEASON=12,
        LASTDAYINWEEKFL=13,LASTDAYINMONTHFL=14,HOLIDAYFL=15,WEEKDAYFL=16};
    uint64_t d_key;
    // char d_date[19];
    util::Padded<char[19]> d_date;
    // char d_dayofweek[10];
    util::Padded<char[10]> d_dayofweek;
    // char d_month[10];
    util::Padded<char[10]> d_month;
    uint16_t d_year;
    uint32_t d_yearmonthnum;
    // char d_yearmonth[8];
    util::Padded<char[8]> d_yearmonth;
    uint16_t d_daynuminweek;
    uint16_t d_daynuminmonth;
    uint16_t d_daynuminyear;
    uint16_t d_monthnuminyear;
    uint16_t d_weeknuminyear;
    // char d_sellingseason[13];
    util::Padded<char[13]> d_sellingseason;
    uint8_t d_lastdayinweekfl;
    uint8_t d_lastdayinmonthfl;
    uint8_t d_holidayfl;
    uint8_t d_weekdayfl;
};


template <typename MEM_TYPE>
class SSB_Tables_col{
public:
    SSB_Tables_col(uint64_t scale_factor, uint64_t customer_factor):scale_factor(scale_factor),
            lineorder("lineorder",Lineorder::LINEORDER_ATTR,0),part("part",Part::PART_ATTR,0),supplier("supplier",Supplier::SUPPLIER_ATTR,0),customer("customer",Customer::CUSTOMER_ATTR,0),date("date",Date::DATE_ATTR,0)
        {
        order_num = scale_factor * 1500000;
        part_num = (uint64_t)(200000*floor(1+log2(scale_factor)));
        supplier_num = scale_factor * 2000;
        customer_num = scale_factor * 30000 * customer_factor;
        date_num = 365*7;
        lineorder_num = order_num * 5;
    }
    void init(){
        init_date();
        init_customer();
        init_supplier();
        init_part();
        init_lineorder();
    }

    void init_lineorder(){
        lineorder.resize(lineorder_num);
    }
    void init_part(){
        part.resize(part_num);
    }
    void init_supplier(){
        supplier.resize(supplier_num);
    }
    void init_customer(){
        customer.resize(customer_num);
    }
    void init_date(){
        date.resize(date_num);
    }


    void populate(){
        populate_date();
        populate_customer();
        populate_supplier();
        populate_part();
        populate_lineorder();
    }

    void repopulate_custkey(){
        uint64_t custkey;
        for (uint64_t lo_idx = 0; lo_idx<lineorder.num_tuples; ++lo_idx){
            // if its the first line in that order, generate the custkey
            if(lineorder.template col<Lineorder::LINENUMBER>().data()[lo_idx] == 1){
                do custkey = util::uniform_int(rand_seed,0,customer.num_tuples);
                while (custkey % CUST_MORTALITY == 0);
            }

            lineorder.template col<Lineorder::CUSTKEY>().data()[lo_idx] = custkey;

        }
    }

    void populate_lineorder(){
        uint8_t cur_lines, cur_quantity, cur_discount, cur_tax;
        uint64_t total_lines = 0;
        uint64_t cur_o_key,low_bits,custkey=0,partkey=0,suppkey=0,datekey=0;
        uint32_t cur_totalprice,cur_extendedprice;
        uint64_t ten_percent = (uint64_t) (order_num / 10.0);

        for(uint64_t order_idx=0; order_idx < order_num; ++order_idx){
            cur_lines = (uint8_t) util::uniform_int(rand_seed,1,8);
            datekey = date.template col<Date::KEY>().data()[util::uniform_int(rand_seed,0,date.num_tuples)];
            do custkey = util::uniform_int(rand_seed,0,customer.num_tuples);
            while (custkey % CUST_MORTALITY == 0);

            cur_totalprice = 0;

            std::string cur_prio = PRIORITY[util::uniform_int(rand_seed,0,PRIORITY.size())];


            low_bits = (order_idx & ((1 << SPARSE_KEEP) - 1));
            cur_o_key = order_idx;
            cur_o_key >>= SPARSE_KEEP;
            cur_o_key <<= SPARSE_BITS;
            cur_o_key <<= SPARSE_KEEP;
            cur_o_key += low_bits;
            // order_idx: 0, lo_key: 0
            // order_idx: 1, lo_key: 1
            // ...
            // order_idx: 8, lo_key: 32
            // order_idx: 9, lo_key: 33

            for(uint8_t line_idx = 0; line_idx < cur_lines; ++line_idx){
                cur_quantity = (uint8_t) util::uniform_int(rand_seed,1,51);
                cur_discount = (uint8_t) util::uniform_int(rand_seed,0,11);
                cur_tax = (uint8_t) util::uniform_int(rand_seed,0,9);
                partkey = part.template col<Part::KEY>().data()[util::uniform_int(rand_seed,0,part.num_tuples)];
                suppkey = supplier.template col<Supplier::KEY>().data()[util::uniform_int(rand_seed,0,supplier.num_tuples)];
                cur_extendedprice = cur_quantity * partkey_to_price(partkey);

                lineorder.template col<Lineorder::KEY>().data()[total_lines+line_idx] = cur_o_key;
                lineorder.template col<Lineorder::LINENUM>().data()[total_lines+line_idx] = cur_lines;
                lineorder.template col<Lineorder::CUSTKEY>().data()[total_lines+line_idx] = custkey;
                lineorder.template col<Lineorder::PARTKEY>().data()[total_lines+line_idx] = partkey;
                lineorder.template col<Lineorder::SUPPKEY>().data()[total_lines+line_idx] = suppkey;
                lineorder.template col<Lineorder::ORDERDATE>().data()[total_lines+line_idx] = datekey;
                lineorder.template col<Lineorder::LINENUMBER>().data()[total_lines+line_idx] = (uint8_t) (line_idx+1);
                lineorder.template col<Lineorder::SHIPPRIORITY>().data()[total_lines+line_idx] = ' ';
                lineorder.template col<Lineorder::QUANTITY>().data()[total_lines+line_idx] = cur_quantity;
                lineorder.template col<Lineorder::EXTENDEDPRICE>().data()[total_lines+line_idx] = cur_extendedprice;
                // lineorder.template col<Lineorder::ORDTOTALPRICE>().data()[total_lines+line_idx] = 0;
                lineorder.template col<Lineorder::DISCOUNT>().data()[total_lines+line_idx] = cur_discount;
                lineorder.template col<Lineorder::REVENUE>().data()[total_lines+line_idx] = cur_extendedprice * (100.0-cur_discount)/100;
                lineorder.template col<Lineorder::SUPPLYCOST>().data()[total_lines+line_idx] = 6 * partkey_to_price(partkey) / 10;
                lineorder.template col<Lineorder::TAX>().data()[total_lines+line_idx] = cur_tax;
                lineorder.template col<Lineorder::COMMITDATE>().data()[total_lines+line_idx] = datekey + util::uniform_int(rand_seed,30,91);

                cur_totalprice += ((cur_extendedprice *  (100 - cur_discount))/100) * (100 + cur_tax) / 100;

                snprintf(lineorder.template col<Lineorder::ORDERPRIORITY>().data()[total_lines+line_idx].d,
                                sizeof(decltype(Lineorder::lo_orderpriority.d)), "%s", cur_prio.c_str());
                snprintf(lineorder.template col<Lineorder::SHIPMODE>().data()[total_lines+line_idx].d,
                                sizeof(decltype(Lineorder::lo_shipmode.d)), "%s", SHIPMODE[util::uniform_int(rand_seed,0,SHIPMODE.size())].c_str());


            }
            // update ordtotalprice
            for(uint8_t line_idx = 0; line_idx < cur_lines; ++line_idx){
                lineorder.template col<Lineorder::ORDTOTALPRICE>().data()[total_lines+line_idx] = cur_totalprice;
            }

            if (order_idx % ten_percent == 0){
                util::Log::get().debug_fmt("Lineorder table: %lu%% of orders populated",10*order_idx/ten_percent);
            }

            total_lines += cur_lines;

        }
        lineorder.num_tuples = total_lines;
        util::Log::get().debug_fmt("Actual total lines populated: %lu",total_lines);
    }

    void populate_part(){
        uint32_t num1,num2,num3,tsyl1,tsyl2,tsyl3,csyl1,csyl2,name1,name2,col1;
        for(uint64_t t_idx=0; t_idx < part_num; ++t_idx){

            part.template col<Part::KEY>().data()[t_idx] = t_idx;
            part.template col<Part::SIZE>().data()[t_idx] = util::uniform_int(rand_seed,1,51);

            num1 = util::uniform_int(rand_seed,1,6);
            num2 = util::uniform_int(rand_seed,1,6);
            num3 = util::uniform_int(rand_seed,1,41);
            tsyl1 = util::uniform_int(rand_seed,0,TYPE[0].size());
            tsyl2 = util::uniform_int(rand_seed,0,TYPE[1].size());
            tsyl3 = util::uniform_int(rand_seed,0,TYPE[2].size());
            csyl1 = util::uniform_int(rand_seed,0,CONT[0].size());
            csyl2 = util::uniform_int(rand_seed,0,CONT[1].size());
            name1 = util::uniform_int(rand_seed,0,COLOR.size());
            name2 = util::uniform_int(rand_seed,0,COLOR.size());
            col1 = util::uniform_int(rand_seed,0,COLOR.size());
            // name is 2 colors, color is 1 color. here theyre not unique
            snprintf(part.template col<Part::NAME>().data()[t_idx].d,
                     sizeof(Part::p_name.d), "%s %s", COLOR[name1].c_str(), COLOR[name2].c_str());
            snprintf(part.template col<Part::MFGR>().data()[t_idx].d,
                     sizeof(Part::p_mfgr.d), "MFGR#%u", num1);
            snprintf(part.template col<Part::CATEGORY>().data()[t_idx].d,
                     sizeof(Part::p_category.d), "MFGR#%u%u", num1, num2);
            snprintf(part.template col<Part::BRAND1>().data()[t_idx].d,
                     sizeof(Part::p_brand1.d), "MFGR#%u%u%02u", num1, num2, num3);
            snprintf(part.template col<Part::COLOR>().data()[t_idx].d,
                     sizeof(Part::p_color.d), "%s", COLOR[col1].c_str());
            snprintf(part.template col<Part::TYPE>().data()[t_idx].d,
                     sizeof(Part::p_type.d), "%s %s %s", TYPE[0][tsyl1].c_str(), TYPE[1][tsyl2].c_str(), TYPE[2][tsyl3].c_str());
            snprintf(part.template col<Part::CONTAINER>().data()[t_idx].d,
                     sizeof(Part::p_container.d), "%s %s", CONT[0][csyl1].c_str(), CONT[1][csyl2].c_str());
        }
        part.num_tuples = part_num;
    }

    void populate_supplier(){
        uint32_t nation_idx,city_code;
        uint32_t acode,exchg,pnumber;

        std::vector<std::string> NATION_NAMES;
        for(auto& tup: NATION) NATION_NAMES.push_back(tup.first);

        for(uint64_t t_idx=0; t_idx < supplier_num; ++t_idx){

            supplier.template col<Supplier::KEY>().data()[t_idx] = t_idx;
            nation_idx = util::uniform_int(rand_seed,0,NATION.size());
            city_code = util::uniform_int(rand_seed,0,10);
            acode = util::uniform_int(rand_seed,100,1000);
            exchg = util::uniform_int(rand_seed,100,1000);
            pnumber = util::uniform_int(rand_seed,1000,10000);
            snprintf(supplier.template col<Supplier::NAME>().data()[t_idx].d, sizeof(Supplier::s_name.d), "Supplier%lu", t_idx);
            snprintf(supplier.template col<Supplier::ADDRESS>().data()[t_idx].d, sizeof(Supplier::s_address.d), "TODO");
            snprintf(supplier.template col<Supplier::CITY>().data()[t_idx].d, sizeof(Supplier::s_city.d), "%.*s%u", 9, NATION_NAMES[nation_idx].c_str(), city_code);
            snprintf(supplier.template col<Supplier::NATION>().data()[t_idx].d, sizeof(Supplier::s_nation.d), "%s", NATION_NAMES[nation_idx].c_str());
            snprintf(supplier.template col<Supplier::REGION>().data()[t_idx].d, sizeof(Supplier::s_region.d), "%s", REGION[NATION.at(NATION_NAMES[nation_idx])].c_str());
            snprintf(supplier.template col<Supplier::PHONE>().data()[t_idx].d, sizeof(Supplier::s_phone.d), "%02d-%03d-%03d-%04d", 10 + nation_idx,acode,exchg,pnumber);

        }
        supplier.num_tuples = supplier_num;
    }

    void populate_customer(){
        uint32_t nation_idx,city_code,mktsegment;
        uint32_t acode,exchg,pnumber;

        std::vector<std::string> NATION_NAMES;
        for(auto& tup: NATION) NATION_NAMES.push_back(tup.first);

        for(uint64_t t_idx=0; t_idx < customer_num; ++t_idx){

            customer.template col<Customer::KEY>().data()[t_idx] = t_idx;
            nation_idx = util::uniform_int(rand_seed,0,NATION.size());
            city_code = util::uniform_int(rand_seed,0,10);
            acode = util::uniform_int(rand_seed,100,1000);
            exchg = util::uniform_int(rand_seed,100,1000);
            pnumber = util::uniform_int(rand_seed,1000,10000);
            mktsegment = util::uniform_int(rand_seed, 0, SEGMENT.size());
            snprintf(customer.template col<Customer::NAME>().data()[t_idx].d, sizeof(Customer::c_name.d), "Customer%lu", t_idx);
            snprintf(customer.template col<Customer::ADDRESS>().data()[t_idx].d, sizeof(Customer::c_address.d), "TODO");
            snprintf(customer.template col<Customer::CITY>().data()[t_idx].d, sizeof(Customer::c_city.d), "%.*s%u", 9, NATION_NAMES[nation_idx].c_str(), city_code);
            snprintf(customer.template col<Customer::NATION>().data()[t_idx].d, sizeof(Customer::c_nation.d), "%s", NATION_NAMES[nation_idx].c_str());
            snprintf(customer.template col<Customer::REGION>().data()[t_idx].d, sizeof(Customer::c_region.d), "%s", REGION[NATION.at(NATION_NAMES[nation_idx])].c_str());
            snprintf(customer.template col<Customer::PHONE>().data()[t_idx].d, sizeof(Customer::c_phone.d), "%02d-%03d-%03d-%04d", 10 + nation_idx,acode,exchg,pnumber);
            snprintf(customer.template col<Customer::MKTSEGMENT>().data()[t_idx].d, sizeof(Customer::c_mktsegment.d), "%s", SEGMENT[mktsegment].c_str());

        }
        customer.num_tuples = customer_num;
    }

    void populate_date(){
        std::time_t cur = 694245661; // Wed Jan  1 06:01:01 1992 GMT
        std::tm *tm;
        for(uint64_t t_idx=0; t_idx < date_num; ++t_idx){
            tm = std::gmtime(&cur);
            
            date.template col<Date::KEY>().data()[t_idx] = ((1900+tm->tm_year)*10000 + (tm->tm_mon+1)*100 + tm->tm_mday);
            std::strftime(date.template col<Date::DATE>().data()[t_idx].d, sizeof(Date::d_date.d), "%B %d, %Y", tm);
            std::strftime(date.template col<Date::DAYOFWEEK>().data()[t_idx].d, sizeof(Date::d_dayofweek.d), "%A", tm);
            std::strftime(date.template col<Date::MONTH>().data()[t_idx].d, sizeof(Date::d_month.d), "%B", tm);
            date.template col<Date::YEAR>().data()[t_idx] = (1900+tm->tm_year);
            date.template col<Date::YEARMONTHNUM>().data()[t_idx] = ((1900+tm->tm_year)*100 + tm->tm_mon+1);
            std::strftime(date.template col<Date::YEARMONTH>().data()[t_idx].d, sizeof(Date::d_yearmonth.d), "%b%Y", tm);
            date.template col<Date::DAYNUMINWEEK>().data()[t_idx] = (tm->tm_wday+1);
            date.template col<Date::DAYNUMINMONTH>().data()[t_idx] = (tm->tm_mday);
            date.template col<Date::DAYNUMINYEAR>().data()[t_idx] = (tm->tm_yday+1);
            date.template col<Date::MONTHNUMINYEAR>().data()[t_idx] = (tm->tm_mon+1);
            date.template col<Date::WEEKNUMINYEAR>().data()[t_idx] = (((tm->tm_yday + 7 - (tm->tm_wday ? (tm->tm_wday - 1) : 6)) / 7)+1);
            // 
            date.template col<Date::LASTDAYINWEEKFL>().data()[t_idx] = (tm->tm_wday == 6 ? 1:0);
            // 
            // 
            //

            cur += 60*60*24;
        }
        date.num_tuples = date_num;
    }

    template<typename Fn>
    void apply(Fn&& f){
        f(lineorder);
        f(part);
        f(supplier);
        f(customer);
        f(date);
    }

    uint64_t size_bytes(){
        return lineorder.size_bytes()+part.size_bytes()+supplier.size_bytes()
                +customer.size_bytes()+date.size_bytes();
    }

    uint32_t scale_factor;
    uint64_t rand_seed = 0xBEE5BEE5;
    uint64_t order_num,part_num,supplier_num,customer_num,date_num,lineorder_num;
    golap::ColumnTable<MEM_TYPE,decltype(Lineorder::lo_key),decltype(Lineorder::lo_linenum),decltype(Lineorder::lo_custkey),decltype(Lineorder::lo_partkey),decltype(Lineorder::lo_suppkey),decltype(Lineorder::lo_orderdate),decltype(Lineorder::lo_linenumber),decltype(Lineorder::lo_orderpriority),decltype(Lineorder::lo_shippriority),decltype(Lineorder::lo_quantity),decltype(Lineorder::lo_extendedprice),decltype(Lineorder::lo_ordtotalprice),decltype(Lineorder::lo_discount),decltype(Lineorder::lo_revenue),decltype(Lineorder::lo_supplycost),decltype(Lineorder::lo_tax),decltype(Lineorder::lo_commitdate),decltype(Lineorder::lo_shipmode)> lineorder;
    golap::ColumnTable<MEM_TYPE,decltype(Part::p_key),decltype(Part::p_name),decltype(Part::p_mfgr),decltype(Part::p_category),decltype(Part::p_brand1),decltype(Part::p_color),decltype(Part::p_type),decltype(Part::p_size),decltype(Part::p_container)> part;
    golap::ColumnTable<MEM_TYPE,decltype(Supplier::s_key),decltype(Supplier::s_name),decltype(Supplier::s_address),decltype(Supplier::s_city),decltype(Supplier::s_nation),decltype(Supplier::s_region),decltype(Supplier::s_phone)> supplier;
    golap::ColumnTable<MEM_TYPE,decltype(Customer::c_key),decltype(Customer::c_name),decltype(Customer::c_address),decltype(Customer::c_city),decltype(Customer::c_nation),decltype(Customer::c_region),decltype(Customer::c_phone),decltype(Customer::c_mktsegment)> customer;
    golap::ColumnTable<MEM_TYPE,decltype(Date::d_key),decltype(Date::d_date),decltype(Date::d_dayofweek),decltype(Date::d_month),decltype(Date::d_year),decltype(Date::d_yearmonthnum),decltype(Date::d_yearmonth),decltype(Date::d_daynuminweek),decltype(Date::d_daynuminmonth),decltype(Date::d_daynuminyear),decltype(Date::d_monthnuminyear),decltype(Date::d_weeknuminyear),decltype(Date::d_sellingseason),decltype(Date::d_lastdayinweekfl),decltype(Date::d_lastdayinmonthfl),decltype(Date::d_holidayfl),decltype(Date::d_weekdayfl)> date;
};


