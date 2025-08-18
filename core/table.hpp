#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <memory>
#include <execution>


#include "helper_cuda.h"

#include "util.hpp"
#include "mem.hpp"

namespace golap{

class Table{
public:
    Table(std::string name): name(name){}

    void add_attribute(std::string name){
        attributes.emplace_back(name);
    }

    std::string describe(){
        std::stringstream ss;
        ss << name << "(";
        for (auto &col_name: attributes){
            ss << col_name << ", ";
        }
        ss << ')';
        return ss.str();
    }
    
    /**
     * Should return the *allocated* memory size of the table.
     */
    virtual uint64_t size_bytes() = 0;

    std::string name;
    std::vector<std::string> attributes;
    uint64_t num_tuples = 0;
    uint64_t num_slots = 0;
};



/**
 * A fixed type version of Memory
 */
template <typename MEM_TYPE, typename T>
class Column : public MEM_TYPE{
public:
    Column(uint64_t num, uint64_t alloc_unit, std::string attr_name = ""):MEM_TYPE(golap::Tag<T>{}, num, alloc_unit),
        attr_name(attr_name){}

    using value_t = T;
    static const uint64_t value_size = sizeof(T);

    void resize_col(uint64_t num, uint64_t alloc_unit = 4096){
        return MEM_TYPE::template resize_num<T>(num, alloc_unit);
    }

    uint64_t size(){
        return MEM_TYPE::template size<T>();
    }
    T* data(){
        return MEM_TYPE::template ptr<T>();
    }

    void transfer(T *other, uint64_t num, cudaStream_t stream = 0){
        if(stream == 0){
            checkCudaErrors(cudaMemcpy((void*)other, (void*)data(), num*sizeof(T), cudaMemcpyDefault));
        }else {
            checkCudaErrors(cudaMemcpyAsync((void*)other, (void*)data(), num*sizeof(T), cudaMemcpyDefault, stream));
        }
    }

    std::string attr_name;
};

template<uint64_t idx, typename T>
struct GetHelper;

/**
 * Base case of chained columns
 */
template<typename MEM_TYPE, typename ... T>
struct Columns{
    Columns(std::string attr_string, uint64_t num, uint64_t alloc_unit = 4096){}
    uint64_t to_bin_col(std::ofstream &stream, uint64_t num_tuples){return 0;}
    uint64_t from_bin_col(std::ifstream &stream, uint64_t num_tuples){return 0;}
    void sort(std::vector<uint64_t> &sort_order){}
    template <typename F>
    void apply(F&& f, uint64_t num_tuples, uint64_t col_idx){}
    uint64_t size_bytes(){return 0;}
    void to_csv(std::ostream &stream, std::string delimiter, uint64_t idx){
        stream << '\n';
    }
    void resize(uint64_t num, uint64_t alloc_unit = 0){}
};

/**
 * General case of chained columns
 */
template<typename MEM_TYPE, typename T, typename ... Rest>
struct Columns<MEM_TYPE, T, Rest ...>{

    Columns(std::string attr_string, uint64_t alloc_num, uint64_t alloc_unit = 4096):
            first(alloc_num,alloc_unit,util::str_split(attr_string,",")[0]),
            rest(attr_string.length() == first.attr_name.length() ? "" : attr_string.substr( first.attr_name.length()+1),alloc_num,alloc_unit){

    }
    template<uint64_t idx>
    auto& col(){
        return GetHelper<idx, Columns<MEM_TYPE,T,Rest...>>::col(*this);
    }
    uint64_t size_bytes(){
        uint64_t result = first.size_bytes();
        result += rest.size_bytes();
        return result;
    }

    template <typename F>
    void apply(F&& f, uint64_t num_tuples, uint64_t col_idx){
        f(first, num_tuples, col_idx);
        rest.apply(f, num_tuples, col_idx+1);
    }

    void to_csv(std::ostream &stream, std::string delimiter, uint64_t idx){
        if constexpr (std::is_same_v<T,uint8_t>){
            stream << + first.data()[idx];
        }else{
            stream << first.data()[idx];
        }
        stream << delimiter;
        rest.to_csv(stream,delimiter, idx);
    }

    uint64_t to_bin_col(std::ofstream &stream, uint64_t num_tuples){
        uint64_t written = sizeof(T)*num_tuples;
        stream.write((char*) first.data(), sizeof(T)*num_tuples);
        return written + rest.to_bin_col(stream,num_tuples);
    }

    uint64_t from_bin_col(std::ifstream &stream, uint64_t num_tuples){
        uint64_t read = sizeof(T)*num_tuples;
        stream.read((char*) first.data(), sizeof(T)*num_tuples);
        return read + rest.from_bin_col(stream,num_tuples);
    }

    void sort(std::vector<uint64_t> &sort_order){
        Column<MEM_TYPE,T> first_copy(sort_order.size(),1);
        first.transfer(first_copy.data(),sort_order.size());
        for(uint64_t i = 0; i<sort_order.size(); ++i){
            // first.data()[i] = first_copy.data()[sort_order[i]];
            memcpy(first.data()+i, first_copy.data() + sort_order[i], sizeof(T));
        }

        rest.sort(sort_order);
    }

    void resize(uint64_t num, uint64_t alloc_unit = 0){
        first.resize_col(num, alloc_unit);
        rest.resize(num, alloc_unit);
    }

    Column<MEM_TYPE, T> first;
    Columns<MEM_TYPE, Rest ... > rest;
};

template<typename MEM_TYPE, typename T, typename ... Rest>
struct GetHelper<0, Columns<MEM_TYPE, T, Rest ... >>{
    static Column<MEM_TYPE, T>& col(Columns<MEM_TYPE, T, Rest...>& data){
        return data.first;
    }
};
template<uint64_t idx, typename MEM_TYPE, typename T, typename ... Rest>
struct GetHelper<idx, Columns<MEM_TYPE, T, Rest ... >>{
    static auto& col(Columns<MEM_TYPE, T, Rest...>& data){
        return GetHelper<idx-1, Columns<MEM_TYPE, Rest ...>>::col(data.rest);
    }
};
/**
 *
 */
template<typename MEM_TYPE, typename ... T>
struct ColumnTable : public Table{
    ColumnTable(std::string attr_string, uint64_t alloc_num, uint64_t alloc_unit = 4096):
            ColumnTable("UnnamedTable",attr_string,alloc_num,alloc_unit){}
    ColumnTable(std::string tbl_name, std::string attr_string, uint64_t alloc_num, uint64_t alloc_unit = 4096):
            Table(tbl_name),
            attr_string(attr_string),
            columns(attr_string,alloc_num,alloc_unit)
    {
        this->num_slots = alloc_num;
        for(auto &attr_name : util::str_split(attr_string,",")){
            this->add_attribute(attr_name);
        }
    }

    template<uint64_t idx>
    auto& col(){
        return GetHelper<idx, Columns<MEM_TYPE,T...>>::col(columns);
    }

    uint64_t size_bytes(){
        return columns.size_bytes();
    }

    void to_csv(std::ostream &stream, std::string delimiter, uint64_t start_idx = 0, uint64_t end_idx = 0){
        if (end_idx == 0) end_idx = num_tuples;
        stream << util::str_replace(attr_string,",",delimiter) << '\n';
        for (; start_idx < std::min(end_idx,num_tuples); ++start_idx){
            columns.to_csv(stream,delimiter,start_idx);
        }
    }

    void from_csv(std::istream &stream, std::string delimiter, uint64_t start_idx = 1, uint64_t end_idx = 0){
        // if (end_idx == 0) end_idx = start_idx + this->num_slots;
        if (end_idx != 0 && end_idx - start_idx > this->num_slots){
            util::Log::get().info_fmt("Cant read the specified line range [%lu,%lu), reading [%lu,%lu) instead",
                start_idx,end_idx,start_idx,start_idx + this->num_slots);
            end_idx = start_idx + this->num_slots;
        }

        uint64_t line_idx = 0;
        std::string line;
        std::vector<std::string> line_split;

        while(std::getline(stream,line)){
            line_split = util::str_split(line,delimiter,true);

            if ((end_idx != 0 && line_idx >= end_idx)) break;

            if (this->num_tuples == this->num_slots) {
                util::Log::get().warn_fmt("Stopping csv reader at line idx %lu, no slots left (%lu total)",line_idx,this->num_slots);
                break;
            }

            if(line_idx < start_idx) {
                line_idx += 1;
                continue;
            }

            apply([&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
                // if the csv has empty columns, dont set a value
                if (col_idx >= line_split.size()) return;

                std::istringstream ss(line_split[col_idx]);
                using COL_TYPE = typename std::remove_reference<decltype(a_col)>::type::value_t;

                if constexpr (std::is_same_v<COL_TYPE,uint8_t>){
                    uint16_t tmp;
                    ss >> tmp;
                    a_col.data()[this->num_tuples] = static_cast<uint8_t>(tmp);
                }else{
                    if constexpr (std::is_same_v<COL_TYPE,char>){
                        if(line_split[col_idx] == " ") {
                            a_col.data()[this->num_tuples] = ' ';
                            return;
                        }
                    }
                    ss >> a_col.data()[this->num_tuples];
                }
            });

            this->num_tuples += 1;
            line_idx += 1;
        }
    }

    template<typename F>
    void apply(F&& f){
        columns.apply(f, num_tuples, 0);
    }

    uint64_t to_bin_col(std::ofstream &stream){
        return columns.to_bin_col(stream, num_tuples);
    }

    uint64_t from_bin_col(std::ifstream &stream){
        return columns.from_bin_col(stream, num_tuples);
    }

    void resize(uint64_t num, uint64_t alloc_unit = 0){
        Table::num_slots = num;
        columns.resize(num, alloc_unit);
    }

    void sort(std::vector<uint64_t> &sort_order){
        columns.sort(sort_order);
    }

    template <uint64_t idx>
    bool compare(uint64_t left, uint64_t right){
        return col<idx>().data()[left] < col<idx>().data()[right];
    }
    template<uint64_t first, uint64_t second, uint64_t ... idx_rest>
    bool compare(uint64_t left, uint64_t right){
        if (col<first>().data()[left] == col<first>().data()[right]){
            return compare<second,idx_rest...>(left,right);
        }else{
            return col<first>().data()[left] < col<first>().data()[right];
        }
    }

    template<uint64_t ... idxs>
    std::vector<uint64_t> sort_by(bool asc = true, bool dry_run = false){
        auto f = [&](uint64_t left, uint64_t right) -> bool {
            // sort indices according to corresponding array element
            return compare<idxs...>(left,right);
        };
        std::vector<uint64_t> sort_order(num_tuples);
        std::iota(sort_order.begin(), sort_order.end(), 0);
        std::sort(std::execution::par_unseq, sort_order.begin(), sort_order.end(), f);

        if (!asc) std::reverse(sort_order.begin(),sort_order.end());
        if (!dry_run) columns.sort(sort_order);
        return sort_order;
    }

    void shuffle_ratio(double ratio){
        uint64_t swaps = (uint64_t) (ratio * num_tuples);
        std::vector<uint64_t> sort_order(num_tuples);
        std::iota(sort_order.begin(), sort_order.end(), 0);
        uint64_t seed = util::Timer::time_seed();
        uint64_t a,b,tmp;
        for (uint64_t i = 0; i<swaps; ++i){
            a = util::uniform_int(seed,0,num_tuples);
            b = util::uniform_int(seed,0,num_tuples);

            tmp = sort_order[a];
            sort_order[a] = sort_order[b];
            sort_order[b] = tmp;

        }
        columns.sort(sort_order);
    }

    std::vector<uint64_t> sort_random(){
        std::vector<uint64_t> sort_order(num_tuples);
        std::iota(sort_order.begin(), sort_order.end(), 0);
        std::random_shuffle(sort_order.begin(), sort_order.end());
        columns.sort(sort_order);
        return sort_order;
    }


    void print_col_histogram(){
        auto fun = [&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
            using COL_TYPE = typename std::remove_reference<decltype(a_col)>::type::value_t;
            printf("#[INFO ] Values in column %lu,%s: ",col_idx,a_col.attr_name.c_str());

            std::unordered_map<COL_TYPE,uint64_t> values;
            for(uint64_t i = 0; i < num_tuples; ++i){
                values[a_col.data()[i]] += 1;
            }

            // for now only print the number of values
            printf("#vals=%lu\n",values.size());
        };
        apply(fun);
    }


    Columns<MEM_TYPE, T ...> columns;
    std::string attr_string;
};


} // end of namespace

