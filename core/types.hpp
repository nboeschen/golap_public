#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#include "util.hpp"

namespace util {

/**
 * A wrapper for a char array, to make it fit the next power of two.
 */

template <typename T>
struct Padded{
    T d;
    using value_t = T;

    __host__ __device__
    Padded(){}
    __host__
    Padded(const char *ptr){
        strncpy(d, ptr, sizeof(T));
    }

#ifdef __CUDA_ARCH__
    __device__
    int cmp(const Padded<T>& other) const{
        for(int i = 0; i<sizeof(T); ++i){
            char comp = d[i] - other.d[i];
            if(comp != 0) return comp;
            if(d[i] == '\0') return 0;
        }
        printf("Reached end of loop in Padded char[] compare, shouldnt ever happen!\n");
        return 0;
    }
    __device__
    bool operator==(const Padded<T>& other) const{
        return cmp(other) == 0;
    }
    __device__
    bool operator!=(const Padded<T>& other) const{
        return !(*this==other);
    }
    __device__
    bool operator<(const Padded<T>& other) const{
        return cmp(other) < 0;
    }
    __device__
    bool operator<=(const Padded<T>& other) const{
        return cmp(other) <= 0;
    }
    __device__
    bool operator>=(const Padded<T>& other) const{
        return cmp(other) >= 0;
    }
#else
    bool operator<(const Padded<T>& other) const{
        return std::string(d) < std::string(other.d);
    }

    bool operator==(const Padded<T>& other) const{
        return std::string(d) == std::string(other.d);
    }

    bool operator!=(const Padded<T>& other) const{
        return std::string(d) != std::string(other.d);
    }
#endif //__CUDA_ARCH__

    __host__ __device__
    size_t operator-(const Padded<T>& other) const{
        size_t res = 0;
        size_t cur_mul = 1;
        for (int i = sizeof(T)-2; i >= 0; --i){
            res += cur_mul * (d[i]-other.d[i]);
            cur_mul *= 95;
        }
        return res;
    }

    // future: check
    __host__ __device__
    operator double() const {
        double res;
        size_t cur_mul = 1;
        for (int i = sizeof(T)-2; i >= 0; --i){
            res += cur_mul * d[i];
            cur_mul *= 95;
        }
        return res;
    }


    __host__ __device__
    uint64_t hash() const{
        uint64_t res = 0;
        uint64_t ppow = 1;
        constexpr uint64_t p = 97;

        for(int i = 0; i<sizeof(T); ++i){
            if(d[i] == '\0') break;
            res += d[i] * ppow;
            ppow *= p;
        }
        return res;
    }

    // FUTURE:
    // uint64_t fast_hash() const{
    //     return hash();
    // }

    friend std::istream& operator>>(std::istream& in, Padded<T>& obj){
        std::string tmp;
        std::getline(in,tmp);

        if (tmp.size()+1 > sizeof(T)){
            util::Log::get().warn_fmt("\'%s\' is too long to fit into buffer of size %ld, will not be null-terminated!",
                                    tmp.c_str(), sizeof(T));
        }
        strncpy(obj.d, tmp.c_str(), sizeof(T));
        return in;
    }

    friend std::ostream& operator<<(std::ostream &out, Padded<T> const& obj){
        out << obj.d;
        return out;
    }
private:
    char __padding[nextP2(sizeof(T))-sizeof(T)] = {};
};


struct Datetime{
    std::time_t t;

    __host__ __device__
    int64_t cmp(const Datetime& other) const{
        return t - other.t;
    }
    __host__ __device__
    int64_t operator-(const Datetime& other) const{
        return cmp(other);
    }
    __host__ __device__
    bool operator==(const Datetime& other) const{
        return cmp(other) == 0;
    }
    __host__ __device__
    bool operator!=(const Datetime& other) const{
        return !(*this==other);
    }
    __host__ __device__
    bool operator<(const Datetime& other) const{
        return cmp(other) < 0;
    }
    __host__ __device__
    bool operator<=(const Datetime& other) const{
        return cmp(other) <= 0;
    }
    __host__ __device__
    bool operator>=(const Datetime& other) const{
        return cmp(other) >= 0;
    }
    __host__ __device__
    uint64_t hash() const{
        return (uint64_t) t;
    }

    __host__ __device__
    operator double() const { return double(t); }

    friend std::istream& operator>>(std::istream& in, Datetime& obj){
        std::tm tm{};
        in >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        obj.t = std::mktime(&tm);
        return in;
    }
    friend std::ostream& operator<<(std::ostream &out, Datetime const& obj){
        std::tm tm = *std::gmtime(&obj.t);;
        out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return out;
    }
};

struct Date{
    std::time_t t;
    // future: fix redundancy

    __host__ __device__
    int64_t cmp(const Date& other) const{
        return t - other.t;
    }
    __host__ __device__
    int64_t operator-(const Date& other) const{
        return cmp(other);
    }
    __host__ __device__
    bool operator==(const Date& other) const{
        return cmp(other) == 0;
    }
    __host__ __device__
    bool operator!=(const Date& other) const{
        return !(*this==other);
    }
    __host__ __device__
    bool operator<(const Date& other) const{
        return cmp(other) < 0;
    }
    __host__ __device__
    bool operator<=(const Date& other) const{
        return cmp(other) <= 0;
    }
    __host__ __device__
    bool operator>=(const Date& other) const{
        return cmp(other) >= 0;
    }
    __host__ __device__
    uint64_t hash() const{
        return (uint64_t) t;
    }

    __host__ __device__
    operator double() const { return double(t); }

    friend std::istream& operator>>(std::istream& in, Date& obj){
        std::tm tm{};
        in >> std::get_time(&tm, "%Y-%m-%d");
        obj.t = std::mktime(&tm);
        return in;
    }
    friend std::ostream& operator<<(std::ostream &out, Date const& obj){
        std::tm tm = *std::gmtime(&obj.t);;
        out << std::put_time(&tm, "%Y-%m-%d");
        return out;
    }
};


struct Decimal32{
    uint32_t val;
    inline static char decimal_delimiter = '.';

    __host__ __device__
    int64_t cmp(const Decimal32& other) const{
        return val - other.val;
    }
    __host__ __device__
    Decimal32 operator+(const Decimal32& other) const{
        return Decimal32{this->val+other.val};
    }
    __host__ __device__
    Decimal32 operator-(const Decimal32& other) const{
        return Decimal32{this->val-other.val};
    }
    __host__ __device__
    Decimal32 operator*(const Decimal32& other) const{
        return Decimal32{(this->val*other.val)/100};
    }
    __host__ __device__
    bool operator==(const Decimal32& other) const{
        return cmp(other) == 0;
    }
    __host__ __device__
    bool operator!=(const Decimal32& other) const{
        return !(*this==other);
    }
    __host__ __device__
    bool operator<(const Decimal32& other) const{
        return cmp(other) < 0;
    }
    __host__ __device__
    bool operator<=(const Decimal32& other) const{
        return cmp(other) <= 0;
    }
    __host__ __device__
    bool operator>=(const Decimal32& other) const{
        return cmp(other) >= 0;
    }
    __host__ __device__
    uint64_t hash() const{
        return (uint64_t) val;
    }

    __host__ __device__
    operator double() const { return double(val); }

    friend std::istream& operator>>(std::istream& in, Decimal32& obj){
        std::string s;
        std::getline(in,s);
        auto decimalPos = s.find('.');

        obj.val = std::stoul(s.substr(0, decimalPos)) * 100;
        if (decimalPos != std::string::npos && decimalPos != s.size()-1){
            obj.val += std::stoul(s.substr(decimalPos+1) + std::string(2 - s.size() + decimalPos + 1, '0'));
        }
        return in;
    }

    friend std::ostream& operator<<(std::ostream &out, Decimal32 const& obj){
        out << obj.val / 100 << Decimal32::decimal_delimiter << std::setfill('0') << std::setw(2) << obj.val % 100 << std::setfill(' ');
        return out;
    }
};

struct Decimal64{
    uint64_t val;
    inline static char decimal_delimiter = '.';

    __host__ __device__
    int64_t cmp(const Decimal64& other) const{
        return val - other.val;
    }
    __host__ __device__
    Decimal64 operator+(const Decimal64& other) const{
        return Decimal64{this->val+other.val};
    }
    __host__ __device__
    Decimal64 operator-(const Decimal64& other) const{
        return Decimal64{this->val-other.val};
    }
    __host__ __device__
    Decimal64 operator*(const Decimal64& other) const{
        return Decimal64{(this->val*other.val)/100};
    }
    __host__ __device__
    bool operator==(const Decimal64& other) const{
        return cmp(other) == 0;
    }
    __host__ __device__
    bool operator!=(const Decimal64& other) const{
        return !(*this==other);
    }
    __host__ __device__
    bool operator<(const Decimal64& other) const{
        return cmp(other) < 0;
    }
    __host__ __device__
    bool operator<=(const Decimal64& other) const{
        return cmp(other) <= 0;
    }
    __host__ __device__
    bool operator>=(const Decimal64& other) const{
        return cmp(other) >= 0;
    }
    __host__ __device__
    uint64_t hash() const{
        return (uint64_t) val;
    }

    __host__ __device__
    operator double() const { return double(val); }

    friend std::istream& operator>>(std::istream& in, Decimal64& obj){
        std::string s;
        std::getline(in,s);
        auto decimalPos = s.find('.');

        obj.val = std::stoul(s.substr(0, decimalPos)) * 100;
        if (decimalPos != std::string::npos && decimalPos != s.size()-1){
            obj.val += std::stoul(s.substr(decimalPos+1) + std::string(2 - s.size() + decimalPos + 1, '0'));
        }
        return in;
    }

    friend std::ostream& operator<<(std::ostream &out, Decimal64 const& obj){
        out << obj.val / 100 << Decimal64::decimal_delimiter << std::setfill('0') << std::setw(2) << obj.val % 100 << std::setfill(' ');
        return out;
    }
};

} // end of namespace

namespace std {
    template <typename T>
    struct hash<util::Padded<T>> {
        size_t operator()(util::Padded<T> const& p) const {
            return p.hash();
        }
    };

    template <>
    struct hash<util::Datetime> {
        size_t operator()(util::Datetime const& d) const {
            return d.hash();
        }
    };
} // end of namespace std
