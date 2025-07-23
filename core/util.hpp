#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <stdarg.h>
#include <stdlib.h>
#include <sched.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iomanip>

// make some functions usable for both gcc and nvcc
#ifndef __host__
#define __host__
#endif //__host__
#ifndef __device__
#define __device__
#endif //__device__


namespace util{
/**
 * https://stackoverflow.com/a/34519373
 */
template<typename T>
class BetterSingleton{
protected:
    BetterSingleton() noexcept = default;
    BetterSingleton(const BetterSingleton&) = delete;
    BetterSingleton& operator=(const BetterSingleton&) = delete;

    virtual ~BetterSingleton() = default; // to silence base class BetterSingleton<T> has a
    // non-virtual destructor [-Weffc++]

public:
    static T& get() noexcept(std::is_nothrow_constructible<T>::value){
        // Guaranteed to be destroyed.
        // Instantiated on first use.
        // Thread safe in C++11
        static T instance{};

        return instance;
    }
};

/**
 * Singleton helper
 */
template <typename T>
struct Singleton {
    static T& get() {
        static T instance;
        return instance;
    }
};
/**
 * Simple Logger.
 */
struct Log : Singleton<Log> {
    enum LogLevel{
        INFO,
        WARN,
        ERROR
    };
    /**
     * The current loglevel.
     * To use DEBUG, compile in debug mode.
     */
    LogLevel lvl = INFO;

    void debug(std::string str){
        (void) str;
        (void) lvl;
#ifndef NDEBUG
        std::cout << "#[DEBUG] " << str << '\n';
#endif //NDEBUG
    }
    void debug_fmt(const char* format...){
#ifndef NDEBUG
        std::cout << "#[DEBUG] ";
        va_list argptr;
        va_start(argptr, format);
        vprintf(format, argptr);
        va_end(argptr);
        std::cout << '\n';
#endif //NDEBUG
    }
    void info(std::string str){
        log(INFO, str);
    }
    void info_fmt(const char* format...){
        std::cout << "#[INFO ] ";
        va_list argptr;
        va_start(argptr, format);
        vprintf(format, argptr);
        va_end(argptr);
        std::cout << '\n';
    }
    void warn(std::string str){
        log(WARN, str);
    }
    void warn_fmt(const char* format...){
        std::cout << "#[WARN ] ";
        va_list argptr;
        va_start(argptr, format);
        vprintf(format, argptr);
        va_end(argptr);
        std::cout << '\n';
    }
    void error(std::string str){
        log(ERROR, str);
    }
    void error_fmt(const char* format...){
        std::cout << "#[ERROR] ";
        va_list argptr;
        va_start(argptr, format);
        vprintf(format, argptr);
        va_end(argptr);
        std::cout << '\n';
    }
    /**
     * Can be used directly, but will produce code for debug level!
     */
    void log(LogLevel level, std::string str){
        if(level >= lvl) std::cout << l2s(level) << str << '\n';
    }
private:
    inline static std::string l2s(LogLevel level) {
        switch(level) {
            case INFO:  return "#[INFO ] ";
            case WARN:  return "#[WARN ] ";
            case ERROR: return "#[ERROR] ";
            default:    return "#[what?] ";
        }
    }
};

/**
 * ANSI Escape sequences for colors.
 */
char const* FG[] = {
    "\033[0m",
    "\x1B[31m",
    "\x1B[32m",
    "\x1B[93m",
    "\x1B[34m",
    "\x1B[35m",
};
char const* BG[] = {
    "\033[0m",
    "\033[3;41;30m",
    "\033[3;42;30m",
    "\033[3;43;30m",
    "\033[3;44;30m",
    "\033[3;45;30m",
};
enum COLOR{
    RESET = 0,
    RED = 1,
    GREEN = 2,
    YELLOW = 3,
    BLUE = 4,
    PURPLE = 5,
};

/**
 * Get a random uint32 between a (inclusive) and b (exclusive).
 */
static inline uint32_t uniform_int(uint64_t &seed, uint32_t a, uint32_t b){
    seed = seed * 1103515245 + 12345; // LCG with standard params
    // return (uint32_t) (seed >> 32) % (b-a) + a; //modulo version. slower for non-pow2 (b-a)s
    return (uint32_t) (((seed>>32) * (b-a)) >> 32) + a; // multiply and shift version
}

/**
 * A simple timer using the systems steady clock.
 */
class Timer{
private:
    // Type aliases to make accessing nested type easier
    using clock_t = std::chrono::steady_clock;
    using ms_t = std::chrono::duration<double, std::ratio<1,1000>>;

    std::chrono::time_point<clock_t> m_beg;

public:
    Timer() : m_beg(clock_t::now()){}

    void reset(){
        m_beg = clock_t::now();
    }

    double elapsed() const{
        return std::chrono::duration_cast<ms_t>(clock_t::now() - m_beg).count();
    }

    static void sleep_ms(uint32_t ms){
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    static uint64_t time_seed(){
        return (uint64_t) std::chrono::duration_cast<ms_t>(clock_t::now().time_since_epoch()).count();
    }

};

/**
 * Threadpool
 */
class ThreadPool {
    std::vector<std::thread> threads;

public:
    /**
     * Example Usage:
     * ThreadPool pool;
     * pool.parallel_n(8, [&](std::stop_token, int tid) {
     *     std::cout << "Hello, here is thread " << tid << '\n';
     * });
     * pool.join();
     */
    template <typename Fn>
    void parallel_n(int n, Fn&& fn) {
        threads.reserve(n);
        for (int i = 0; i < n; ++i) {
            threads.emplace_back(std::thread(fn, i));
        }
    }

    void join() {
        for (auto& t : threads) {
            t.join();
        }
    }
};

static int pin_thread(int core_id, int pid = 0){
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    return sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset);
}

struct RangeHelper {
    /**
     * Example usage:
     * Split 5 work items into two chunks
     * RangeHelper::nth_chunk(0, 5, 2, 0)  -> 0,3
     * RangeHelper::nth_chunk(0, 5, 2, 1)  -> 3,5
     */
    static auto nth_chunk(uint64_t start, uint64_t end, uint64_t chunks, uint64_t n) {
        // ensure(start <= end);

        uint64_t total_size = end - start;
        uint64_t left = total_size % chunks;
        uint64_t p_size = (total_size - left) / chunks;

        uint64_t offset = start + std::min(n, left);

        uint64_t s = offset + p_size * n;
        uint64_t e = offset + p_size * (n + 1) + (n < left);


        return std::make_pair(s, e);
    }
};


/**
 * Divide and ceil, e.g. for number of needed runs to complete num items using groups of denom.
 */
static long long div_ceil(long long num, long long denom){
    std::lldiv_t res = std::div(num, denom);
    return res.rem ? (res.quot + 1) : res.quot;
}
/**
 * Returns the next long long with specified alignment.
 */
static long long next(long long ptr, long long alignment){
    std::lldiv_t res = std::div(ptr, alignment);
    if (res.rem == 0){
        return ptr;
    }
    return ((res.quot+1) * alignment);
}
/**
 * Returns the distance to the next long long with specified alignment
 */
static long long to_next(long long ptr, long long alignment){
    return next(ptr,alignment) - ptr;
}
/**
 * Returns the next power of two larger equal x
 */
constexpr size_t nextP2(size_t x){
    size_t power = 1;
    while(power < x) {
        power<<=1;
    }
    return power;
}

/**
 * Compute the triangle number (sum of all previous integers).
 */
__host__ __device__ static inline uint64_t triangle_number(uint64_t num){
    return (num*num - num) >> 1;
}
/**
 * Compute the 2d coords of an upper triangle matrix given a flattened index.
 */
__host__ __device__ static inline void triangle_coord(uint64_t idx, uint64_t n, uint64_t &a, uint64_t &b){
#ifdef __CUDA_ARCH__
    a = n - 2 - (uint64_t) floorf(sqrtf(-8.0*idx + 4*n*(n-1)-7)/2.0 -0.5);
#else
    a = n - 2 - (uint64_t) std::floor(std::sqrt(-8.0*idx + 4*n*(n-1)-7)/2.0 -0.5);
#endif //__CUDA_ARCH__
    b = idx + a + 1 + triangle_number(n-a) - triangle_number(n);
}


/**
 * Get sequential slices, e.g. to assign work to workers.
 * If theres less work x than workers n, assign to the first x workers.
 * Else, assign a slice of x//n + 1 to the first x%n, and x//n to the rest.
 * E.g. for a work=5, workers=3 --> [(0,2),(2,4),(4,5)]
 */
struct SliceSeq{
    SliceSeq(uint64_t work, uint64_t workers):work(work),workers(workers),
                        slice_len(work / workers),rest(work%workers){}
    void get(uint64_t &slicestart, uint64_t &sliceend){
        if (cur_start>=work){
            slicestart = 0;
            sliceend = 0;
            return;
        }

        slicestart = cur_start;
        sliceend = cur_start + slice_len;
        if (rest!=0){
            sliceend += 1;
            rest -= 1;
        }
        cur_start = sliceend;
    }
    uint64_t work,workers,slice_len,rest,cur_start=0;
};

/**
 * Get num samples in the integer range [min,max). 
 */
static std::unordered_set<uint32_t> sample_range(uint32_t min, uint32_t max, uint64_t num, uint64_t seed=0xFEFECDCD){
    if (max-min < num) throw std::runtime_error("Sample Range: Range not large enough to sample num items!");
    
    std::unordered_set<uint32_t> res;
    res.reserve(num);
    uint32_t cur;
    while (res.size() != num){
        cur = uniform_int(seed, min, max);
        if (res.find(cur) != res.end()) continue;

        res.insert(cur);
    }
    return res;
}


/**
 * Argsort.
 * https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
 */
template<typename T>
static std::vector<uint64_t> argsort(T* array, uint64_t num) {
    std::vector<uint64_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](uint64_t left, uint64_t right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}
/**
 * Shuffle/Sort/Permute a vector given a specific order/shuffle/...
 */
template<typename T>
static void sort(std::vector<T> &values, std::vector<uint64_t> &sort_order) {
    std::vector<T> copy(values);
    uint64_t idx = 0;
    for (auto& val: values){
        val = copy[sort_order[idx]];
        idx += 1;
    }
}



/**
 * Remove val from vector
 */
template<typename T>
static void remove_val(std::vector<T> &vec, T val){
    vec.erase(std::remove(vec.begin(), vec.end(), val), vec.end());
}

/**
 * Split string at delim.
 */
static std::vector<std::string> str_split(std::string s, std::string delim, bool allow_empty = false) {
    std::vector<std::string> found;
    size_t prev = 0, pos = 0, delim_len = delim.length();
    do{
        pos = s.find(delim, prev);
        if (pos == std::string::npos) pos = s.length();
        std::string token = s.substr(prev, pos-prev);
        if (!token.empty() || allow_empty) found.push_back(token);
        prev = pos + delim_len;
    }while (pos < s.length() && prev < s.length());

    return found;
}
/**
 * Returns a new string, where every occurence of @p search is replaced by @p replace.
 */
static std::string str_replace(std::string s, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = s.find(search, pos)) != std::string::npos) {
         s.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return s;
}

/**
 * Interactive prompt
 */
bool ask(std::string prompt, std::string &out, std::string BREAK = "break"){
    std::cout << prompt;
    std::cin >> out;
    return out != BREAK;
}


/**
 * CPP 17 helper
 * https://stackoverflow.com/a/42844629
 */
static bool ends_with(std::string_view str, std::string_view suffix){
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}
static bool starts_with(std::string str, std::string_view prefix){
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

/**
 * Read /self/proc/<type>, and return the value of the specified key @which (e.g. rchar or wchar).
 */
static uint64_t get_proc_info(std::string type, std::string which){
    std::ifstream in("/proc/self/"+type);
    std::string line;
    while(std::getline(in,line)){
        auto parts = str_split(line,":");
        if (parts[0] == which){
            return std::stoull(parts[1]);
        }
    }
    return (uint64_t) -1;
}

static void drop_file_cache(std::string path){
    // dd of=<file> oflag=nocache conv=notrunc,fdatasync count=0
    std::stringstream ss;
    ss << "find "<<path << " -type f -exec dd of={} oflag=nocache conv=notrunc,fdatasync count=0 > /dev/null 2>&1 \\;";
    // std::cout << ss.str() << "\n";

    auto returned = system(ss.str().c_str());

    if(returned){
        util::Log::get().error_fmt("\"%s\" returned: %d",ss.str().c_str(), returned);
    }
}


} // end of namespace
