#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cfloat>

#include <cub/cub.cuh>
#include <curand_kernel.h>


#include "../table.hpp"
#include "../util.hpp"


namespace golap {


template <typename T, typename FLOATTYPE>
__global__ void normalize_column(FLOATTYPE *dst, T *col, T *col_range, uint64_t num){
    for(uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        dst[idx] = ((FLOATTYPE)(col[idx]-col_range[0])) / (FLOATTYPE)(col_range[1]-col_range[0]);
    }
}

template <typename FLOATTYPE>
__global__ void compute_diffs(FLOATTYPE *col_vals, FLOATTYPE *diffs, uint64_t num_cols, uint64_t num){
    uint64_t a,b;
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < util::triangle_number(num); idx += blockDim.x * gridDim.x){
        util::triangle_coord(idx,num,a,b);
        // printf("Thread[%lu] : (%lu,%lu)\n",idx,a,b);

        for(uint32_t cur_col = 0; cur_col < num_cols; ++cur_col){
            // add the squared dimensional difference
            // (only compute squared difference, it compares equally)
            diffs[a*num + b] += (col_vals[a + cur_col*num] - col_vals[b +cur_col*num])*
                          (col_vals[a + cur_col*num] - col_vals[b +cur_col*num]);
            diffs[b*num + a] += (col_vals[a + cur_col*num] - col_vals[b +cur_col*num])*
                          (col_vals[a + cur_col*num] - col_vals[b +cur_col*num]);

        }
        __syncwarp();
    }
}

template <typename FLOATTYPE>
__global__ void random_centroids(FLOATTYPE *centroids, uint64_t num){
    curandState_t state;
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        curand_init(1234, idx, 0, &state);
        centroids[idx] = curand_uniform(&state);
    }
}

template <typename FLOATTYPE>
__global__ void random_point_centroids(FLOATTYPE *col_vals, FLOATTYPE *centroids, uint64_t k, uint64_t num_cols, uint64_t tuples_total){
    curandState_t state;
    uint64_t data_pt_idx;
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < k; idx += blockDim.x * gridDim.x){
        curand_init(1234, idx, 0, &state);
        data_pt_idx = (uint64_t)(curand_uniform(&state)*tuples_total);

        for(uint32_t cur_col = 0; cur_col < num_cols; ++cur_col){
            centroids[idx + (cur_col * k)] = col_vals[data_pt_idx + (cur_col * tuples_total)];
        }
    }
}
__global__ void iota(uint64_t *assignments, uint64_t num){
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        assignments[idx] = idx;
    }
}


__global__ void count_cluster_assigned(uint64_t *assignments, uint64_t *assigned_tuples, uint64_t num){
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        atomicAdd((unsigned long long*)&assigned_tuples[assignments[idx]], (unsigned long long)1);
    }
}
__global__ void cluster_arrange(uint64_t *sort_order, uint64_t index_start, uint64_t *assignments, uint64_t *assigned_tuples, uint64_t num){
    // rearrange the sort order based on the cluster assignments
    uint64_t insert_idx;

    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        insert_idx = atomicAdd((unsigned long long*)&assigned_tuples[assignments[idx]], 1);
        sort_order[insert_idx] = index_start + idx;
    }
}


/**
 * Assign the tuples to the given centroids and update the number of assigned tuples per centroid.
 */
template <typename FLOATTYPE>
__global__ void kmeans_assign(FLOATTYPE *col_vals, FLOATTYPE *centroids,
                                uint64_t *assignments, uint64_t *assigned_tuples, uint64_t k, uint64_t num_cols, uint64_t num){

    FLOATTYPE min_dist;
    if constexpr (std::is_same_v<FLOATTYPE,float>) min_dist = FLT_MAX;
    else if constexpr (std::is_same_v<FLOATTYPE,double>) min_dist = DBL_MAX;
    FLOATTYPE dist;
    uint64_t assignment;

    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){

        for(uint32_t cluster_idx = 0; cluster_idx < k; ++cluster_idx){
            dist = 0.0;
            for(uint32_t cur_col = 0; cur_col < num_cols; ++cur_col){
                // add the squared dimensional difference
                // (only compute squared difference, it compares equally)
                dist += (col_vals[idx + cur_col*num] - centroids[cluster_idx + cur_col * k])*
                        (col_vals[idx + cur_col*num] - centroids[cluster_idx + cur_col * k]);
            }


            if (dist < min_dist){
                assignment = cluster_idx;
                min_dist = dist;
            }
        }
        assignments[idx] = assignment;
        atomicAdd((unsigned long long*)&assigned_tuples[assignment], (unsigned long long)1);
        __syncwarp();
    }
}

/**
 * Recalculate the centroids as the mean of the assigned tuples.
 */
template <typename FLOATTYPE>
__global__ void kmeans_recalculate(FLOATTYPE *col_vals, FLOATTYPE *centroids,
                                    uint64_t* assignments, uint64_t *assigned_tuples, uint64_t k, uint64_t num_cols, uint64_t num){
    uint64_t assigned_to;
    for(uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x){
        assigned_to = assignments[idx];
        for(uint32_t cur_col = 0; cur_col < num_cols; ++cur_col){
            // update centroid atomically
            atomicAdd(&centroids[assigned_to + cur_col * k], col_vals[idx + cur_col*num] / assigned_tuples[assigned_to]);
        }
    }
}



template <typename FLOATTYPE>
struct Clustering{
    Clustering(uint64_t tuples_total, uint64_t num_cols):
        tuples_total(tuples_total),num_cols(num_cols){
        checkCudaErrors(cudaMalloc(&sort_order_dev, tuples_total*sizeof(uint64_t)));
        checkCudaErrors(cudaMallocHost(&sort_order_hst, tuples_total*sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc(&col_vals, tuples_total*num_cols*sizeof(FLOATTYPE)));
        uint64_t needed_for_min,needed_for_scan;

        // set temp_storage
        cub::DeviceReduce::Min(temp_storage, needed_for_min, (uint64_t*)nullptr, (uint64_t*)nullptr, tuples_total);
        cub::DeviceScan::ExclusiveSum(temp_storage, needed_for_scan, (uint64_t*)nullptr, (uint64_t*)nullptr, tuples_total);
        temp_storage_bytes = std::max(needed_for_min,needed_for_scan);
        checkCudaErrors(cudaMalloc(&temp_storage, temp_storage_bytes));
    }

    ~Clustering(){
        cudaFree(sort_order_dev);
        cudaFreeHost(sort_order_hst);
        cudaFree(col_vals);
        cudaFree(temp_storage);
    }

    template <typename T>
    void add_column_normalized(T *col){
        _add_column_normalized(col, std::is_arithmetic<T>{});
    }

    virtual void cluster(uint32_t rounds, uint64_t *assignments_hst = nullptr, uint64_t *assigned_each_cluster_hst = nullptr) = 0;

    FLOATTYPE *col_vals;
    uint64_t temp_storage_bytes;
    uint64_t *sort_order_hst,*sort_order_dev;
    void *temp_storage = nullptr;

    uint64_t num_cols,_added=0;
    uint64_t tuples_total;
    CStream stream{"ClusteringStream"};

protected:
    // https://stackoverflow.com/questions/30380406/c-template-for-numeric-types
    template <typename T>
    void _add_column_normalized(T *col, std::true_type){
        if (_added == num_cols) {
            printf("Error: Cant add more columns to Clustering!!\n");
        }

        T *col_dev,*col_range;
        checkCudaErrors(cudaMalloc(&col_dev,tuples_total*sizeof(T)));
        checkCudaErrors(cudaMemcpy(col_dev,col,tuples_total*sizeof(T), cudaMemcpyDefault));
        checkCudaErrors(cudaMalloc(&col_range, 2*sizeof(T)));

        checkCudaErrors(cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, col_dev, col_range+0, tuples_total, stream.stream));
        checkCudaErrors(cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, col_dev, col_range+1, tuples_total, stream.stream));
        normalize_column<<<512,512,0,stream.stream>>>(col_vals+(tuples_total*_added), col_dev, col_range, tuples_total);
        cudaStreamSynchronize(stream.stream);

        cudaFree(col_dev);
        cudaFree(col_range);
        _added += 1;
    }

    template <typename T>
    void _add_column_normalized(T *col, std::false_type){
        // non numeric column, dont do anything
        printf("Added non-numeric column for clustering, ignoring for now\n");
    }
};



template <typename FLOATTYPE = float>
struct KMeansClustering : public Clustering<FLOATTYPE>{
    KMeansClustering(uint64_t tuples_total, uint64_t num_cols, uint64_t k, bool force_same_size = false)
        :Clustering<FLOATTYPE>(tuples_total,num_cols),k(k),force_same_size(force_same_size)
            {
        
        checkCudaErrors(cudaMalloc(&centroids, k*num_cols*sizeof(FLOATTYPE)));

        printf("k=%lu\n",k);
    }

    ~KMeansClustering(){
        cudaFree(centroids);
    }

    uint64_t k;
    FLOATTYPE *centroids;

    bool force_same_size;


    void cluster(uint32_t rounds, uint64_t *assignments_hst = nullptr, uint64_t *assigned_each_cluster_hst = nullptr){
        if (this->_added != this->num_cols) {
            printf("Number of columns added and expected dont match (%lu vs %lu)\n",this->_added,this->num_cols);
            return;
        }


        uint64_t *assignments,*assigned_tuples;
        checkCudaErrors(cudaMalloc(&assignments,this->tuples_total*sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc(&assigned_tuples,k*sizeof(uint64_t)));


        // initialize centroids:
        // random_centroids<<<512,512,0,stream.stream>>>(centroids, k*num_cols);
        random_point_centroids<<<512,512,0,this->stream.stream>>>(this->col_vals, centroids, k, this->num_cols, this->tuples_total);

        for(uint32_t round = 0; round < rounds; ++round){

            checkCudaErrors(cudaMemsetAsync(assigned_tuples, 0, k*sizeof(uint64_t), this->stream.stream));
            kmeans_assign<<<512,512,0,this->stream.stream>>>(this->col_vals, centroids, assignments, assigned_tuples, k, this->num_cols, this->tuples_total);

            checkCudaErrors(cudaMemsetAsync(centroids, 0, k*this->num_cols*sizeof(FLOATTYPE), this->stream.stream));
            kmeans_recalculate<<<512,512,0,this->stream.stream>>>(this->col_vals, centroids, assignments, assigned_tuples, k, this->num_cols, this->tuples_total);
           
            checkCudaErrors(cudaStreamSynchronize(this->stream.stream));
        }
        // debug
        if(assigned_each_cluster_hst) checkCudaErrors(cudaMemcpyAsync(assigned_each_cluster_hst,assigned_tuples,k*sizeof(uint64_t),cudaMemcpyDefault, this->stream.stream));
        if(assignments_hst)checkCudaErrors(cudaMemcpyAsync(assignments_hst,assignments,this->tuples_total*sizeof(uint64_t),cudaMemcpyDefault, this->stream.stream));

        // prefix sum the assigned tuples
        checkCudaErrors(cub::DeviceScan::ExclusiveSum(this->temp_storage, this->temp_storage_bytes, assigned_tuples, assigned_tuples, k, this->stream.stream));
 
        cluster_arrange<<<512,512,0,this->stream.stream>>>(this->sort_order_dev, 0, assignments, assigned_tuples, this->tuples_total);
        checkCudaErrors(cudaStreamSynchronize(this->stream.stream));
        
        if (force_same_size){

        }

        checkCudaErrors(cudaMemcpyAsync(this->sort_order_hst,this->sort_order_dev,this->tuples_total*sizeof(uint64_t),cudaMemcpyDefault,this->stream.stream));
        checkCudaErrors(cudaStreamSynchronize(this->stream.stream));

        cudaFree(assignments);
        cudaFree(assigned_tuples);
    }

};


struct ClusterHelper{
    std::string def_string;
    std::string cluster_algo;
    uint64_t cluster_param_max_tuples;
    uint32_t cluster_param_rounds;
    uint64_t cluster_param_k;

    template <typename MEM_TYPE, typename ... ATTR>
    void apply(ColumnTable<MEM_TYPE,ATTR...> &table, std::vector<uint64_t> &chunk_size_vec){
        auto cluster_col_names = util::str_split(def_string.substr(8),"|");

        std::shared_ptr<Clustering<float>> clustering;
        if (cluster_algo == "kmeans") clustering = std::make_shared<KMeansClustering<float>>(table.num_tuples,
                                        cluster_col_names.size(), cluster_param_k);
        else {
            std::cout << "Unknown clustering algorithm: "<< cluster_algo << "\n";
            std::exit(1);
        }
        util::Log::get().debug_fmt("Will add %lu columns total.",cluster_col_names.size());

        table.apply([&](auto& a_col, uint64_t num_tuples, uint64_t col_idx){
            bool found = false;
            for (auto &col_name : cluster_col_names){
                if (a_col.attr_name == col_name){
                    found = true;
                    break;
                }
            }
            if (!found) return;
            using COL_TYPE = typename std::remove_reference<decltype(a_col)>::type::value_t;
            util::Log::get().info_fmt("Adding column %s to cluster features.",a_col.attr_name.c_str());
            clustering->add_column_normalized(a_col.data());
        });
        HostMem assigned_each_cluster_hst{Tag<uint64_t>{}, 50000};

        clustering->cluster(cluster_param_rounds, nullptr, assigned_each_cluster_hst.ptr<uint64_t>());

        std::vector<uint64_t> sort_order(clustering->sort_order_hst,clustering->sort_order_hst+table.num_tuples);

        // if(use_variable_chunks){
        chunk_size_vec.clear();
        uint64_t tmp,cur;
        for (int cluster_idx = 0; cluster_idx < cluster_param_k; ++cluster_idx){
            tmp = assigned_each_cluster_hst.ptr<uint64_t>()[cluster_idx];

            // cut large chunks down to max size
            while(tmp > 0){
                cur = std::min(tmp,cluster_param_max_tuples);
                chunk_size_vec.push_back(cur);
                printf("%lu, ",cur);
                tmp -= cur;
            }

        }
        printf("\n");
        printf("%lu chunks after\n",chunk_size_vec.size());
        // }
        table.sort(sort_order);
    }
};



} // end of namespace
