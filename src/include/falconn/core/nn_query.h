#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>

#include "../falconn_global.h"
#include "heap.h"
#include "metadata_store.h"

namespace falconn {
namespace core {

class NearestNeighborQueryError : public FalconnError {
 public:
  NearestNeighborQueryError(const char* msg) : FalconnError(msg) {}
};

template <typename LSHTableQuery, typename LSHTablePointType,
          typename LSHTableKeyType, typename ComparisonPointType,
          typename DistanceType, typename DistanceFunction,
          typename DataStorage>
class NearestNeighborQuery {
 public:
  NearestNeighborQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage,
                       const std::unordered_map<int,std::vector<int>>& metadata_storage,
                       const std::unordered_map<int, std::vector<int>>& small_labels_store
                       )
      : table_query_(table_query), data_storage_(data_storage), metadata_storage_(metadata_storage), small_labels_store_(MetadataStore(small_labels_store)) {}

  LSHTableKeyType find_nearest_neighbor(const LSHTablePointType& q,
                                        const ComparisonPointType& q_comp,
                                        const std::set<int>& q_filter,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        int_fast64_t max_iterations) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // printf("HELLO WHAT");

    auto distance_start_time = std::chrono::high_resolution_clock::now();
    LSHTableKeyType best_key = -1;
    bool no_distance_found = true;
    int smallest_label = -1;
    int smallest_size = -1;
    for (std::set<int>::iterator it=q_filter.begin(); it!=q_filter.end(); ++it) {
      int size = small_labels_store_.get_indices_for_label(*it).size();
      if(size != 0 && (size<smallest_size || smallest_size == -1)) {
        smallest_label = *it;
        smallest_size = size;
      }
    }

    // printf("small label: %d\n", smallest_label);
    if(smallest_label != -1)
    {
      std::vector<int> indices = small_labels_store_.get_indices_for_label(smallest_label);
      auto iter = data_storage_.get_subsequence(indices);
      DistanceType best_distance = -1;
      ++iter;

      while (iter.is_valid()) {
        auto point = iter.get_point();
        int index = iter.get_key();
        bool is_good = true;
        std::vector<int> current_point_metadata = metadata_storage_[index];
        for (std::set<int>::iterator it=q_filter.begin(); it!=q_filter.end(); ++it) {
          auto search = std::find(current_point_metadata.begin(), current_point_metadata.end(), *it);
          bool found = search != current_point_metadata.end();
          is_good = is_good && found;
        }
        if(is_good) {
          DistanceType cur_distance = dst_(q_comp, point);
          if (cur_distance < best_distance || no_distance_found) {
            best_distance = cur_distance;
            no_distance_found = false;
            best_key = index;
          }
        }
        ++iter;
      }

    }
    int iteration = 0;
    while (best_key == -1 && iteration < max_iterations) {
      // printf("Start iteration %d\n", iteration);
      table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                          &candidates_, iteration);
      iteration += 1;
      // TODO: use nullptr for pointer types
      // printf("Fundet candidates %ld\n", candidates_.size());
      if (candidates_.size() > 0) {
        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);

        DistanceType best_distance = -1;
        ++iter;


        // printf("%d %f\n", candidates_[0], best_distance);
        // pretty print int q_filter
        // for (std::set<int>::iterator it=q_filter.begin(); it!=q_filter.end(); ++it) {
        //   printf("%d ", *it);
        // }
        // printf("\n");

        while (iter.is_valid()) {
          auto point = iter.get_point();
          int index = iter.get_key();
          bool is_good = true;
          std::vector<int> current_point_metadata = metadata_storage_[index];
          // printf("Found point: %d\n", index);
          // for (std::set<int>::iterator it=current_point_metadata.begin(); it!=current_point_metadata.end(); ++it) {
          //   printf("%d ", *it);
          // }
          // printf("\n");
          for (std::set<int>::iterator it=q_filter.begin(); it!=q_filter.end(); ++it) {
            auto search = std::find(current_point_metadata.begin(), current_point_metadata.end(), *it);
            bool found = search != current_point_metadata.end();
            is_good = is_good && found;
            // is_good = true;
          }
          if(is_good) {
            DistanceType cur_distance = dst_(q_comp, point);
            // printf("%d %f\n", iter.get_key(), cur_distance);
            if (cur_distance < best_distance || no_distance_found) {

              best_distance = cur_distance;
              no_distance_found = false;
              best_key = index;
              // printf("%d  is new best\n", best_key);
            }
          }
          ++iter;
        }
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();

    return best_key;
  }

  void find_k_nearest_neighbors(const LSHTablePointType& q,
                                const ComparisonPointType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);

    heap_.reset();
    heap_.resize(k);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    int_fast64_t initially_inserted = 0;
    for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
        ++iter;
      } else {
        break;
      }
    }

    if (initially_inserted >= k) {
      heap_.heapify();
      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        if (cur_distance < -heap_.min_key()) {
          heap_.replace_top(-cur_distance, iter.get_key());
        }
        ++iter;
      }
    }

    res.resize(initially_inserted);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + initially_inserted);
    for (int_fast64_t ii = 0; ii < initially_inserted; ++ii) {
      res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void find_near_neighbors(const LSHTablePointType& q,
                           const ComparisonPointType& q_comp,
                           DistanceType threshold, int_fast64_t num_probes,
                           int_fast64_t max_num_candidates,
                           std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);
    while (iter.is_valid()) {
      DistanceType cur_distance = dst_(q_comp, iter.get_point());
      if (cur_distance < threshold) {
        res.push_back(iter.get_key());
      }
      ++iter;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_candidates_with_duplicates(const LSHTablePointType& q,
                                      int_fast64_t num_probes,
                                      int_fast64_t max_num_candidates,
                                      std::vector<LSHTableKeyType>* result) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_candidates_with_duplicates(q, num_probes,
                                                 max_num_candidates, result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_unique_candidates(const LSHTablePointType& q,
                             int_fast64_t num_probes,
                             int_fast64_t max_num_candidates,
                             std::vector<LSHTableKeyType>* result,
                             int_fast8_t iterations = 0) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        result, iterations);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void reset_query_statistics() {
    table_query_->reset_query_statistics();
    stats_.reset();
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res = table_query_->get_query_statistics();
    res.average_total_query_time = stats_.average_total_query_time;
    res.average_distance_time = stats_.average_distance_time;

    if (res.num_queries > 0) {
      res.average_total_query_time /= res.num_queries;
      res.average_distance_time /= res.num_queries;
    }
    return res;
  }

 private:
  LSHTableQuery* table_query_;
  const DataStorage& data_storage_;
  std::unordered_map<int,std::vector<int>> metadata_storage_;
  MetadataStore small_labels_store_;
  std::vector<LSHTableKeyType> candidates_;
  DistanceFunction dst_;
  SimpleHeap<DistanceType, LSHTableKeyType> heap_;

  QueryStatistics stats_;
};

}  // namespace core
}  // namespace falconn

#endif
