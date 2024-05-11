#pragma once

#include "Frame.h"
#include <Graph.hpp>
#include <BinaryPolynomial.h>
#include <Samplers.h>

#define LDPC_ERDOS 0
#define LDPC_REGULAR 1
#define LDPC_LATTICE_5 2
#define LDPC_LATTICE_3 3
#define LDPC_LATTICE_4 4

class LDPCConfig : public dataframe::Config {
  private:
    uint32_t system_size;

    double pr;

    int model_type;
    double pb;
    size_t k;
    bool obc;

    bool single_site;

    LinearCodeSampler sampler;

    std::minstd_rand rng;

    uint32_t rand() {
      return rng();
    }

    double randf() {
      return double(rng())/double(RAND_MAX);
    }

    std::shared_ptr<ParityCheckMatrix> from_graph(const Graph<int>& g) {
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(system_size, system_size);

      for (size_t a = 0; a < g.num_vertices; a++) {
        if (single_site) {
          A->set(a, a, 1);
          if (randf() < pr) {
            for (auto const& [i, _] : g.edges[a]) {
              A->set(a, i, 1);
            }
          }
        } else {
          if (randf() < pr) {
            A->set(a, a, 1);
            for (auto const& [i, _] : g.edges[a]) {
              A->set(a, i, 1);
            }
          }
        }
      }
      
      return A;
    }

    std::shared_ptr<ParityCheckMatrix> generate_erdos_interaction_matrix() {
      Graph<int> g = Graph<int>::erdos_renyi_graph(system_size, pb, &rng);
      return from_graph(g);
    }

    std::shared_ptr<ParityCheckMatrix> generate_regular_interaction_matrix() {
      Graph<int> g = Graph<int>::random_regular_graph(system_size, k, &rng);
      return from_graph(g);
    }

    size_t mod(int i) const {
      int L = static_cast<int>(system_size);
      return (i % L + L) % L;
    }

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i / system_size;
      int y = i % system_size;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x) * system_size + mod(y);
    }

    std::shared_ptr<ParityCheckMatrix> generate_5body_lattice_interaction_matrix() {
      size_t N = system_size*system_size;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < system_size*system_size; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (x == 0 || x == system_size - 1)) {
          continue;
        }
        size_t i1 = to_index(x+1, y);
        size_t i2 = to_index(x-1, y);
        size_t i3 = to_index(x, y+1);
        size_t i4 = to_index(x, y-1);

        std::vector<size_t> inds{i1, i2, i3, i4};
        std::vector<size_t> to_include;
        if (single_site) {
          // Impurity turns 5-body term into single-site term
          to_include.push_back(i);
          if (randf() < pr) {
            for (auto j : inds) {
              to_include.push_back(j);
            }
          }
        } else {
          // Impurity removes single-site as well
          if (randf() < pr) {
            to_include.push_back(i);
            for (auto j : inds) {
              to_include.push_back(j);
            }
          }
        }

        std::vector<bool> row(N, false);
        for (auto j : to_include) {
          row[j] = true;
        }

        A->append_row(row);
      }

      return A;
    }

    std::shared_ptr<ParityCheckMatrix> generate_4body_lattice_interaction_matrix() {
      size_t N = system_size*system_size;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (x == 0 || x == system_size - 1)) {
          continue;
        }

        size_t i1 = to_index(x+1, y);
        size_t i2 = to_index(x-1, y);
        size_t i3 = to_index(x, y+1);
        size_t i4 = to_index(x, y-1);

        std::vector<size_t> to_include;
        if (randf() < pr) {
          to_include = {i};
        } else {
          to_include = {i1, i2, i3, i4};
        }

        std::vector<bool> row(N, false);
        for (auto j : to_include) {
          row[j] = true;
        }

        A->append_row(row);
      }

      return A;
    }

    std::shared_ptr<ParityCheckMatrix> generate_3body_lattice_interaction_matrix() {
      size_t N = system_size*system_size;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (x == 0 || x == system_size - 1)) {
          continue;
        }

        size_t i1, i2;
        if (x % 2 == y % 2) {
          // red site
          i1 = to_index(x, y+1);
          i2 = to_index(x, y-1);
        } else {
          // black site
          i1 = to_index(x+1, y);
          i2 = to_index(x-1, y);
        }

        std::vector<size_t> inds{i1, i2};
        std::vector<size_t> to_include;
        if (single_site) {
          // Impurity turns 3-body term into single-site term
          to_include.push_back(i);
          if (randf() < pr) {
            for (auto j : inds) {
              to_include.push_back(j);
            }
          }
        } else {
          // Impurity removes single-site as well
          if (randf() < pr) {
            to_include.push_back(i);
            for (auto j : inds) {
              to_include.push_back(j);
            }
          }
        }

        std::vector<bool> row(N, false);
        for (auto j : to_include) {
          row[j] = true;
        }

        A->append_row(row);
      }

      return A;
    }

	public:
    LDPCConfig(dataframe::Params &params) : dataframe::Config(params), sampler(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      pr = dataframe::utils::get<double>(params, "pr", 0.0);

      model_type = dataframe::utils::get<int>(params, "model_type", LDPC_ERDOS);
      if (model_type == LDPC_ERDOS) {
        pb = dataframe::utils::get<double>(params, "pb", 0.0);
      } else if (model_type == LDPC_REGULAR) {
        k = dataframe::utils::get<int>(params, "k", 0);
      } else if (model_type == LDPC_LATTICE_5 || model_type == LDPC_LATTICE_3 || model_type == LDPC_LATTICE_4) {
        obc = dataframe::utils::get<int>(params, "obc", true);
      }

      single_site = dataframe::utils::get<int>(params, "single_site", true);

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    LDPCConfig()=default;

    ~LDPCConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      auto start = std::chrono::high_resolution_clock::now();
      std::shared_ptr<ParityCheckMatrix> A;
      if (model_type == LDPC_ERDOS) {
        A = generate_erdos_interaction_matrix();
      } else if (model_type == LDPC_REGULAR) {
        A = generate_regular_interaction_matrix();
      } else if (model_type == LDPC_LATTICE_5) {
        A = generate_5body_lattice_interaction_matrix();
      } else if (model_type == LDPC_LATTICE_3) {
        A = generate_3body_lattice_interaction_matrix();
      } else if (model_type == LDPC_LATTICE_4) {
        A = generate_4body_lattice_interaction_matrix();
      }

      dataframe::DataSlide slide;
      sampler.add_samples(slide, A, rng);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::micro> duration = end - start;

      slide.add_data("time");
      slide.push_samples_to_data("time", duration.count()/1e6);

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<LDPCConfig>(params);
    }
};
