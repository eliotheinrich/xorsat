#pragma once

#include "Frame.h"
#include <Graph.hpp>
#include <BinaryPolynomial.h>
#include <Samplers.h>

class SlantedCheckerboardConfig : public dataframe::Config {
  private:
    uint32_t system_size;

    double p4;
    bool single_site;
    bool delete_plaquettes;

    LinearCodeSampler sampler;

    std::minstd_rand rng;

    uint32_t rand() {
      return rng();
    }

    double randf() {
      return double(rng())/double(RAND_MAX);
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

    void add_row(std::shared_ptr<ParityCheckMatrix>& A, const std::vector<size_t>& inds) const {
      std::vector<bool> row(system_size*system_size, false);
      for (auto const j : inds) {
        row[j] = true;
      }

      A->append_row(row);
    }


    std::shared_ptr<ParityCheckMatrix> generate_interaction_matrix() {
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, system_size*system_size);

      for (int x = 0; x < system_size; x++) {
        for (int y = 0; y < system_size; y++) {
          bool include_plaquette = delete_plaquettes ? (randf() < p4) : true;
          if (include_plaquette) {
            std::vector<size_t> to_include = {to_index(x, y), to_index(x+1, y), to_index(x, y+1), to_index(x+1, y+1)};
            add_row(A, to_include);
          }

          bool include_onsite = single_site ? (randf() < p4) : false;
          if (include_onsite) {
            std::vector<size_t> to_include = {to_index(x, y)};
            add_row(A, to_include);
          }
        }
      }

      return A;
    }

	public:
    SlantedCheckerboardConfig(dataframe::Params &params) : dataframe::Config(params), sampler(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      p4 = dataframe::utils::get<double>(params, "p4", 0.0);

      single_site = dataframe::utils::get<int>(params, "single_site", true);
      delete_plaquettes = dataframe::utils::get<int>(params, "delete_plaquettes", false);

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    SlantedCheckerboardConfig()=default;

    ~SlantedCheckerboardConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      auto start = std::chrono::high_resolution_clock::now();
      std::shared_ptr<ParityCheckMatrix> A = generate_interaction_matrix();

      std::vector<size_t> sites1(system_size);
      for (int x = 0; x < system_size; x++) {
        sites1[x] = to_index(x, 0);
      }

      std::vector<size_t> sites2(system_size);
      for (int y = 0; y < system_size; y++) {
        sites2[y] = to_index(0, y);
      }

      dataframe::DataSlide slide;
      sampler.add_samples(slide, A, rng, sites1, sites2);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::micro> duration = end - start;

      slide.add_data("time");
      slide.push_samples_to_data("time", duration.count()/1e6);

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<SlantedCheckerboardConfig>(params);
    }
};
