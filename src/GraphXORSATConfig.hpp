#pragma once

#include "Frame.h"
#include <BinaryPolynomial.h>
#include <Samplers.h>

class GraphXORSATConfig : public dataframe::Config {
  private:
    uint32_t num_variables;
    uint32_t k;
    uint32_t num_rows;

    InterfaceSampler sampler;

    std::minstd_rand rng;

    std::vector<uint32_t> draw_random_variables() {
      std::vector<uint32_t> result(num_variables);
      std::iota(result.begin(), result.end(), 0);
      std::shuffle(result.begin(), result.end(), rng);
      result.resize(k);
      return result;
    }

	public:
    GraphXORSATConfig(dataframe::Params &params) : dataframe::Config(params), sampler(params) {
      num_variables = dataframe::utils::get<int>(params, "system_size");
      k = dataframe::utils::get<int>(params, "k");
      if (k != 2) {
        throw std::invalid_argument("Only k = 2 implemented for this simulator.");
      }
      num_rows = dataframe::utils::get<int>(params, "num_rows");

      if (k > num_variables) {
        throw std::invalid_argument("k must be smaller than the total number of variables.");
      }

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    GraphXORSATConfig()=default;

    ~GraphXORSATConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      Graph g(num_variables);
      for (size_t i = 0; i < num_rows; i++) {
        std::vector<uint32_t> vars = draw_random_variables();
        g.add_edge(vars[0], vars[1]);
      }

      QuantumGraphState state(g);

      auto interface = state.get_entropy_surface<int>();
      
      dataframe::data_t samples;
      sampler.add_samples(samples, interface);

      dataframe::DataSlide slide;
      for (auto const &[key, val] : samples) {
        slide.add_data(key);
        for (auto const &v : val) {
          slide.push_samples_to_data(key, v);
        }
      }

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<GraphXORSATConfig>(params);
    }
};
