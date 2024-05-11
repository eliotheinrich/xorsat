#pragma once

#include "Frame.h"
#include <BinaryPolynomial.h>
#include <Samplers.h>

class XORSATConfig : public dataframe::Config {
  private:
    uint32_t num_variables;
    uint32_t k;
    uint32_t num_rows;

    LinearCodeSampler sampler;

    std::minstd_rand rng;

    std::vector<uint32_t> draw_random_variables() {
      std::vector<uint32_t> result(num_variables);
      std::iota(result.begin(), result.end(), 0);
      std::shuffle(result.begin(), result.end(), rng);
      result.resize(k);
      return result;
    }

	public:
    XORSATConfig(dataframe::Params &params) : dataframe::Config(params), sampler(params) {
      num_variables = dataframe::utils::get<int>(params, "num_variables");
      k = dataframe::utils::get<int>(params, "k");
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

    XORSATConfig()=default;

    ~XORSATConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(num_rows, num_variables);
      for (size_t i = 0; i < num_rows; i++) {
        std::vector<uint32_t> vars = draw_random_variables();
        for (auto const v : vars) {
          A->set(i, v, 1);
        }
      }

      dataframe::DataSlide slide;
      sampler.add_samples(slide, A, rng);

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<XORSATConfig>(params);
    }
};
