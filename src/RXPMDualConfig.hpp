#pragma once

#include "Frame.h"
#include <CliffordState.h>
#include <Samplers.h>
#include <cstdint>

class RXPMDualConfig : public dataframe::Config {
  private:
    uint32_t system_size;
    double p;

    std::minstd_rand rng;

    size_t mod(int i) const {
      int L = static_cast<int>(system_size);
      return (i % L + L) % L;
    }

    double randf() {
      return double(rng())/RAND_MAX;
    }

	public:
    RXPMDualConfig(dataframe::Params &params) : dataframe::Config(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      p = dataframe::utils::get<double>(params, "p");

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    RXPMDualConfig()=default;

    ~RXPMDualConfig()=default;

    QuantumCHPState prepare_state() {
      size_t L = system_size;
      size_t N = L*L;

      QuantumCHPState state(N);
      for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
          size_t r = x + L*y;
          size_t row = r + N;
          state.set_z(row, r, 0);

          state.set_x(row, r, 1);

          if (x % 2 == y % 2) {
            if (randf() < p) {
              size_t yi = mod(y + 1);
              size_t pi = x + L*yi;
              state.set_z(row, pi, 1);

              yi = mod(y - 1);
              pi = x + L*yi;
              state.set_z(row, pi, 1);
            }
          } else {
            if (randf() < p) {
              size_t xi = mod(x + 1);
              size_t pi = xi + L*y;
              state.set_z(row, pi, 1);

              xi = mod(x - 1);
              pi = xi + L*y;
              state.set_z(row, pi, 1);
            }
          }
        }
      }

      return state;
    }

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      std::vector<uint32_t> qubits(system_size*system_size);
      std::iota(qubits.begin(), qubits.end(), 0);

      dataframe::DataSlide slide;

      slide.add_data("rank");

      QuantumCHPState state = prepare_state();
      uint32_t rank = state.partial_rank(qubits);
      slide.push_samples_to_data("rank", rank);

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<RXPMDualConfig>(params);
    }
};
