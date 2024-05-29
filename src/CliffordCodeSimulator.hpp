#pragma once

#include "CliffordState.h"
#include "BinaryPolynomial.h"

inline static void rc_timestep(std::shared_ptr<CliffordState> state, uint32_t gate_width, bool offset_layer, bool periodic_bc = true) {
	uint32_t system_size = state->system_size();
	uint32_t num_gates = system_size / gate_width;

	std::vector<uint32_t> qubits(gate_width);
	std::iota(qubits.begin(), qubits.end(), 0);

	for (uint32_t j = 0; j < num_gates; j++) {
		uint32_t offset = offset_layer ? gate_width*j : gate_width*j + gate_width/2;

		bool periodic = false;
		std::vector<uint32_t> offset_qubits(qubits);
		std::transform(offset_qubits.begin(), offset_qubits.end(), offset_qubits.begin(), 
						[system_size, offset, &periodic](uint32_t x) { 
							uint32_t q = x + offset;
							if (q % system_size != q) {
								periodic = true;
							}
							return q % system_size; 
						});
		
		if (!(!periodic_bc && periodic)) {
			state->random_clifford(offset_qubits);
		}
	}
}

class CliffordCodeSimulator: public dataframe::Simulator {
	private:
		std::shared_ptr<QuantumCHPState> state;
		int seed;

		uint32_t system_size;
		double mzr_prob;
		
		bool offset;

		bool start_sampling;

	public:
    CliffordCodeSimulator(dataframe::Params &params, uint32_t num_threads) : dataframe::Simulator(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      mzr_prob = dataframe::utils::get<double>(params, "mzr_prob");

      seed = dataframe::utils::get<int>(params, "seed", -1);

      start_sampling = false;
      offset = false;
      state = std::make_shared<QuantumCHPState>(system_size, seed);
    }

		virtual void equilibration_timesteps(uint32_t num_steps) override {
			start_sampling = false;
			timesteps(num_steps);
			start_sampling = true;
		}

    virtual void timesteps(uint32_t num_steps) override {
      for (uint32_t i = 0; i < num_steps; i++) {
        rc_timestep(state, 2, offset, true);

        // Apply measurements
        for (uint32_t j = 0; j < system_size; j++) {
          if (state->randf() < mzr_prob) {
            state->mzr(j);
          }
        }

        offset = !offset;
      }
    }

    void add_rank_samples(dataframe::data_t& samples, ParityCheckMatrix& matrix) {
      auto r = matrix.rank();
      dataframe::utils::emplace(samples, "rank", r);
    }

    void add_locality_samples(dataframe::data_t& samples, GeneratorMatrix& matrix) {
      std::vector<double> s;
      std::vector<size_t> sites;
      size_t num_samples = matrix.num_cols;
      size_t n = 0;
      for (size_t i = 0; i < num_samples; i++) {
        // Sample locality
        s.push_back(static_cast<double>(matrix.generator_locality(sites)));

        // Add new sites
        sites.push_back(n);
        n++;
      }

      dataframe::utils::emplace(samples, "locality", s);
    }

    void add_core_size_samples(dataframe::data_t& samples, ParityCheckMatrix& matrix) {
      ParityCheckMatrix H(matrix);

      auto [r, s] = H.leaf_removal_iteration(rng);
      size_t n = 0;
      while (r.has_value()) {
        std::tie(r, s) = H.leaf_removal_iteration(rng);
        n++;
      }

      size_t core_size = H.num_rows;
      dataframe::utils::emplace(samples, "core_size", core_size);
      dataframe::utils::emplace(samples, "num_iters", num_iters);
    }

		virtual dataframe::data_t take_samples() override {
      dataframe::data_t samples;

      size_t n = state->tableau.num_rows()/2;
      size_t m = state->tableau.num_qubits;
      ParityCheckMatrix H(n, m);
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
          H.set(i, j, state->tableau.x(i + n, j) || state->tableau.z(i + n, j));
        }
      }

      GeneratorMatrix G = H.to_generator_matrix();

      add_rank_samples(samples, H);
      add_locality_samples(samples, G);
      add_core_size_samples(samples, H);

      return samples;
    }
};
