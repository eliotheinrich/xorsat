#pragma once

#include "CliffordState.h"
#include "BinaryPolynomial.h"

#define GCS_ERDOS 0
#define GCS_REGULAR 1

class GraphClusteringSimulator: public dataframe::Simulator {
	private:
		std::shared_ptr<QuantumGraphState> state;
		int seed;

		uint32_t system_size;

    double pz;

    uint32_t graph_type;
		double pb;
    size_t k;

    void prepare_erdos_graph() {
      Graph<int, int> g = Graph<int, int>::erdos_renyi_graph(system_size, pb, &rng);
      state = std::make_shared<QuantumGraphState>(g, seed);
    }

    void prepare_regular_graph() {
      Graph<int, int> g = Graph<int, int>::random_regular_graph(system_size, k, &rng);
      state = std::make_shared<QuantumGraphState>(g, seed);
    }

	public:
    GraphClusteringSimulator(dataframe::Params &params, uint32_t num_threads) : dataframe::Simulator(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      if (system_size <= 2) {
        throw std::invalid_argument("Must have at least three qubits for GraphClusteringSimulator.");
      }

      pz = dataframe::utils::get<double>(params, "pz");

      graph_type = dataframe::utils::get<int>(params, "graph_type", GCS_ERDOS);
      if (graph_type == GCS_ERDOS) {
        pb = dataframe::utils::get<double>(params, "pb");
        prepare_erdos_graph();
      } else if (graph_type == GCS_REGULAR) {
        k = dataframe::utils::get<int>(params, "k");
        prepare_regular_graph();
      }
    }

    virtual void timesteps(uint32_t num_steps) override {
      for (uint32_t i = 0; i < num_steps; i++) {
        for (size_t j = 2; j < system_size; j++) {
          if (randf() < pz) {
            state->mzr(j);
          } else {
            state->myr(j);
          }
        }
      }
    }

		virtual dataframe::data_t take_samples() override {
      dataframe::data_t samples;

      auto s1 = state->entropy({0}, 2);
      auto s2 = state->entropy({1}, 2);
      auto s = state->entropy({0, 1}, 2);

      samples.emplace("entropy", s);
      samples.emplace("mutual_information", s1 + s2 - s);

      return samples;
    }
};
