#pragma once

#include "Frame.h"
#include <BinaryPolynomial.h>
#include <Graph.hpp>


class FourBodyGraphGenerators {
  private:
    size_t mod(int i) const {
      return (i % L + L) % L;
    }

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i % L;
      int y = i / L;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x) + mod(y) * L;
    }

  public:
    int L;
    
    Graph<int, int> g;
    Graph<int, int> dual_g;

    std::set<size_t> break_impurities;

    FourBodyGraphGenerators()=default;

    FourBodyGraphGenerators(size_t L) : L(L) {
      g = Graph(2*L*L);
      dual_g = Graph(L*L);

      for (size_t x = 0; x < L; x++) {
        for (size_t y = 0; y < L; y++) {
          // Horizontal in first subsystem
          size_t i1 = to_index(x, y);
          size_t i2 = to_index(x+1, y);
          g.add_edge(i1, i2);

          // Vertical in second subsystem
          size_t j1 = to_index(x, y)   + L*L;
          size_t j2 = to_index(x, y+1) + L*L;
          g.add_edge(j1, j2);
        }
      }
    }

    void add_field_impurity(int x, int y) {
      size_t i = to_index(x, y);
      g.add_edge(i, i + L*L);
    }

    void add_break_impurity_horizontal(int x, int y) {
//std::cout << fmt::format("Calling add_break_impurity_horizontal at {}, {}\n", x, y);
      size_t i1 = to_index(x, y);
      size_t i2 = to_index(x+1, y);
      g.remove_edge(i1, i2);

      size_t i3 = to_index(x, y-1);
      std::cout << fmt::format("Removing edge between ({}, {})\n", i1, i2);
      std::cout << fmt::format("Adding edge ({}, {}) to dual graph\n", i1, i3);
      dual_g.add_edge(i1, i3);
    }

    void add_break_impurity_vertical(int x, int y) {
//std::cout << fmt::format("Calling add_break_impurity_vertical at {}, {}\n", x, y);
      size_t i1 = to_index(x, y)   + L*L;
      size_t i2 = to_index(x, y+1) + L*L;
      g.remove_edge(i1, i2);

      i1 = i1 - L*L;
      size_t i3 = to_index(x-1, y);
      std::cout << fmt::format("Removing edge between ({}, {})\n", i1, i2 - L*L);
      std::cout << fmt::format("Adding edge ({}, {}) to dual graph\n", i1, i3);
      dual_g.add_edge(i1, i3);
    }

    void add_break_impurity(int x, int y) {
      size_t i = to_index(x, y);

      break_impurities.insert(i);
//std::cout << fmt::format("Calling add_break_impurity. break_impurities = {}\n", break_impurities);

      size_t i1 = to_index(x+1, y);
      if (break_impurities.contains(i1)) {
        add_break_impurity_vertical(x+1, y);
      }

      size_t i2 = to_index(x-1, y);
      if (break_impurities.contains(i2)) {
        add_break_impurity_vertical(x, y);
      }

      size_t i3 = to_index(x, y+1);
      if (break_impurities.contains(i3)) {
        add_break_impurity_horizontal(x, y+1);
      }

      size_t i4 = to_index(x, y-1);
      if (break_impurities.contains(i4)) {
        add_break_impurity_horizontal(x, y);
      }
    }

    std::vector<std::set<uint32_t>> get_components() const {
      return g.component_partition();
    }

    int get_rank() const {
      auto components = get_components();
      return components.size() - dual_g.num_loops() - 1;
    }
};

class GraphXORSATConfig : public dataframe::Config {
  private:
    size_t L;
    double p;

    FourBodyGraphGenerators fbg;

    std::minstd_rand rng;

	public:
    GraphXORSATConfig(dataframe::Params &params) : dataframe::Config(params) {
      L = dataframe::utils::get<int>(params, "L");
      p = dataframe::utils::get<double>(params, "p");

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    ~GraphXORSATConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      fbg = FourBodyGraphGenerators(L);

      dataframe::DataSlide slide;
      slide.add_data("rank");
      slide.push_samples_to_data("rank", (double) fbg.get_rank());

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<GraphXORSATConfig>(params);
    }
};
