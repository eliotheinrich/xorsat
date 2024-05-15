#pragma once

#include "Frame.h"
#include <Graph.hpp>
#include <BinaryPolynomial.h>
#include <Samplers.h>

#define RPMCA_3BODY 0
#define RPMCA_5BODY 1

class RPMCAConfig : public dataframe::Config {
  private:
    uint32_t system_size;

    double p;

    int model_type;

    bool sample_generators;

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
      int x = i % system_size;
      int y = i / system_size;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x) + mod(y) * system_size;
    }

    struct Generator {
      std::vector<std::vector<bool>> vals;
      Generator(size_t L) {
        vals = std::vector<std::vector<bool>>(L, std::vector<bool>(L, false));
      }
      Generator()=default;

      std::vector<bool>& operator[](size_t i) {
        return vals[i];
      }

      std::vector<bool> operator[](size_t i) const {
        return vals[i];
      }

      Generator operator+(const Generator& other) const {
        Generator g(vals.size());
        for (size_t i = 0; i < vals.size(); i++) {
          for (size_t j = 0; j < vals.size(); j++) {
            g[i][j] = vals[i][j] != other[i][j];
          }
        }

        return g;
      }

      std::string to_string() const {
        std::string s = "";
        for (size_t i = 0; i < vals.size(); i++) {
          for (size_t j = 0; j < vals.size(); j++) {
            s += std::to_string(int(vals[i][j])) + " ";
          }
        }

        return "[ " + s + "]";
      }
    };

    std::vector<Generator> generators_3body() {
      size_t L = system_size; 
      std::vector<Generator> generators(L);
      for (size_t k = 0; k < L; k++) {
        Generator g = Generator(L);
        g[0][k] = 1;
        for (size_t t = 1; t < L; t++) {
          for (size_t j = 0; j < L; j++) {
            if (g[t-1][j] == g[t-1][mod(j-1)]) {
              g[t][j] = 0;
            } else {
              g[t][j] = (randf() < p);
            }
          }
        }

        generators[k] = g;
      }

      return generators;
    }
    
    void apply_constraint(std::vector<Generator>& generators, int t, int j) {
      size_t L = system_size;
      size_t idx = -1;

      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][t-1][j]) {
          idx = k;
          break;
        }
      }

      // Constraint is already not broken anywhere; can continue
      if (idx == -1) {
        return;
      }

      // Copy chosen generator to do row-reduction with
      Generator g = generators[idx];

      // Set new generator
      Generator x = Generator(L);
      x[t][j] = 1;
      generators[idx] = x;

      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][t-1][j] && k != idx) {
          generators[k] = generators[k] + g;
        }
      }

      // Sanity check that the constraint is now enforced
      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][t-1][j] && k != idx) {
          throw std::runtime_error("apply_constraint failed.");
        }
      }
    }

    std::vector<Generator> generators_5body() {
      size_t L = system_size; 
      std::vector<bool> impurities(L*L);
      for (size_t i = 0; i < L*L; i++) {
        impurities[i] = !(randf() < p);
      }

      std::vector<Generator> generators(L);
      for (size_t k = 0; k < L; k++) {
        Generator g = Generator(L);
        g[0][k] = 1;
        g[1][k] = 1;
        generators[k] = g;
      }


      for (int t = 2; t < L; t++) {
        for (int j = 0; j < L; j++) {
          for (size_t k = 0; k < generators.size(); k++) {
            generators[k][t][j] = (generators[k][t-1][j] + generators[k][t-1][mod(j-1)] + generators[k][t-1][mod(j+1)] + generators[k][t-2][j]) % 2;
          }

          size_t i = to_index(t, j);
          if (impurities[i]) {
            apply_constraint(generators, t, j);
          }
        }
      }

      // Hacky for now
      return std::vector<Generator>(generators.begin(), generators.begin() + L);
    }


    std::shared_ptr<GeneratorMatrix> to_generator_matrix(const std::vector<Generator>& gens) const {
      size_t L = system_size;
      size_t N = L*L;
      std::shared_ptr<GeneratorMatrix> G = std::make_shared<GeneratorMatrix>(L, N);

      for (size_t k = 0; k < gens.size(); k++) {
        for (size_t x = 0; x < L; x++) {
          for (size_t y = 0; y < L; y++) {
            size_t idx = to_index(x, y);
            G->set(k, idx, gens[k][x][y]);
          }
        }
      }

      return G;
    }

	public:
    RPMCAConfig(dataframe::Params &params) : dataframe::Config(params), sampler(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      p = dataframe::utils::get<double>(params, "p", 0.0);

      model_type = dataframe::utils::get<int>(params, "model_type", RPMCA_3BODY);

      sample_generators = dataframe::utils::get<int>(params, "sample_generators", false);

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

    RPMCAConfig()=default;

    ~RPMCAConfig()=default;

    virtual dataframe::DataSlide compute(uint32_t num_threads) {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<Generator> generators;
      if (model_type == RPMCA_3BODY) {
        generators = generators_3body();
      } else if (model_type == RPMCA_5BODY) {
        generators = generators_5body();
      }

      dataframe::DataSlide slide;

      if (sample_generators) {
        std::vector<std::vector<double>> vals(generators.size());
        for (size_t k = 0; k < generators.size(); k++) {
          vals[k] = std::vector<double>(system_size*system_size);
          for (size_t x = 0; x < system_size; x++) {
            for (size_t y = 0; y < system_size; y++) {
              size_t i = to_index(x, y);
              vals[k][i] = generators[k][x][y];
            }
          }
        }

        slide.add_samples("generators", vals.size());
        slide.push_samples("generators", vals);
      }


      auto G = to_generator_matrix(generators);

      sampler.add_samples(slide, G, rng);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::micro> duration = end - start;

      slide.add_data("time");
      slide.push_samples_to_data("time", duration.count()/1e6);

      return slide;
    }

    virtual std::shared_ptr<dataframe::Config> clone() {
      return std::make_shared<RPMCAConfig>(params);
    }
};
