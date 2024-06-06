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
    uint32_t num_iterations;

    double p;

    int model_type;

    bool sample_generators;
    bool sample_boundary_sym;

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

    std::vector<double> boundary_symmetry_entropy(const std::shared_ptr<GeneratorMatrix>& generators, const std::vector<size_t>& sites) const {
      GeneratorMatrix boundary_generators = generators->supported(sites);
      size_t num_sites = sites.size();
      std::vector<double> S(num_sites);
      for (size_t i = 0; i < num_sites; i++) {
        std::vector<size_t> A(sites.begin(), sites.begin() + i);
        std::vector<size_t> Ab(sites.begin() + i, sites.end());

        S[i] = boundary_generators.sym(A, Ab);
      }

      return S;
    }

    std::vector<size_t> vector_complement(const std::vector<size_t>& sites, const std::vector<size_t>& all_sites) const {
      std::vector<bool> mask(all_sites.size(), false);
      for (size_t i = 0; i < sites.size(); i++) {
        mask[i] = true;
      }

      std::vector<size_t> complement;
      for (size_t i = 0; i < all_sites.size(); i++) {
        if (!mask[i]) {
          complement.push_back(i);
        }
      }

      return complement;
    }

    double sym(GeneratorMatrix& G, const std::vector<size_t>& A) const {
      std::vector<size_t> all_sites(G.num_rows);
      std::iota(all_sites.begin(), all_sites.end(), 0);
      std::vector<size_t> Abar = vector_complement(A, all_sites);

      return G.partial_rank(A) + G.partial_rank(Abar) - G.rank();
    }

    std::vector<double> boundary_mutual_information(std::shared_ptr<GeneratorMatrix> generators, const std::vector<size_t>& sites) const {
      GeneratorMatrix boundary_generators = generators->supported(sites);
      size_t num_sites = sites.size();
      
      size_t LA = system_size/8;

      std::vector<double> samples;
      for (size_t i = 0; i < system_size/2; i++) {
        std::vector<size_t> A(LA);
        std::vector<size_t> B(LA);
        for (size_t j = 0; j < LA; j++) {
          A[j] = to_index(j + i, num_iterations - 1);
          B[j] = to_index(j + i + system_size/2, num_iterations - 1);
        }

        std::vector<size_t> AB(A.begin(), A.end());
        AB.insert(AB.end(), B.begin(), B.end());

        double d = sym(boundary_generators, A) + sym(boundary_generators, B) + sym(boundary_generators, AB);
        samples.push_back(d);
      }

      return samples;
    }

    std::vector<double> bulk_symmetry_entropy(const std::shared_ptr<GeneratorMatrix>& generators, size_t LA) const {
      size_t Lx = system_size;
      size_t Ly = num_iterations;

      std::vector<size_t> all_sites(Lx*Ly);
      std::iota(all_sites.begin(), all_sites.end(), 0);

      std::vector<double> samples;
      for (size_t k = 0; k < Ly; k++) {
        std::vector<size_t> sites(false);
        std::vector<bool> mask(Lx*Ly);
        for (size_t x = 0; x < Lx; x++) {
          for (size_t y = 0; y < LA; y++) {
            size_t i = to_index(x, y + k);
            sites.push_back(i);
            mask[i] = true;
          }
        }

        std::vector<size_t> sites_complement;
        for (size_t i = 0; i < Lx*Ly; i++) {
          if (!mask[i]) {
            sites_complement.push_back(i);
          }
        }

        //GeneratorMatrix G1 = generators->submatrix(sites);
        //GeneratorMatrix G2 = generators->submatrix(sites_complement);
        //GeneratorMatrix G3 = generators->submatrix(all_sites);

        //G1.transpose();
        //G2.transpose();
        //G3.transpose();

        //double d = G1.rank() + G2.rank() - G3.rank();

        double d = generators->partial_rank(sites) + generators->partial_rank(sites_complement) - generators->rank(); 
        samples.push_back(d);
      }

      return samples;
    }


    struct Generator {
      size_t Lx;
      size_t Ly;

      std::vector<std::vector<bool>> vals;
      Generator(size_t Lx, size_t Ly) : Lx(Lx), Ly(Ly) {
        vals = std::vector<std::vector<bool>>(Lx, std::vector<bool>(Ly, false));
      }
      Generator()=default;

      std::vector<bool>& operator[](size_t i) {
        return vals[i];
      }

      std::vector<bool> operator[](size_t i) const {
        return vals[i];
      }

      void append_row(const std::vector<bool>& row) {
        if (row.size() != Ly) {
          throw std::runtime_error(
            fmt::format(
              "row dimension mismatch: Ly = {}, row.size() = {}.",
              Ly, row.size()
            )
          );
        }

        vals.push_back(row);
        Lx++;
      }

      void append_col(const std::vector<bool>& col) {
        if (col.size() != Lx) {
          throw std::runtime_error(
            fmt::format(
              "col dimension mismatch: Lx = {}, col.size() = {}.",
              Lx, col.size()
            )
          );
        }

        for (size_t i = 0; i < Lx; i++) {
          vals[i].push_back(col[i]);
        }
        Ly++;
      }

      Generator operator+(const Generator& other) const {
        if (Lx != other.Lx || Ly != other.Ly) {
          throw std::runtime_error(
            fmt::format(
              "Generators have mismatched dimension: ({} x {}), ({}, {})",
              Lx, Ly, other.Lx, other.Ly
            )
          );
        }

        Generator g(Lx, Ly);
        for (size_t i = 0; i < Lx; i++) {
          for (size_t j = 0; j < Ly; j++) {
            g[i][j] = vals[i][j] != other[i][j];
          }
        }

        return g;
      }

      std::string to_string() const {
        std::string s = "";
        for (size_t i = 0; i < Lx; i++) {
          for (size_t j = 0; j < Ly; j++) {
            s += std::to_string(int(vals[i][j])) + " ";
          }
        }

        return "[ " + s + "]";
      }
    };

    std::vector<Generator> generators_3body() {
      size_t Lx = system_size; 
      size_t Ly = num_iterations;

      std::vector<Generator> generators(Lx);
      for (size_t k = 0; k < Lx; k++) {
        Generator g = Generator(Lx, 1);
        g[k][0] = 1;
        for (size_t t = 1; t < Ly; t++) {
          std::vector<bool> col(Lx);
          for (int j = 0; j < Lx; j++) {
            if (g[j][t-1] == g[mod(j-1)][t-1]) {
              col[j] = 0;
            } else {
              col[j] = (randf() < p);
            }
          }

          g.append_col(col);
        }

        generators[k] = g;
      }

      return generators;
    }
    
    void apply_constraint(std::vector<Generator>& generators, int t, int j) {
      size_t Lx = system_size;
      size_t Ly = num_iterations;

      size_t idx = -1;

      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][j][t-1]) {
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
      Generator x = Generator(Lx, generators[idx].Ly);
      x[j][t] = 1;
      generators[idx] = x;

      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][j][t-1] && k != idx) {
          generators[k] = generators[k] + g;
        }
      }

      // Sanity check that the constraint is now enforced
      for (size_t k = 0; k < generators.size(); k++) {
        if (generators[k][j][t-1] && k != idx) {
          throw std::runtime_error("apply_constraint failed.");
        }
      }
    }

    std::vector<Generator> generators_5body() {
      size_t Lx = system_size; 
      size_t Ly = num_iterations;

      std::vector<bool> impurities(Lx*Ly);
      for (size_t i = 0; i < Lx*Ly; i++) {
        impurities[i] = !(randf() < p);
      }

      std::vector<Generator> generators(Lx);
      for (size_t k = 0; k < Lx; k++) {
        Generator g = Generator(Lx, 2);
        g[k][0] = 1;
        g[k][1] = 1;
        generators[k] = g;
      }


      for (int t = 2; t < Ly; t++) {
        for (size_t k = 0; k < Lx; k++) {
          std::vector<bool> col(Lx);
          for (size_t j = 0; j < Lx; j++) {
            col[j] = (generators[k][j][t-1] + generators[k][mod(j-1)][t-1] + generators[k][mod(j+1)][t-1] + generators[k][j][t-2]) % 2;
          }
          generators[k].append_col(col);
        }

        for (size_t k = 0; k < Lx; k++) {
          for (size_t j = 0; j < Lx; j++) {
            size_t i = to_index(j, t);
            if (impurities[i]) {
              apply_constraint(generators, t, j);
            }
          }
        }
      }

      return generators;
    }


    std::shared_ptr<GeneratorMatrix> to_generator_matrix(const std::vector<Generator>& gens) const {
      size_t Lx = system_size;
      size_t Ly = num_iterations;
      size_t N = Lx*Ly;
      std::shared_ptr<GeneratorMatrix> G = std::make_shared<GeneratorMatrix>(gens.size(), N);

      for (size_t k = 0; k < gens.size(); k++) {
        for (size_t x = 0; x < Lx; x++) {
          for (size_t y = 0; y < Ly; y++) {
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
      num_iterations = dataframe::utils::get<int>(params, "num_iterations", system_size);
      p = dataframe::utils::get<double>(params, "p", 0.0);

      model_type = dataframe::utils::get<int>(params, "model_type", RPMCA_3BODY);

      sample_generators = dataframe::utils::get<int>(params, "sample_generators", false);
      sample_boundary_sym = dataframe::utils::get<int>(params, "sample_boundary_sym", false);

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

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
            for (size_t y = 0; y < num_iterations; y++) {
              size_t i = to_index(x, y);
              vals[k][i] = generators[k][x][y];
            }
          }
        }

        slide.add_samples("generators", vals.size());
        slide.push_samples("generators", vals);
      }


      auto G = to_generator_matrix(generators);

      if (sample_boundary_sym) {
        std::vector<size_t> boundary(system_size);

        for (size_t i = 0; i < system_size; i++) {
          boundary[i] = i;
        }

        auto sym = boundary_symmetry_entropy(G, boundary);
        slide.add_data("bottom_boundary_sym", sym.size());
        slide.push_samples_to_data("bottom_boundary_sym", sym);

        for (size_t i = 0; i < system_size; i++) {
          size_t idx = to_index(i, num_iterations-1);
          boundary[i] = idx;
        }

        sym = boundary_symmetry_entropy(G, boundary);
        slide.add_data("top_boundary_sym", sym.size());
        slide.push_samples_to_data("top_boundary_sym", sym);

        slide.add_data("boundary_mutual_information");
        auto mutual_information = boundary_mutual_information(G, boundary);
        slide.push_samples_to_data("boundary_mutual_information", std::vector<std::vector<double>>{mutual_information});

        std::vector<std::vector<double>> bulk_symmetry;
        for (size_t i = 0; i < system_size; i++) {
          std::vector<double> s = bulk_symmetry_entropy(G, i);
          bulk_symmetry.push_back(s);
        }

        slide.add_data("bulk_symmetry", bulk_symmetry.size());
        slide.push_samples_to_data("bulk_symmetry", bulk_symmetry);
      }

      slide.add_data("scf");
      slide.push_samples_to_data("scf", (double) G->rank());

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
