#pragma once

#include "Frame.h"
#include <Graph.hpp>
#include <BinaryPolynomial.h>
#include <Samplers.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#define LDPC_ERDOS 0
#define LDPC_REGULAR 1
#define LDPC_LATTICE_5 2
#define LDPC_LATTICE_3 3
#define LDPC_LATTICE_4 4
#define LDPC_TRIANGULAR_PLAQUETTE 5


class BoundarySymmetrySampler {
  private:
    size_t Lx;
    size_t Ly;

    bool sample_bulk_symmetry;
    size_t max_width;

    bool sample_bulk_mutual_information;

    size_t mod(int i, int L) const {
      return (i % L + L) % L;
    }

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i % Lx;
      int y = i / Lx;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x, Lx) + mod(y, Lx) * Lx;
    }

  public:
    BoundarySymmetrySampler()=default;
    BoundarySymmetrySampler(dataframe::Params& params, size_t Lx, size_t Ly) : Lx(Lx), Ly(Ly) {
      sample_bulk_symmetry = dataframe::utils::get<int>(params, "sample_bulk_symmetry", false);
      max_width = dataframe::utils::get<int>(params, "max_width", Ly);
      sample_bulk_mutual_information = dataframe::utils::get<int>(params, "sample_bulk_mutual_information", false);
    }

    std::vector<size_t> vector_complement(const std::vector<size_t>& sites, const std::vector<size_t>& all_sites) const {
      std::vector<bool> mask(all_sites.size(), false);
      for (size_t i = 0; i < sites.size(); i++) {
        mask[sites[i]] = true;
      }

      std::vector<size_t> complement;
      for (size_t i = 0; i < all_sites.size(); i++) {
        if (!mask[i]) {
          complement.push_back(i);
        }
      }

      return complement;
    }

    std::pair<std::vector<size_t>, std::vector<size_t>> bulk_symmetry_entropy_strip(size_t y0, size_t width) const {
      std::vector<bool> mask(Lx*Ly, false);

      std::vector<size_t> A;
      for (size_t x = 0; x < Lx; x++) {
        for (size_t y = 0; y < width; y++) {
          size_t i = to_index(x, y + y0);
          mask[i] = true;
          A.push_back(i);
        }
      }

      std::vector<size_t> Abar;
      for (size_t i = 0; i < Lx*Ly; i++) {
        if (!mask[i]) {
          Abar.push_back(i);
        }
      }

      return {A, Abar};
    }

    std::vector<double> bulk_symmetry_entropy(GeneratorMatrix& generators, size_t LA) const {
      std::vector<double> samples(Ly);
      for (size_t y0 = 0; y0 < Ly; y0++) {
        auto [A, Abar] = bulk_symmetry_entropy_strip(y0, LA);
        samples[y0] = generators.sym(A, Abar);
      }

      return samples;
    }

    std::pair<std::vector<size_t>, std::vector<size_t>> bulk_mi_strips(size_t y0, size_t seperation) const {
      std::vector<size_t> A(Lx);
      std::vector<size_t> B(Lx);

      for (size_t i = 0; i < Lx; i++) {
        A[i] = to_index(i, y0);
        B[i] = to_index(i, y0 + seperation);
      }

      return {A, B};
    }

    std::vector<double> bulk_mutual_information(GeneratorMatrix& generators) const {
      std::vector<double> samples(Ly);
      for (size_t y = 0; y < Ly; y++) {
        auto [A, B] = bulk_mi_strips(y, Ly/2);
        samples[y] = generators.sym(A, B);
      }

      return samples;
    }

    void add_samples(GeneratorMatrix& G, dataframe::DataSlide& slide) const {
      // Bulk symmetry entropy
      if (sample_bulk_symmetry) {
        std::vector<std::vector<double>> bulk_symmetry;
        for (size_t i = 1; i < max_width; i++) {
          std::vector<double> s = bulk_symmetry_entropy(G, i);
          bulk_symmetry.push_back(s);
        }

        slide.add_data("bulk_symmetry", bulk_symmetry.size());
        slide.push_samples_to_data("bulk_symmetry", bulk_symmetry);
      }

      // Bulk mutual information
      if (sample_bulk_mutual_information) {
        slide.add_data("bulk_mutual_information");
        auto mutual_information = bulk_mutual_information(G);
        slide.push_samples_to_data("bulk_mutual_information", std::vector<std::vector<double>>{mutual_information});
      }
    }
};



class LDPCConfig : public dataframe::Config {
  private:
    uint32_t Lx;
    uint32_t Ly;

    double pr;

    int model_type;
    double pb;
    size_t k;
    bool obc;

    bool avg_y;

    bool single_site;

    LinearCodeSampler sampler;
    BoundarySymmetrySampler sym_sampler;

    std::minstd_rand rng;

    uint32_t rand() {
      return rng();
    }

    double randf() {
      return double(rng())/double(RAND_MAX);
    }

    std::shared_ptr<ParityCheckMatrix> from_graph(const Graph<int>& g) {
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(Lx, Lx);

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
      Graph<int> g = Graph<int>::erdos_renyi_graph(Lx, pb, &rng);
      return from_graph(g);
    }

    std::shared_ptr<ParityCheckMatrix> generate_regular_interaction_matrix() {
      Graph<int> g = Graph<int>::random_regular_graph(Lx, k, &rng);
      return from_graph(g);
    }

    size_t mod(int i, int L) const {
      return (i % L + L) % L;
    }

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i % Lx;
      int y = i / Lx;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x, Lx) + mod(y, Ly) * Lx;
    }

    std::shared_ptr<ParityCheckMatrix> generate_5body_lattice_interaction_matrix() {
      size_t N = Lx*Ly;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (y == 0 || y == Ly - 1)) {
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
      size_t N = Lx*Ly;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (y == 0 || y == Ly - 1)) {
          continue;
        }

        size_t i1 = to_index(x+1, y);
        size_t i2 = to_index(x-1, y);
        size_t i3 = to_index(x, y+1);
        size_t i4 = to_index(x, y-1);

        std::vector<size_t> to_include;
        if (randf() < pr) {
          to_include = {i1, i2, i3, i4};
        } else {
          to_include = {i};
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
      size_t N = Lx*Ly;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (y == 0 || y == Ly - 1)) {
          continue;
        }

        size_t i1 = to_index(x, y+1);
        size_t i2 = to_index(x+1, y+1);

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

    std::shared_ptr<ParityCheckMatrix> generate_triangular_plaquette_interaction_matrix() {
      size_t N = Lx*Ly;
      std::shared_ptr<ParityCheckMatrix> A = std::make_shared<ParityCheckMatrix>(0, N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (y == 0 || y == Ly - 1)) {
          continue;
        }

        size_t i1 = to_index(x, y-1);
        size_t i2 = to_index(x-1, y-1);

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
      Lx = dataframe::utils::get<int>(params, "system_size");
      pr = dataframe::utils::get<double>(params, "pr", 0.0);

      model_type = dataframe::utils::get<int>(params, "model_type", LDPC_ERDOS);
      if (model_type == LDPC_ERDOS) {
        pb = dataframe::utils::get<double>(params, "pb", 0.0);
      } else if (model_type == LDPC_REGULAR) {
        k = dataframe::utils::get<int>(params, "k", 0);
      } else if (model_type == LDPC_LATTICE_5 || model_type == LDPC_LATTICE_3 || model_type == LDPC_LATTICE_4 || model_type == LDPC_TRIANGULAR_PLAQUETTE) {
        Ly = dataframe::utils::get<int>(params, "Ly", Lx);
        obc = dataframe::utils::get<int>(params, "obc", true);
        avg_y = dataframe::utils::get<int>(params, "avg_y", !obc);
        sym_sampler = BoundarySymmetrySampler(params, Lx, Ly);
      }

      single_site = dataframe::utils::get<int>(params, "single_site", true);

      int seed = dataframe::utils::get<int>(params, "seed", 0);
      if (seed == 0) {
        thread_local std::random_device rd;
        seed = rd();
      }

      rng.seed(seed);
    }

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
      } else if (model_type == LDPC_TRIANGULAR_PLAQUETTE) {
        A = generate_triangular_plaquette_interaction_matrix();
      }


      dataframe::DataSlide slide;
      sampler.add_samples(slide, A, rng);

      // ----------------- TEMP ----------------- //
      auto G = A->to_generator_matrix();

      sym_sampler.add_samples(G, slide);

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
