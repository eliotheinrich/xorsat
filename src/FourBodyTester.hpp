#pragma once

#include "Frame.h"
#include <Graph.hpp>
#include <BinaryPolynomial.h>
#include <Samplers.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

class FourBodyTester {
  private:
    size_t mod(int i, int L) const {
      return (i % L + L) % L;
    }

  public:
    ParityCheckMatrix p;

    int L;

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i % L;
      int y = i / L;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x, L) + mod(y, L) * L;
    }

    FourBodyTester(size_t L) : L(L) {
      p = ParityCheckMatrix(0, L*L);
      for (size_t x = 0; x < L; x++) {
        for (size_t y = 0; y < L; y++) {
          size_t i1 = to_index(x, y);
          size_t i2 = to_index(x+1, y);
          size_t i3 = to_index(x+1, y+1);
          size_t i4 = to_index(x, y+1);
          
          std::vector<size_t> inds{i1, i2, i3, i4};
          std::vector<bool> row(L*L, false);
          for (auto i : inds) {
            row[i] = true;
          }

          p.append_row(row);
        }
      }
    }

    bool in_solution_space(const std::vector<int>& vals) const {
      std::vector<bool> _vals(vals.size());
      for (size_t i = 0; i < vals.size(); i++) {
        _vals[i] = (bool) vals[i];
      }

      return p.is_in_space(_vals);
    }

    size_t cardinality(size_t x, size_t y) const {
      size_t n = 0;
      size_t i = to_index(x, y);

      for (size_t r = 0; r < p.num_rows; r++) {
        if (p.get(r, i)) {
          n++;
        }
      }

      return n;
    }

    size_t rank() {
      return p.rank(false);
    }

    std::string to_string() const {
      return p.to_string();
    }

    void add_impurity1(size_t x, size_t y) {
      size_t i = to_index(x, y);

      std::vector<bool> row(L*L, false);
      row[i] = true;
      p.append_row(row);
    }

    void add_impurity2(size_t x, size_t y) {
      size_t i = to_index(x, y);

      std::vector<bool> row(L*L, false);
      p.set_row(i, row);
    }
};
