#include <Frame.h>
#include "LDPCConfig.hpp"

#include <fmt/core.h>

#include <iostream>

using namespace dataframe;
using namespace dataframe::utils;

int main(int argc, char* argv[]) {
  Params p = load_params(argv[1]);
  
  for (size_t i = 0; i < 100; i++) {
    LDPCConfig config(p);

    auto slide = config.compute(1);
  }
}
