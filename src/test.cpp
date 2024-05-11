#include <Frame.h>
#include "LDPCConfig.hpp"
#include "XORSATConfig.hpp"
#include "RXPMDualConfig.hpp"

#include <iostream>

using namespace dataframe;
using namespace dataframe::utils;

int main(int argc, char* argv[]) {
  Params params;
  params.emplace("system_size", 4.0);
  params.emplace("p", 0.0);
  params.emplace("num_runs", 10.0);

  RXPMDualConfig config(params);
  auto state = config.prepare_state();
  std::cout << state.to_string() << "\n";
}
