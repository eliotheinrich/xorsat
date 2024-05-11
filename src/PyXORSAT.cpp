#include "CliffordCodeSimulator.hpp"
#include "XORSATConfig.hpp"
#include "LDPCConfig.hpp"
#include "CliffordCodeSimulator.hpp"
#include "GraphClusteringSimulator.hpp"
#include "GraphXORSATConfig.hpp"
#include "RXPMDualConfig.hpp"
#include "SlantedCheckerboardConfig.hpp"

#include <PyDataFrame.hpp>

NB_MODULE(xorsat_bindings, m) {
  INIT_CONFIG();
  EXPORT_CONFIG(XORSATConfig);
  EXPORT_CONFIG(LDPCConfig);
  EXPORT_CONFIG(GraphXORSATConfig);
  EXPORT_CONFIG(RXPMDualConfig);
  EXPORT_CONFIG(SlantedCheckerboardConfig);

  EXPORT_SIMULATOR_DRIVER(CliffordCodeSimulator);
  EXPORT_SIMULATOR_DRIVER(GraphClusteringSimulator);
}

