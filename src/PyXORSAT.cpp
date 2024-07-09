#include "CliffordCodeSimulator.hpp"
#include "XORSATConfig.hpp"
#include "LDPCConfig.hpp"
#include "CliffordCodeSimulator.hpp"
#include "GraphClusteringSimulator.hpp"
#include "GraphXORSATConfig.hpp"
#include "RXPMDualConfig.hpp"
#include "RPMCAConfig.hpp"
#include "SlantedCheckerboardConfig.hpp"

#include "FourBodyTester.hpp"

#include <PyDataFrame.hpp>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>

NB_MODULE(xorsat_bindings, m) {
  INIT_CONFIG();
  EXPORT_CONFIG(XORSATConfig);
  EXPORT_CONFIG(LDPCConfig);
  EXPORT_CONFIG(GraphXORSATConfig);
  EXPORT_CONFIG(RXPMDualConfig);
  EXPORT_CONFIG(RPMCAConfig);
  EXPORT_CONFIG(SlantedCheckerboardConfig);

  EXPORT_SIMULATOR_DRIVER(CliffordCodeSimulator);
  EXPORT_SIMULATOR_DRIVER(GraphClusteringSimulator);

  nanobind::class_<FourBodyTester>(m, "FourBodyTester")
    .def(nanobind::init<size_t>())
    .def("__str__", &FourBodyTester::to_string)
    .def("to_generators", [](FourBodyTester& fbt) { auto g = fbt.p.to_generator_matrix(); return g.to_string(); })
    .def("rank", &FourBodyTester::rank)
    .def("grank", [](FourBodyTester& t) { return t.L*t.L - t.rank(); })
    .def("to_index", &FourBodyTester::to_index)
    .def("to_coordinates", &FourBodyTester::to_coordinates)
    .def("cardinality", &FourBodyTester::cardinality)
    .def("in_solution_space", &FourBodyTester::in_solution_space)
    .def("add_field_impurity", &FourBodyTester::add_impurity1)
    .def("add_break_impurity", &FourBodyTester::add_impurity2)
    .def("__str__", &FourBodyTester::to_string);

  nanobind::class_<FourBodyGraphGenerators>(m, "FourBodyGraphGenerators")
    .def(nanobind::init<size_t>())
    .def("__str__", [](FourBodyGraphGenerators& fbg) { return fbg.g.to_string(); })
    .def("add_field_impurity", &FourBodyGraphGenerators::add_field_impurity)
    .def("add_break_impurity", &FourBodyGraphGenerators::add_break_impurity)
    .def("components", &FourBodyGraphGenerators::get_components)
    .def("rank", &FourBodyGraphGenerators::get_rank);

}

