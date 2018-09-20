#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>


namespace py = pybind11;

void noop(const std::vector<double>& vec) {
    ;
}

PYBIND11_MODULE(module, m) {
    m.def("noop", &noop, py::call_guard<py::gil_scoped_release>());
}
