#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <lr.hpp>

namespace lr = logisticregression;
namespace py = pybind11;


class LogisticRegressionWrapper: public lr::LogisticRegression {
    public:
        LogisticRegressionWrapper(unsigned int num_threads) : lr::LogisticRegression(num_threads) {}
        std::tuple<double, double, double> compute_coefficients(
            const py::array_t<double>& X, const py::array_t<long>& Y) {

            const double* xbegin = static_cast<const double*>(X.data());
            const double* xend = xbegin + X.size();
            const long* ybegin = static_cast<const long*>(Y.data());
            const long* yend = ybegin + Y.size();
            py::gil_scoped_release release;
            return lr::LogisticRegression::compute_coefficients(xbegin, xend, ybegin, yend);
        }
};

PYBIND11_MODULE(lrcpp, m) {
    py::class_<LogisticRegressionWrapper>(m, "LogisticRegression")
        .def(py::init<unsigned int>(),
            py::arg("num_threads") = 1)
        .def("compute_coefficients",
             &LogisticRegressionWrapper::compute_coefficients);
}
