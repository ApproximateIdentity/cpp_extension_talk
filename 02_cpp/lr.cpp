#include <lr.hpp>
#include <cmath>

using namespace logisticregression;


Coefficients LogisticRegression::compute_coefficients(
        const std::vector<Point>& X, const std::vector<Label>& Y) {

    // Some fixed constants
    const double dlt = 0.01;
    const double eps = 0.000001;
    const unsigned int max_iter = 100;
    // Choose starting point at origin.
    double a1 = 0, a2 = 0, b = 0;
    for (unsigned int counter = 0; counter < max_iter; ++counter) {
        const auto grad = _compute_gradient(X, Y, a1, a2, b);
        if ((grad.da1 * grad.da1 + grad.da2 * grad.da2 + grad.db * grad.db) < (eps * eps)) {
            break;
        }
        a1 -= dlt * grad.da1;
        a2 -= dlt * grad.da2;
        b  -= dlt * grad.db;
    }
    return Coefficients(a1, a2, b);
}

static double h(double z) {
    return 1 / (1 + std::exp(-z));
}

_Gradient LogisticRegression::_compute_gradient(
        const std::vector<Point>& X, const std::vector<Label>& Y,
        const double a1, const double a2, const double b) {

    // This is the regularization term.
    double da1 = a1, da2 = a2, db = b;
    // This is the rest of the cost function.
    for (unsigned int i = 0; i < X.size(); ++i) {
        const auto x1 = X.at(i).first;
        const auto x2 = X.at(i).second;
        const auto y = Y.at(i);
        da1 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x1;
        da2 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x2;
        db  -= h((-y * (a1 * x1 + a2 * x2 + b))) * y;
    }
    return _Gradient { da1, da2, db };
}
