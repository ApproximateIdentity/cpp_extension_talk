#include <lr.hpp>
#include <cmath>
#include <thread>
#include <mutex>

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
    // Define a workhorse reduction function.
    std::mutex mtx;
    auto reducer = [&] (unsigned int start, unsigned int stop) {
        // Store the partial gradients.
        double partial_da1 = 0, partial_da2 = 0, partial_db = 0;
        for (unsigned int i = start; i < stop; ++i) {
            const auto x1 = X.at(i).first;
            const auto x2 = X.at(i).second;
            const auto y = Y.at(i);
            partial_da1 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x1;
            partial_da2 -= h((-y * (a1 * x1 + a2 * x2 + b))) * y * x2;
            partial_db  -= h((-y * (a1 * x1 + a2 * x2 + b))) * y;
        }
        std::lock_guard<std::mutex> lock(mtx);
        da1 += partial_da1;
        da2 += partial_da2;
        db  += partial_db;
    };
    // Run the reduction function.
    unsigned int num_samples = X.size();
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < _num_threads; ++i) {
        unsigned int start = (i * num_samples) / _num_threads;
        unsigned int stop = ((i + 1) * num_samples) / _num_threads;
        threads.push_back(std::thread(reducer, start, stop));
    }
    for (auto& thread: threads) {
        thread.join();
    }
    return _Gradient { da1, da2, db };
}
