#ifndef LR_HPP
#define LR_HPP

#include <vector>
#include <tuple>

namespace logisticregression {

typedef std::pair<double, double> Point;
typedef long int Label;
typedef std::tuple<double, double, double> Coefficients;

struct _Gradient {
    double da1;
    double da2;
    double db;
};

class LogisticRegression {
    public:
        LogisticRegression(unsigned int num_threads) : _num_threads{num_threads} {}
        Coefficients compute_coefficients(
            const std::vector<Point>& X, const std::vector<Label>& Y);
    private:
        const unsigned int _num_threads;
        _Gradient _compute_gradient(
            const std::vector<Point>& X, const std::vector<Label>& Y,
            const double a1, const double a2, const double b);
};

}

#endif /* LR_HPP */
