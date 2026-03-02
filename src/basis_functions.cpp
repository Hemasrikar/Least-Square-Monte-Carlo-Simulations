#include "basis_functions.hpp"
#include <cmath>

namespace lms {

// ConstantBasis
class ConstantBasis : public BasisFunction {
public:
    double evaluate(double x) const {
        return 1.0;
    }

    std::string name() const {
        return "Const";
    }
};

// MonomialBasis
class MonomialBasis : public BasisFunction {
public:
    MonomialBasis(int power) {
        if (power < 0) {
            throw std::invalid_argument("power must be >= 0");
        }
        power_ = power;
    }

    double evaluate(double x) const {
        return std::pow(x, power_);
    }

    std::string name() const {
        return "x^" + std::to_string(power_);
    }

private:
    int power_;
};

}