#pragma once

//  basis_functions.hpp  —  Concrete basis function implementations

#include "lsm_types.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace lsm {

//  ConstantBasis  —  intercept / bias term (evaluates to 1.0)

class ConstantBasis : public BasisFunction {
public:
    double evaluate(double /*x*/) const override { return 1.0; }
    std::string name() const override { return "Const"; }
};

//  LaguerrePolynomial  —  weighted Laguerre basis (§2.2, Equations 2–5)

class LaguerrePolynomial : public BasisFunction {
public:
    explicit LaguerrePolynomial(int order) : order_(order) {
        if (order < 0 || order > 5)
            throw std::invalid_argument("LaguerrePolynomial: order must be 0–5");
    }

    double evaluate(double x) const override {
        // Guard against large x causing underflow in exp(−x/2)
        if (x < 0.0) x = 0.0;
        const double e = std::exp(-x / 2.0);
        switch (order_) {
            case 0: return e;
            case 1: return e * (1.0 - x);
            case 2: return e * (1.0 - 2.0*x + 0.5*x*x);
            case 3: return e * (1.0 - 3.0*x + 1.5*x*x - x*x*x/6.0);
            case 4: return e * (1.0 - 4.0*x + 3.0*x*x - 2.0*x*x*x/3.0
                                      + x*x*x*x/24.0);
            case 5: {
                double x2 = x*x, x3 = x2*x, x4 = x3*x, x5 = x4*x;
                return e * (1.0 - 5.0*x + 5.0*x2 - 5.0*x3/3.0
                                 + 5.0*x4/24.0 - x5/120.0);
            }
            default: return 0.0;
        }
    }

    std::string name() const override {
        return "Laguerre_L" + std::to_string(order_);
    }

private:
    int order_;
};

}