#pragma once

#include "lsm_types.hpp"
#include <cmath>
#include <string>
#include <stdexcept>


//  ConstantBasis  
// - intercept term; evaluates to 1.0 for any given values

class ConstantBasis : public BasisFunction {
public:
    double evaluate(double x) const;
    std::string name() const;
};

// Monomial Basis Function

class MonomialBasis : public BasisFunction {
public:
    MonomialBasis(int power);

    double evaluate(double x) const;
    std::string name() const;

private:
    int power_;
};