


//  BasisFunction  
// — Abstract: strategy for regression basis

class BasisFunction {
public:
    virtual ~BasisFunction() {}
    virtual double evaluate(double x) const = 0;
    virtual std::string name() const = 0;
};