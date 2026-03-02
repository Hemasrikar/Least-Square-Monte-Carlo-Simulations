// =============================================================================
//  main.cpp  —  LSM American Option Pricer  (IB9JHO Group Project)
//
//  Sections:
//    1. American put  — vary spot, maturity, volatility
//    2. American call — basic sanity (call on non-dividend stock ≈ European)
//    3. Jump-diffusion put
//    4. Convergence analysis — value vs. basis functions M
//    5. Convergence analysis — value vs. path count N
//    6. Out-of-sample stability test
//    7. Benchmark table — L&S (2001) Table 1 reference cases
// =============================================================================

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "lsm_types.hpp"
#include "stochastic_processes.hpp"
#include "payoffs.hpp"
#include "basis_functions.hpp"
#include "lsm_pricer.hpp"
#include "convergence_analyzer.hpp"

using namespace lsm;

// ---------------------------------------------------------------------------
//  Formatting helpers
// ---------------------------------------------------------------------------
static void separator(char c = '-', int w = 72) {
    std::cout << std::string(w, c) << "\n";
}

static void printResult(const std::string& label, double spot,
                        const SimulationResult& res)
{
    std::cout << std::fixed << std::setprecision(4)
              << std::setw(30) << std::left  << label
              << "  S="        << std::setw(6) << spot
              << "  Am="       << std::setw(7) << res.optionValue
              << "  Eu="       << std::setw(7) << res.europeanValue
              << "  EEP="      << std::setw(7) << res.earlyExercisePremium
              << "  SE="       << res.standardError << "\n";
}

// ---------------------------------------------------------------------------
//  Build a standard put pricer
// ---------------------------------------------------------------------------
static LSMPricer makePutPricer(double K, double r, double sigma,
                                double T, int N, int dates, int seed = 42)
{
    LSMConfig cfg;
    cfg.numPaths         = N;
    cfg.useAntithetic    = false;
    cfg.numExerciseDates = dates;
    cfg.maturity         = T;
    cfg.riskFreeRate     = r;
    cfg.rngSeed          = seed;

    return LSMPricer(cfg,
                     std::make_unique<GeometricBrownianMotion>(r, sigma),
                     std::make_unique<PutPayoff>(K),
                     makeLaguerreSet(3));
}

// =============================================================================
int main()
{
    std::cout << "\n";
    separator('=');
    std::cout << "  Longstaff-Schwartz LSM American Option Pricer — IB9JHO\n";
    separator('=');

    // =========================================================================
    //  1. American put: vary spot
    // =========================================================================
    std::cout << "\n[1] American Put  K=40  r=6%  sigma=20%  T=1yr  N=10,000\n";
    separator();
    std::cout << std::left
              << std::setw(30) << "Case"
              << "  Spot    Am       Eu       EEP      SE\n";
    separator();
    for (double S : {36.0, 38.0, 40.0, 42.0, 44.0}) {
        auto p = makePutPricer(40, 0.06, 0.20, 1.0, 10000, 50);
        printResult("AmericanPut", S, p.price(S));
    }

    // =========================================================================
    //  2. Vary maturity
    // =========================================================================
    std::cout << "\n[2] American Put: vary maturity  S=40  K=40  r=6%  sigma=20%\n";
    separator();
    for (double T : {0.5, 1.0, 2.0}) {
        auto p = makePutPricer(40, 0.06, 0.20, T, 10000,
                               static_cast<int>(50 * T));
        printResult("T=" + std::to_string(T).substr(0,3) + "yr", 40.0, p.price(40));
    }

    // =========================================================================
    //  3. Vary volatility
    // =========================================================================
    std::cout << "\n[3] American Put: vary sigma  S=40  K=40  r=6%  T=1yr\n";
    separator();
    for (double sig : {0.10, 0.20, 0.30, 0.40}) {
        auto p = makePutPricer(40, 0.06, sig, 1.0, 10000, 50);
        printResult("sigma=" + std::to_string(sig).substr(0,4), 40.0, p.price(40));
    }

    // =========================================================================
    //  4. American call
    // =========================================================================
    std::cout << "\n[4] American Call  K=40  r=6%  sigma=20%  T=1yr  N=10,000\n";
    separator();
    std::cout << "    (For non-dividend stocks, American call = European call;\n"
              << "     early exercise premium should be ~0)\n";
    separator();
    for (double S : {36.0, 40.0, 44.0}) {
        LSMConfig cfg;
        cfg.numPaths = 10000; cfg.numExerciseDates = 50;
        cfg.maturity = 1.0;   cfg.riskFreeRate = 0.06; cfg.rngSeed = 42;
        LSMPricer p(cfg,
                    std::make_unique<GeometricBrownianMotion>(0.06, 0.20),
                    std::make_unique<CallPayoff>(40.0),
                    makeLaguerreSet(3));
        printResult("AmericanCall", S, p.price(S));
    }

    // =========================================================================
    //  5. Jump-diffusion
    // =========================================================================
    std::cout << "\n[5] Jump-Diffusion Put  S=40  K=40  r=6%  T=1yr  N=10,000\n";
    separator();
    std::cout << "    (lambda=0 is pure GBM; sigma adjusted to equalise variance)\n";
    separator();
    for (double lambda : {0.00, 0.05, 0.10}) {
        double sigma = (lambda == 0.0) ? 0.30 : 0.20;
        LSMConfig cfg;
        cfg.numPaths = 10000; cfg.numExerciseDates = 50;
        cfg.maturity = 1.0;   cfg.riskFreeRate = 0.06; cfg.rngSeed = 42;
        LSMPricer p(cfg,
                    std::make_unique<JumpDiffusionProcess>(0.06, sigma, lambda),
                    std::make_unique<PutPayoff>(40.0),
                    makeLaguerreSet(3));
        printResult("lambda=" + std::to_string(lambda).substr(0,4), 40.0, p.price(40));
    }

    // =========================================================================
    //  6. Convergence: value vs. number of basis functions M
    // =========================================================================
    std::cout << "\n[6] Convergence vs. Basis Functions M\n";
    std::cout << "    S=40  K=40  r=6%  sigma=20%  T=1yr  N=10,000\n";
    std::cout << "    (LSM value is a lower bound — should rise then stabilise with M)\n";
    separator();
    std::cout << std::setw(6)  << "M"
              << std::setw(12) << "Value"
              << std::setw(12) << "Std Error" << "\n";
    separator();
    {
        LSMConfig cfg;
        cfg.numPaths = 10000; cfg.numExerciseDates = 50;
        cfg.maturity = 1.0;   cfg.riskFreeRate = 0.06; cfg.rngSeed = 42;

        auto rows = ConvergenceAnalyzer::analyzeByBasisFunctions(cfg, 40.0, 40.0, 0.20, 5);
        for (auto& [M, val, se] : rows) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(6)  << M
                      << std::setw(12) << val
                      << std::setw(12) << se << "\n";
        }
    }

    // =========================================================================
    //  7. Convergence: value vs. path count N
    // =========================================================================
    std::cout << "\n[7] Convergence vs. Path Count N\n";
    std::cout << "    S=40  K=40  r=6%  sigma=20%  T=1yr  M=3 Laguerre\n";
    std::cout << "    (Standard error should fall proportionally to 1/sqrt(N))\n";
    separator();
    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Value"
              << std::setw(12) << "Std Error"
              << std::setw(14) << "SE * sqrt(N)" << "\n";
    separator();
    {
        LSMConfig cfg;
        cfg.numExerciseDates = 50; cfg.maturity = 1.0;
        cfg.riskFreeRate = 0.06;   cfg.rngSeed = 42;

        std::vector<int> Ns = {500, 1000, 2000, 5000, 10000, 20000};
        auto rows = ConvergenceAnalyzer::analyzeByPathCount(cfg, 40.0, 40.0, 0.20, Ns);
        for (auto& [N, val, se] : rows) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(10) << N
                      << std::setw(12) << val
                      << std::setw(12) << se
                      << std::setw(14) << se * std::sqrt(static_cast<double>(N)) << "\n";
        }
    }

    // =========================================================================
    //  8. Out-of-sample stability
    // =========================================================================
    std::cout << "\n[8] Out-of-Sample Stability Test\n";
    std::cout << "    S=40  K=40  r=6%  sigma=20%  T=1yr  N=5,000  5 trials\n";
    std::cout << "    (In-sample and out-of-sample values should be close)\n";
    separator();
    std::cout << std::setw(8)  << "Trial"
              << std::setw(14) << "In-Sample"
              << std::setw(14) << "Out-of-Sample"
              << std::setw(12) << "Difference" << "\n";
    separator();
    {
        LSMConfig cfg;
        cfg.numPaths = 5000; cfg.numExerciseDates = 50;
        cfg.maturity = 1.0;  cfg.riskFreeRate = 0.06; cfg.rngSeed = 42;

        auto trials = ConvergenceAnalyzer::outOfSampleTest(cfg, 40.0, 40.0, 0.20, 5);
        for (int i = 0; i < static_cast<int>(trials.size()); ++i) {
            auto& [inR, outR] = trials[i];
            double diff = outR.optionValue - inR.optionValue;
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(8)  << (i + 1)
                      << std::setw(14) << inR.optionValue
                      << std::setw(14) << outR.optionValue
                      << std::setw(12) << diff << "\n";
        }
    }

    // =========================================================================
    //  9. Benchmark table — L&S (2001) Table 1 reference cases
    //     K=40, r=6%, 50 exercise dates per year, N=20,000
    //     Finite-difference reference values shown for comparison
    // =========================================================================
    std::cout << "\n[9] Benchmark Table  (L&S 2001 Table 1 reference cases)\n";
    std::cout << "    K=40  r=6%  N=20,000  50 exercise dates/year\n";
    separator();
    std::cout << std::setw(6)  << "S"
              << std::setw(7)  << "sigma"
              << std::setw(6)  << "T"
              << std::setw(10) << "LSM"
              << std::setw(10) << "FD Ref"
              << std::setw(10) << "Diff"
              << std::setw(10) << "SE" << "\n";
    separator();

    // {spot, sigma, T, finite-difference reference value}
    struct BenchCase { double S, sigma, T, fdRef; };
    std::vector<BenchCase> cases = {
        {36, 0.20, 1.0, 4.478},
        {36, 0.20, 2.0, 4.840},
        {36, 0.40, 1.0, 7.101},
        {36, 0.40, 2.0, 8.508},
        {38, 0.20, 1.0, 3.250},
        {38, 0.20, 2.0, 3.745},
        {38, 0.40, 1.0, 6.148},
        {38, 0.40, 2.0, 7.670},
        {40, 0.20, 1.0, 2.314},
        {40, 0.20, 2.0, 2.885},
        {40, 0.40, 1.0, 5.312},
        {40, 0.40, 2.0, 6.920},
        {42, 0.20, 1.0, 1.617},
        {42, 0.20, 2.0, 2.212},
        {42, 0.40, 1.0, 4.582},
        {42, 0.40, 2.0, 6.248},
        {44, 0.20, 1.0, 1.110},
        {44, 0.20, 2.0, 1.690},
        {44, 0.40, 1.0, 3.948},
        {44, 0.40, 2.0, 5.647},
    };

    for (auto& c : cases) {
        int dates = static_cast<int>(50 * c.T);
        auto p    = makePutPricer(40.0, 0.06, c.sigma, c.T, 20000, dates);
        auto res  = p.price(c.S);
        double diff = res.optionValue - c.fdRef;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(6)  << c.S
                  << std::setw(7)  << c.sigma
                  << std::setw(6)  << c.T
                  << std::setw(10) << res.optionValue
                  << std::setw(10) << c.fdRef
                  << std::setw(10) << diff
                  << std::setw(10) << res.standardError << "\n";
    }

    separator('=');
    std::cout << "Done.\n\n";
    return 0;
}
