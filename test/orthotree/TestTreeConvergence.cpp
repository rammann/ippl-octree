/*
USAGE

./TestTreeConvergence nTargetsstart maxElementsperleaf --info 5

nTargetsstart : number of targetpoints starting value
maxElementsPercent : max elements per leaf node

loops over nTargetsstart ... [1,9] x nTargetsstart

total points is = source points + target points = 2 x target points


*/
#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "Utility/IpplTimings.h"

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma)) ;
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

typedef ippl::ParticleSpatialLayout<double, 3> playout_type;

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    {
        // IO
        unsigned int nTargetsstart = std::atoi(argv[1]);
        unsigned int maxElements = std::stoi(argv[2]);

        std::vector<double> errors;
        std::vector<unsigned int> numbertargets;

        for(unsigned int mult = 1; mult < 10; ++mult){
            std::cout << "It = " << mult << "\n";
            unsigned int nTargets = mult * nTargetsstart;
            
            // Generate Points   
            playout_type PLayout;
            ippl::OrthoTreeParticle targets(PLayout);
            targets.create(nTargets);
            ippl::OrthoTreeParticle sources(PLayout);
            unsigned int nSources = nTargets;
            sources.create(nSources);
            std::mt19937_64 eng(4);
            std::uniform_real_distribution<double> posDis(0.0, 1.0);

            for(unsigned int idx=0; idx<nTargets; ++idx){
                ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
                targets.R(idx) = r;
                targets.rho(idx) = 0.0;
            }

            for(unsigned int idx=0; idx<nSources; ++idx){
                ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
                sources.R(idx) = r;
                sources.rho(idx) = gaussian(r[0], r[1], r[2]);
            }

            ippl::ParameterList treeparams;
            treeparams.add("maxdepth",          6);
            treeparams.add("maxleafelements",   static_cast<int>(maxElements));
            treeparams.add("boxmin",            0.0);
            treeparams.add("boxmax",            1.0);
            treeparams.add("sourceidx",         nTargets);
        
            ippl::ParameterList solverparams;
            solverparams.add("eps", 1e-3);

            ippl::TreeOpenPoissonSolver solver(targets, sources, treeparams, solverparams);
            auto explicitsol = solver.ExplicitSolution();
            solver.Solve();

            double mape = 0.0;
            double* const ptr = &mape;

            Kokkos::parallel_for("Calculate MAPE", nTargets,
            KOKKOS_LAMBDA(unsigned int i){
                Kokkos::atomic_add(ptr, (Kokkos::abs(explicitsol(i)-targets.rho(i))) / Kokkos::abs(explicitsol(i)));
            });

            mape /= nTargets;
            // std::cout << "nTargets = " << nTargets << ", "; 
            // std::cout << "MAPE = " << mape << "\n";
            numbertargets.push_back(nTargets);
            errors.push_back(mape);
        }
        std::cout << "MAPE:" << "\n";
        for(unsigned int i=0; i<errors.size(); ++i){
            std::cout << errors[i] << ", "; 
        }
        std::cout << "\n";
        std::cout << "nTargets:" << "\n";
        for(unsigned int i=0; i<numbertargets.size(); ++i){
            std::cout << numbertargets[i] << ", "; 
        }
        std::cout << "nTotal:" << "\n";
        for(unsigned int i=0; i<numbertargets.size(); ++i){
            std::cout << 2*numbertargets[i] << ", "; 
        }
        std::cout << "\n";
        

    }
    ippl::finalize();
}