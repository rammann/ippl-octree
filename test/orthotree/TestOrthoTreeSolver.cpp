#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "Utility/IpplTimings.h"

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    {
        // Setup
        static auto timer = IpplTimings::getTimer("Orthotree Poisson Solver");
        
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;

        // Targets
        ippl::OrthoTreeParticle targets(PLayout);
        unsigned int nTargets = 10000;
        targets.create(nTargets);

        // Sources
        ippl::OrthoTreeParticle sources(PLayout);
        unsigned int nSources = 10000;
        sources.create(nSources);

        // Random generators for position and charge
        std::mt19937_64 eng(43);
        std::uniform_real_distribution<double> posDis(0.0, 1.0);
        std::uniform_real_distribution<double> posDist(0.0, 0.5);
        std::uniform_real_distribution<double> posDiss(0.5, 1.0);
        std::uniform_real_distribution<double> chargeDis(-20,20);

        // Generate target points
        
        for(unsigned int idx=0; idx<nTargets; ++idx){
        ippl::Vector<double,3> r = {posDist(eng), posDist(eng), posDist(eng)};
            targets.R(idx) = r;
            targets.rho(idx) = 0.0;
        }

        // Generate source points
        for(unsigned int idx=0; idx<nSources; ++idx){
        ippl::Vector<double,3> r = {posDiss(eng), posDiss(eng), posDiss(eng)};
            sources.R(idx) = r;
            sources.rho(idx) = chargeDis(eng);
        }
        
        // Tree Params
        ippl::ParameterList treeparams;
        treeparams.add("maxdepth",          7);
        treeparams.add("maxleafelements",   100);
        treeparams.add("boxmin",            0.0);
        treeparams.add("boxmax",            1.0);
        treeparams.add("sourceidx",         nTargets);

        // Solver Params
        ippl::ParameterList solverparams;
        solverparams.add("eps", 1e-6);

        
        ippl::TreeOpenPoissonSolver solver(targets, sources, treeparams, solverparams);


        IpplTimings::startTimer(timer);
        solver.Solve();
        IpplTimings::stopTimer(timer);
        IpplTimings::print();

        auto explicitsol = solver.ExplicitSolution();
        double mse = 0.0;
        for(unsigned int i = 0; i<nTargets; ++i){
            mse += (Kokkos::abs(explicitsol(i)-targets.rho(i)));
        }
        mse = mse/nTargets;
        std::cout << "MSE = " << mse << "\n";

         
    
    }
    ippl::finalize();
}