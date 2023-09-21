// Landau Damping Test
//   Usage:
//     srun ./LandauDamping
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
//

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

#include "Random/InverseTransformSampling_ND.h"
#include "Random/Random.h"

constexpr unsigned Dim = 3;

struct custom_cdf{
    KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params) const {
          return x + (params[d*Dim+0] / params[d*Dim+1]) * Kokkos::sin(params[d*Dim+1] * x);
    }
};
struct custom_pdf{
    KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params) const {
          return  1.0 + params[d*Dim+0] * Kokkos::cos(params[d*Dim+1] * x);
    }
};
struct custom_estimate{
    KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params) const {
          return u + params[d]*0.;
    }
};

const char* TestName = "LandauDamping";

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        setSignalHandler();

        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();

        int arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef dumpDataTimer    = IpplTimings::getTimer("dumpData");
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

        IpplTimings::startTimer(mainTimer);

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        msg << "Landau damping" << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> kw = 0.5;
        double alpha             = 0.05;
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax = 2 * pi / kw;

        Vector_t<double, Dim> hr = rmax / nr;
        // Q = -\int\int f dx dv
        double Q = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        Vector_t<double, Dim> origin = rmin;
        const double dt              = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));

        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        std::string solver = argv[arg++];

        if (solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }

        P = std::make_unique<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

        P->nr_m = nr;

        P->initializeFields(mesh, FL);

        P->initSolver();
        P->time_m                 = 0.0;
        P->loadbalancethreshold_m = std::atof(argv[arg++]);
        
        bool isFirstRepartition;

       // Create initial distirbution of particle positions
       using DistR_t = ippl::random::Distribution<double, Dim, 2*Dim, custom_pdf, custom_cdf, custom_estimate>;
       const double parR[2*Dim] = {alpha, kw[0], alpha, kw[1], alpha, kw[2]};
       DistR_t distR(parR);

        if ((P->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            const int nghost               = P->rho_m.getNghost();
            auto rhoview                   = P->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", P->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.full_pdf(xvec);
                });

            Kokkos::fence();

            P->initializeORB(FL, mesh);
            P->repartition(FL, mesh, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        msg << "First domain decomposition done" << endl;
        IpplTimings::startTimer(particleCreation);

        typedef ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>::uniform_type RegionLayout_t;
        const RegionLayout_t& RLayout                           = PL.getRegionLayout();
        const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();

        int seed = 42;
        using size_type = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));
        using samplingH_t = ippl::random::sample_its<double, Dim, Kokkos::DefaultExecutionSpace, DistR_t>;
        samplingH_t samplingR(distR, rmax, rmin, RLayout, totalP);
        size_type nloc = samplingR.getLocalNum();
        P->create(nloc);
        samplingR.generate(P->R.getView(), rand_pool64);

        Kokkos::parallel_for(
            nloc, ippl::random::randn_functor<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->P.getView(), rand_pool64));

        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(particleCreation);

        P->q = P->Q_m / totalP;
        msg << "particles created and initial conditions assigned " << endl;
        isFirstRepartition = false;
        // The update after the particle creation is not needed as the
        // particles are generated locally

        IpplTimings::startTimer(DummySolveTimer);
        P->rho_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP, 0, hr);

        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        P->gatherCIC();

        IpplTimings::startTimer(dumpDataTimer);
        //P->dumpLandau();
        //P->gatherStatistics(totalP);
        // P->dumpLocalDomains(FL, 0);
        IpplTimings::stopTimer(dumpDataTimer);

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            IpplTimings::startTimer(PTimer);
            P->P = P->P - 0.5 * dt * P->E;
            IpplTimings::stopTimer(PTimer);

            // drift
            IpplTimings::startTimer(RTimer);
            P->R = P->R + dt * P->P;
            IpplTimings::stopTimer(RTimer);
            // P->R.print();

            // Since the particles have moved spatially update them to correct processors
            IpplTimings::startTimer(updateTimer);
            P->update();
            IpplTimings::stopTimer(updateTimer);

            // Domain Decomposition
            
            if (P->balance(totalP, it + 1)) {
                msg << "Starting repartition" << endl;
                IpplTimings::startTimer(domainDecomposition);
                P->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
                // IpplTimings::startTimer(dumpDataTimer);
                // P->dumpLocalDomains(FL, it+1);
                // IpplTimings::stopTimer(dumpDataTimer);
            }

            // scatter the charge onto the underlying grid
            P->scatterCIC(totalP, it + 1, hr);

            // Field solve
            IpplTimings::startTimer(SolveTimer);
            P->runSolver();
            IpplTimings::stopTimer(SolveTimer);

            // gather E field
            P->gatherCIC();

            // kick
            IpplTimings::startTimer(PTimer);
            P->P = P->P - 0.5 * dt * P->E;
            IpplTimings::stopTimer(PTimer);

            P->time_m += dt;
            IpplTimings::startTimer(dumpDataTimer);
            P->dumpLandau();
            P->gatherStatistics(totalP);
            IpplTimings::stopTimer(dumpDataTimer);
            msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;

            if (checkSignalHandler()) {
                msg << "Aborting timestepping loop due to signal " << interruptSignalReceived
                    << endl;
                break;
            }
        }

        msg << "LandauDamping: End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
    }
    ippl::finalize();

    return 0;
}