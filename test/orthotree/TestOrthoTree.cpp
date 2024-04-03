#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {
        /* SOLVER INIT TEST*/
        
        // Particles
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;

        ippl::OrthoTreeParticle source_particles(PLayout, 0);
        unsigned int nsources = 50;
        source_particles.create(nsources);

        ippl::OrthoTreeParticle target_particles(PLayout, 0);
        unsigned int ntargets = nsources;
        target_particles.create(ntargets);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0, 1);
        for(unsigned int idx=0; idx<ntargets; ++idx){
            ippl::Vector<double,3> r1 = {unif(eng), unif(eng), unif(eng)};
            ippl::Vector<double,3> r2 = {unif(eng), unif(eng), unif(eng)};
            source_particles.R(idx) = r1;
            target_particles.R(idx) = r2;
            source_particles.rho(idx) = 0.0;
            target_particles.rho(idx) = 0.0;
        }


        // Tree Params
        ippl::ParameterList params;
        params.add("maxdepth",          5);
        params.add("maxleafelements",   5);
        params.add("boxmin",            0.0);
        params.add("boxmax",            1.0);

        
        ippl::TreeOpenPoissonSolver tree(source_particles, 20, params);






        /* UNORDERED MAP TEST 
        Kokkos::UnorderedMap<int, double> map;
        map.insert(1, 1.5);
        map.insert(2, 2.5);

        double& ref = map.value_at(map.find(1));
        ref = 12.5;
        
        std::cout << map.value_at(map.find(1)) << " " << map.value_at(map.find(2)) << std::endl;
        */

        /* OCTREE CONSTRUCTION TEST 
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout, 5);
        unsigned int nsources = 50;

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0, 1);


        particles.create(nsources);
        for (unsigned int i=0; i<nsources; ++i){
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            //std::cout << r[0] << " " << r[1] << " " << r[2] << std::endl;
            particles.R(i)                 = r;
            particles.rho(i)               = 0.0;
        }

        ippl::OrthoTree tree(particles, 4, ippl::BoundingBox<3>{{0,0,0},{1,1,1}}, 3);
        tree.PrintStructure();
        */

        /* LocationIterator distance TEST 
        
        Kokkos::vector<double> vec(100);
        using LocationIterator = typename Kokkos::vector<double>::iterator;
        LocationIterator begin = vec.begin();
        LocationIterator end = vec.end();

        std::cout << end - begin << std::endl;
        */
    }

    ippl::finalize();
    
    return 0;
}