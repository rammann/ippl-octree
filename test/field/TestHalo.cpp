#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);
    Inform msg("TestHalo");

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);

    constexpr unsigned int dim = 3;

//     std::array<int, dim> pt = {8, 7, 13};
    std::array<int, dim> pt = {4, 4, 4};
    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    typedef ippl::FieldLayout<dim> Layout_t;
    ippl::FieldLayout<dim> layout(owned, allParallel, true);

    std::array<double, dim> dx = {
        1.0 / double(pt[0]),
        1.0 / double(pt[1]),
        1.0 / double(pt[2]),
    };
    ippl::Vector<double, 3> hx = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout);

    //field = 1.0;//Ippl::Comm->rank();
     int myRank = Ippl::Comm->rank();
     int nRanks = Ippl::Comm->size();

    
    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == Ippl::Comm->rank()) {
            using face_neighbor_type = typename Layout_t::face_neighbor_type;
            using bound_type = typename Layout_t::bound_type;
            const face_neighbor_type& face_neighbors = layout.getFaceNeighbors();
            using face_neighbor_range_type = typename Layout_t::face_neighbor_range_type;
            //const face_neighbor_range_type& face_neighbors_send_range = layout.getFaceNeighborsSendRange();
            const face_neighbor_range_type& face_neighbors_recv_range = layout.getFaceNeighborsRecvRange();
            
            for (size_t face = 0; face < face_neighbors.size(); ++face) {
                for (size_t i = 0; i < face_neighbors[face].size(); ++i) {

                    int rank = face_neighbors[face][i];
                    //bound_type rangeSend;
                    bound_type rangeRecv;
                    //rangeSend = face_neighbors_send_range[face][i];
                    rangeRecv = face_neighbors_recv_range[face][i];
                    std::cout << "My Rank: " << myRank  
                              << "face: " << face 
                              //<< "neighbor rank: " << rank << std::endl;
                              //<< "Send range low 0: " << rangeSend.lo[0]
                              //<< "Send range hi 0: " << rangeSend.hi[0]
                              //<< "Send range low 1: " << rangeSend.lo[1]
                              //<< "Send range hi 1: " << rangeSend.hi[1]
                              //<< "Send range low 2: " << rangeSend.lo[2]
                              //<< "Send range hi 2: " << rangeSend.hi[2] << std::endl;
                              << "Recv range low 0: " << rangeRecv.lo[0]
                              << "Recv range hi 0: " << rangeRecv.hi[0]
                              << "Recv range low 1: " << rangeRecv.lo[1]
                              << "Recv range hi 1: " << rangeRecv.hi[1]
                              << "Recv range low 2: " << rangeRecv.lo[2]
                              << "Recv range hi 2: " << rangeRecv.hi[2] << std::endl;
                }
            }
            
            //using edge_neighbor_type = typename Layout_t::edge_neighbor_type;
            //const edge_neighbor_type& edge_neighbors = layout.getEdgeNeighbors();
            //for (size_t edge = 0; edge < edge_neighbors.size(); ++edge) {
            //    for (size_t i = 0; i < edge_neighbors[edge].size(); ++i) {

            //        int rank = edge_neighbors[edge][i];
            //        std::cout << "My Rank: " << myRank  
            //                  << "edge: " << edge 
            //                  << "neighbor rank: " << rank << std::endl;
            //    }
            //}
            //using vertex_neighbor_type = typename Layout_t::vertex_neighbor_type;
            //const vertex_neighbor_type& vertex_neighbors = layout.getVertexNeighbors();
            //for (size_t vertex = 0; vertex < vertex_neighbors.size(); ++vertex) {

            //        int rank = vertex_neighbors[vertex];
            //        std::cout << "My Rank: " << myRank  
            //                  << "vertex: " << vertex 
            //                  << "neighbor rank: " << rank << std::endl;
            //}
        }
        Ippl::Comm->barrier();
    }

    // auto& domains = layout.getHostLocalDomains();

    // for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {

    //     if (rank == Ippl::Comm->rank()) {
    //         auto& faces = layout.getFaceNeighbors();
    //         auto& edges = layout.getEdgeNeighbors();
    //         auto& vertices = layout.getVertexNeighbors();

    //         int nFaces = 0, nEdges = 0, nVertices = 0;
    //         for (size_t i = 0; i < faces.size(); ++i) {
    //             nFaces += faces[i].size();
    //         }

    //         for (size_t i = 0; i < edges.size(); ++i) {
    //             nEdges += edges[i].size();
    //         }

    //         for (size_t i = 0; i < vertices.size(); ++i) {
    //             nVertices += (vertices[i] > -1) ? 1: 0;
    //         }


    //         std::cout << "rank " << rank << ": " << std::endl
    //                   << " - domain:   " << domains[rank] << std::endl
    //                   << " - faces:    " << nFaces << std::endl
    //                   << " - edges:    " << nEdges << std::endl
    //                   << " - vertices: " << nVertices << std::endl
    //                   << "--------------------------------------" << std::endl;
    //     }
    //     Ippl::Comm->barrier();
    // }


    //field = field * 10.0;
    typename field_type::view_type& view = field.getView();
    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
    const int nghost = field.getNghost();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    Kokkos::parallel_for("Assign field", 
                          mdrange_type({nghost, nghost, nghost},
                                       {view.extent(0) - nghost,
                                        view.extent(1) - nghost,
                                        view.extent(2) - nghost}),
                          KOKKOS_LAMBDA(const int i, 
                                        const int j, 
                                        const int k)
                          {
                            //local to global index conversion
                            const size_t ig = i + lDom[0].first() - nghost;
                            const size_t jg = j + lDom[1].first() - nghost;
                            const size_t kg = k + lDom[2].first() - nghost;
                            double x = (ig + 0.5) * hx[0] + origin[0];
                            double y = (jg + 0.5) * hx[1] + origin[1];
                            double z = (kg + 0.5) * hx[2] + origin[2];

                            view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
                            //view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
                            //view(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                            //view_exact(i, j, k) = -3.0 * pi * pi * sin(pi * x) * sin(pi * y) * sin(pi * z);
                          });

    int nsteps = 1;

    for (int nt=0; nt < nsteps; ++nt) {

        //static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        //IpplTimings::startTimer(fillHaloTimer);
        //field.accumulateHalo();
        //Ippl::Comm->barrier();
        field.fillHalo();
        //Ippl::Comm->barrier();
        //IpplTimings::stopTimer(fillHaloTimer);
        msg << "Update: " << nt+1 << endl;
    }

    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == Ippl::Comm->rank()) {
            std::ofstream out("field_nRanks_" + std::to_string(nRanks) + "_rank_" + std::to_string(rank) + ".dat", std::ios::out);
            field.write(out);
            out.close();
        }
        Ippl::Comm->barrier();
    }

    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    return 0;
}
