//
// Class HaloCells
//   The guard / ghost cells of BareField.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_HALO_CELLS_H
#define IPPL_HALO_CELLS_H

#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"
#include "Types/IpplTypes.h"
#include "Communicate/Archive.h"
#include "FieldLayout/FieldLayout.h"
#include <array>

namespace ippl {
    namespace detail {
        /*!
         * Helper class to send / receive field data.
         */
        template <typename T>
        struct FieldBufferData {
            using view_type = typename detail::ViewType<T, 1>::view_type;

            void serialize(Archive<>& ar, count_type nsends) {
                ar.serialize(buffer, nsends);
            }

            void deserialize(Archive<>& ar, count_type nrecvs) {
                ar.deserialize(buffer, nrecvs);
            }

            view_type buffer;
        };

        /*!
         * This class provides the functionality to do field halo exchange.
         * @file HaloCells.h
         */
        template <typename T, unsigned Dim>
        class HaloCells
        {

        public:
            using view_type = typename detail::ViewType<T, Dim>::view_type;
            using Layout_t  = FieldLayout<Dim>;
            using bound_type = typename Layout_t::bound_type;

            enum SendOrder {
                HALO_TO_INTERNAL,
                INTERNAL_TO_HALO
            };

            HaloCells();

            /*!
             * Send halo data to internal cells. This operation uses
             * assign_plus functor to assign the data.
             * @param view the original field data
             * @param layout the field layout storing the domain decomposition
             */
            void accumulateHalo(view_type& view,
                                const Layout_t* layout);

            /*!
             * Send interal data to halo cells. This operation uses
             * assign functor to assign the data.
             * @param view the original field data
             * @param layout the field layout storing the domain decomposition
             */
            void fillHalo(view_type&, const Layout_t* layout);

            /*!
             * Pack the field data to be sent into a contiguous array.
             * @param range the bounds of the subdomain to be sent
             * @param view the original view
             * @param fd the buffer to pack into
             */
            void pack(const bound_type& range,
                      const view_type& view,
                      FieldBufferData<T>& fd,
                      count_type& nsends);

            /*!
             * Unpack the received field data and assign it.
             * @param range the bounds of the subdomain to be received
             * @param view the original view
             * @param fd the buffer to unpack from (received data)
             * @tparam Op the data assigment operator
             */
            template <typename Op>
            void unpack(const bound_type& range,
                        const view_type& view,
                        FieldBufferData<T>& fd);

            /*!
             * Operator for the unpack function.
             * This operator is used in case of INTERNAL_TO_HALO.
             */
            struct assign {
                KOKKOS_INLINE_FUNCTION
                void operator()(T& lhs, const T& rhs) const {
                    lhs = rhs;
                }
            };

            /*!
             * Operator for the unpack function.
             * This operator is used in case of HALO_TO_INTERNAL.
             */
            struct plus_assign {
                KOKKOS_INLINE_FUNCTION
                void operator()(T& lhs, const T& rhs) const {
                    lhs += rhs;
                }
            };


        private:

            /*!
             * Exchange the data of faces.
             * @param view is the original field data
             * @param layout the field layout storing the domain decomposition
             * @param order the data send orientation
             * @tparam Op the data assigment operator of the
             * unpack function call
             */
            template <class Op>
            void exchangeFaces(view_type& view,
                               const Layout_t* layout,
                               SendOrder order);

            /*!
             * Exchange the data of edges.
             * @param view is the original field data
             * @param layout the field layout storing the domain decomposition
             * @param order the data send orientation
             * @tparam Op the data assigment operator of the
             * unpack function call
             */
            template <class Op>
            void exchangeEdges(view_type& view,
                               const Layout_t* layout,
                               SendOrder order);

            /*!
             * Exchange the data of vertices.
             * @param view is the original field data
             * @param layout the field layout storing the domain decomposition
             * @param order the data send orientation
             * @tparam Op the data assigment operator of the
             * unpack function call
             */
            template <class Op>
            void exchangeVertices(view_type& view,
                                  const Layout_t* layout,
                                  SendOrder order);

            /*!
             * Extract the subview of the original data. This does not copy.
             * A subview points to the same memory.
             * @param view is the original field data
             * @param intersect the bounds of the intersection
             */
            auto makeSubview(const view_type& view,
                             const bound_type& intersect);

            FieldBufferData<T> fd_m;

            std::array<int, 6> matchface_m = { 1, 0, 3, 2, 5, 4 };
            std::array<int, 12> matchedge_m = { 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8 };
            std::array<int, 8> matchvertex_m = { 7, 6, 5, 4, 3, 2, 1, 0 };

        };
    }
}

#include "Field/HaloCells.hpp"

#endif
