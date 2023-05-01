//
// Struct ViewType
//   Kokkos::Views of different dimensions.
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
#ifndef IPPL_VIEW_TYPES_H
#define IPPL_VIEW_TYPES_H

#include <Kokkos_Core.hpp>

#include <tuple>

#include "Types/Vector.h"

#include "Utility/IpplException.h"

namespace ippl {
    /**
     * @file ViewTypes.h
     * This file defines multi-dimensional arrays to store mesh and particle attributes.
     * It provides specialized versions for 1, 2 and 3 dimensions. The file further
     * provides write functions for the different view types.
     */
    namespace detail {
        /*!
         * Recursively templated struct for defining pointers with arbitrary
         * indirection depth.
         * @tparam T data type
         * @tparam N indirection level
         */
        template <typename T, int N>
        struct NPtr {
            typedef typename NPtr<T, N - 1>::type* type;
        };

        /*!
         * Base case template specialization for a simple pointer.
         */
        template <typename T>
        struct NPtr<T, 1> {
            typedef T* type;
        };

        /*!
         * Recursively templated struct for defining tuples with arbitrary
         * length
         * @tparam Dim the length of the tuple
         * @tparam T the data type to repeat (default size_t)
         */
        template <unsigned Dim, typename T = size_t>
        struct Coords {
            // https://stackoverflow.com/a/53398815/2773311
            // https://en.cppreference.com/w/cpp/utility/declval
            typedef decltype(std::tuple_cat(
                std::declval<typename Coords<1, T>::type>(),
                std::declval<typename Coords<Dim - 1, T>::type>())) type;
        };

        template <typename T>
        struct Coords<1, T> {
            typedef std::tuple<T> type;
        };

        /*!
         * View type for an arbitrary number of dimensions.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         */
        template <typename T, unsigned Dim, class... Properties>
        struct ViewType {
            typedef Kokkos::View<typename NPtr<T, Dim>::type, Properties...> view_type;
        };

        /*!
         * Multidimensional range policies.
         */
        template <unsigned Dim, typename Tag = void>
        struct RangePolicy {
            typedef std::conditional_t<std::is_void_v<Tag>,
                                       Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>,
                                       Kokkos::MDRangePolicy<Tag, Kokkos::Rank<Dim>>>
                policy_type;
            typedef typename policy_type::array_index_type index_type;
            typedef ::ippl::Vector<index_type, Dim> index_array_type;
        };

        /*!
         * Specialized range policy for one dimension.
         */
        template <typename Tag>
        struct RangePolicy<1, Tag> {
            typedef std::conditional_t<std::is_void_v<Tag>, Kokkos::RangePolicy<>,
                                       Kokkos::RangePolicy<Tag>>
                policy_type;
            typedef typename policy_type::index_type index_type;
            typedef ::ippl::Vector<index_type, 1> index_array_type;
        };

        /*!
         * Create a range policy that spans an entire Kokkos view, excluding
         * a specifiable number of ghost cells at the extremes.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to span
         * @param shift number of ghost cells
         *
         * @return A (MD)RangePolicy that spans the desired elements of the given view
         */
        template <unsigned Dim, typename Tag = void, typename View>
        typename RangePolicy<Dim, Tag>::policy_type getRangePolicy(const View& view,
                                                                   int shift = 0) {
            using policy_type = typename RangePolicy<Dim, Tag>::policy_type;
            if constexpr (Dim == 1) {
                return policy_type(shift, view.size() - shift);
            } else {
                using index_type = typename RangePolicy<Dim, Tag>::index_type;
                Kokkos::Array<index_type, Dim> begin, end;
                for (unsigned int d = 0; d < Dim; d++) {
                    begin[d] = shift;
                    end[d]   = view.extent(d) - shift;
                }
                return policy_type(begin, end);
            }
            // Silences incorrect nvcc warning: missing return statement at end of non-void function
            throw IpplException("detail::getRangePolicy", "Unreachable state");
        }

        /*!
         * Create a range policy for an index range given in the form of arrays
         * (required because Kokkos doesn't allow the initialization of 1D range
         * policies using arrays)
         * @tparam Dim the dimension of the range
         * @tparam Tag range policy tags
         *
         * @param begin the starting indices
         * @param end the ending indices
         *
         * @return A (MD)RangePolicy spanning the given range
         */
        template <unsigned Dim, typename Tag = void>
        typename RangePolicy<Dim, Tag>::policy_type createRangePolicy(
            const Kokkos::Array<typename RangePolicy<Dim, Tag>::index_type, Dim>& begin,
            const Kokkos::Array<typename RangePolicy<Dim, Tag>::index_type, Dim>& end) {
            using policy_type = typename RangePolicy<Dim, Tag>::policy_type;
            if constexpr (Dim == 1) {
                return policy_type(begin[0], end[0]);
            } else {
                return policy_type(begin, end);
            }
            // Silences incorrect nvcc warning: missing return statement at end of non-void function
            throw IpplException("detail::getRangePolicy", "Unreachable state");
        }

        enum e_functor_type {
            FOR,
            REDUCE,
            SCAN
        };

        template <e_functor_type, typename, typename, typename>
        struct FunctorWrapper;

        /*!
         * Wrapper struct for reduction kernels
         * Source:
         * https://stackoverflow.com/questions/50713214/familiar-template-syntax-for-generic-lambdas
         * @tparam Functor functor type
         * @tparam T... index types
         * @tparam Acc accumulator data type
         * @tparam R functor return type
         */
        template <typename Functor, typename... T, typename Acc>
        struct FunctorWrapper<REDUCE, Functor, std::tuple<T...>, Acc> {
            Functor f;

            /*!
             * Inline operator forwarding to a specialized instantiation
             * of the functor's own operator()
             * @param x... the indices
             * @param res the accumulator variable
             * @return The functor's return value
             */
            KOKKOS_INLINE_FUNCTION void operator()(T... x, Acc& res) const {
                using index_type = typename RangePolicy<sizeof...(T)>::index_type;
                typename RangePolicy<sizeof...(T)>::index_array_type args = {(index_type)x...};
                f(args, res);
            }
        };

        template <typename Functor, typename... T>
        struct FunctorWrapper<FOR, Functor, std::tuple<T...>, void> {
            Functor f;

            KOKKOS_INLINE_FUNCTION void operator()(T... x) const {
                using index_type = typename RangePolicy<sizeof...(T)>::index_type;
                typename RangePolicy<sizeof...(T)>::index_array_type args = {(index_type)x...};
                f(args);
            }
        };

        /*!
         * Convenience function for wrapping a functor with the wrapper struct.
         * @tparam Dim the loop's rank
         * @tparam Acc the accumulator type
         * @tparam Functor the functor type
         * @return A wrapper containing the given functor
         */
        template <e_functor_type Type, unsigned Dim, typename Acc = void, typename Functor>
        auto functorize(const Functor& f) {
            return FunctorWrapper<Type, Functor, typename Coords<Dim>::type, Acc>{f};
        }

        template <unsigned Dim, unsigned Current = 0, typename View, typename... Args>
        static constexpr void printLoop(const View& view, std::ostream& out, Args&&... args) {
            for (size_t i = 0; i < view.extent(Dim - Current - 1); ++i) {
                if constexpr (Dim - 1 == Current) {
                    out << view(i, args...) << " ";
                } else {
                    printLoop<Dim, Current + 1>(view, out, i, args...);
                }
            }
            if (Current + 1 >= 2 || Current == 0)
                out << std::endl;
        }

        /*!
         * Writes a view to an output stream
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to write
         * @param out stream
         */
        template <typename T, unsigned Dim, class... Properties>
        void write(const typename ViewType<T, Dim, Properties...>::view_type& view,
                   std::ostream& out = std::cout) {
            using view_type = typename ViewType<T, Dim, Properties...>::view_type;
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);

            printLoop<Dim>(hview, out);
        }

        template <typename>
        struct ExtractRank;
        template <typename... T>
        struct ExtractRank<Kokkos::RangePolicy<T...>> {
            static constexpr int rank = 1;
        };
        template <typename... T>
        struct ExtractRank<Kokkos::MDRangePolicy<T...>> {
            static constexpr int rank = Kokkos::MDRangePolicy<T...>::rank;
        };
    }  // namespace detail

    template <class ExecPolicy, class FunctorType>
    void parallel_for(const std::string& name, const ExecPolicy& policy,
                      const FunctorType& functor) {
        Kokkos::parallel_for(
            name, policy,
            detail::functorize<detail::FOR, detail::ExtractRank<ExecPolicy>::rank>(functor));
    }

    template <class ExecPolicy, class FunctorType, class... ReducerArgument>
    void parallel_reduce(const std::string& name, const ExecPolicy& policy,
                         const FunctorType& functor, ReducerArgument&... reducer) {
        Kokkos::parallel_reduce(
            name, policy,
            detail::functorize<detail::REDUCE, detail::ExtractRank<ExecPolicy>::rank,
                               typename ReducerArgument::value_type...>(functor),
            reducer...);
    }

    template <class ExecPolicy, class FunctorType, class... ReducerArgument>
    void parallel_reduce(const std::string& name, const ExecPolicy& policy,
                         const FunctorType& functor, const ReducerArgument&... reducer) {
        Kokkos::parallel_reduce(
            name, policy,
            detail::functorize<detail::REDUCE, detail::ExtractRank<ExecPolicy>::rank,
                               typename ReducerArgument::value_type...>(functor),
            reducer...);
    }
}  // namespace ippl

#endif
