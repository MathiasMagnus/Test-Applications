// SYCL include
#include <CL/sycl.hpp>

namespace impl
{
    namespace reduce
    {
        template <typename KernelName>
        std::size_t device_max_wgs_for_kernel(cl::sycl::queue queue)
        {
            cl::sycl::program prog{ queue.get_info<cl::sycl::info::queue::context>() };
            prog.build_with_kernel_type<KernelName>();
            cl::sycl::kernel krn{ prog.get_kernel<KernelName>() };

            return krn.get_work_group_info<cl::sycl::info::kernel_work_group::work_group_size>(queue.get_info<cl::sycl::info::queue::device>());
        }

        template <typename F, typename T, typename P>
        void in_place_reduce(cl::sycl::h_item<1> i,
                             P first,
                             P last,
                             P to,
                             T init,
                             F f)
        {

        }

        /// <summary>Performs parallel reduction on the range [<c>from</c>;<c>to</c>) using the binary operator <c>F</c>.</summary>
        /// <precondition><c>local % grp.get_group_range().get(0) == 0</c></precondition>
        ///
        template <typename F, typename T>
        void in_place_reduce(cl::sycl::group<1> grp,
                             cl::sycl::local_ptr<T> first,
                             cl::sycl::local_ptr<T> last,
                             cl::sycl::local_ptr<T> to,
                             T init,
                             F f)
        {
            auto range = [=](std::size_t i) -> typename cl::sycl::local_ptr<T>::reference_t { return i < last - first ? first[i] : init; };

            for (; last != first + grp.get_group_range().get(0) ; last -= grp.get_group_range().get(0))
            {
                grp.parallel_for_work_item([=](cl::sycl::h_item<1> i)
                {
                        range(i.get_local_id()) = f(range(i.get_local_id()), first[i.get_local_id()]);
                }
            }

            for (auto I = grp.get_group_range().get(0) / 2; I > 1; I /= 2) grp.parallel_for_work_item([=](cl::sycl::h_item<1> i)
            {
                if (i.get_local_id().get(0) < I)
                    local[i] = f(local[i], local[I + i]);
            });

            *to = local[0];
        }

        template <typename F, typename T>
        void in_place_reduce(cl::sycl::queue queue,
                             cl::sycl::global_ptr<T> first,
                             cl::sycl::global_ptr<T> last,
                             cl::sycl::global_ptr<T> to,
                             T init,
                             F f)
        {
            queue.submit([&]())
        }
    }
}