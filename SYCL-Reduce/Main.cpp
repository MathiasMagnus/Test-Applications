// SYCL include
#include <CL/sycl.hpp>

#include "Reduce.hpp"

// Standard C++ includes
#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>      // std::iota


namespace kernels { class SYCL_Reduce; }


/// <summary>Performs a reduction operation on the provided dataset in a non-destructive manner. Result is written to <c>result</c>.</summary>
///
template <typename KernelName,
          typename ZeroElem,
          typename F,
          typename SourceType,
          typename ResultType,
          typename... Placeholders>
auto reduce(cl::sycl::queue queue,
            ZeroElem zero,
            F f,
            cl::sycl::buffer<SourceType> source,
            cl::sycl::buffer<ResultType> result,
            cl::sycl::range<1> work_group_size)
{
    auto device_max_wgs_for_kernel = [&]()
    {
        cl::sycl::program prog{ queue.get_info<cl::sycl::info::queue::context>() };
        prog.build_with_kernel_type<KernelName>();
        cl::sycl::kernel krn{ prog.get_kernel<KernelName>() };
        
        return krn.get_work_group_info<cl::sycl::info::kernel_work_group::work_group_size>(queue.get_info<cl::sycl::info::queue::device>());
    }();

    auto reduction_step = [=](auto from,
                              auto to,
                              std::size_t l)
    {
        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto local = cl::sycl::accessor<SourceType, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>{ cl::sycl::range<1>{ l }, cgh };
            auto src = from.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>( cgh );
            auto dst = to.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>( cgh );

            cgh.parallel_for_work_group<KernelName>(cl::sycl::range<1>{ gws }, cl::sycl::range<1>{ l }, [=](cl::sycl::group<1> grp)
            {
                grp.async_work_group_copy(src.get_pointer() + grp.get(0) * local.get_range(0),
                                          local.get_range().size(),
                                          local.get_pointer()).wait();

                grp.parallel_for_work_item([=](cl::sycl::h_item<1> i)
                {
                    local[i] = f(local[i], zero);
                });

                for (std::size_t I = grp.get_group_range().get(0) / 2 ; I > 1 ; I /= 2) grp.parallel_for_work_item([=](cl::sycl::h_item<1> i)
                {
                    if (i.get_local_id().get(0) < I)
                        local[i] = f(local[i], local[I + i]);
                });

                grp.async_work_group_copy(local.get_pointer(),
                                          1,
                                          dst.get_pointer() + grp.get(0)).wait();
            });
        });
    };

    if (source.get_range().size() <= work_group_size) // Single-pass reduction
    {
        reduction_step(source, result, work_group_size);
    }
    else if (source.get_range().size() <= (work_group_size * work_group_size)) // Two-pass reduction
    {
        cl::sycl::buffer<SourceType> temp{ source.get_range() / work_group_size };

        reduction_step(source, temp, work_group_size);
        reduction_step(temp, result, work_group_size);
    }
    else // Multi-pass reduction
    {
        cl::sycl::buffer<SourceType> temp{ source.get_range() / work_group_size +
                                           source.get_range() / (work_group_size * work_group_size) };
        cl::sycl::buffer<SourceType> temp_sub1{ temp, 0, source.get_range() / work_group_size };
        cl::sycl::buffer<SourceType> temp_sub2{ temp, (source.get_range() / work_group_size).get(0), source.get_range() / (work_group_size * work_group_size) };

        reduction_step(source, temp_sub1, work_group_size);

        for (std::size_t length = source.get_range() / work_group_size; length > work_group_size; length /= work_group_size)
        {
            reduction_step(temp_sub1,
                           length <= work_group_size ? result : temp_sub2, // When ran for the last time, write to 'result' instead of 'temp'
                           work_group_size,
                           length);
            std::swap(temp_sub1, temp_sub2);
        }
    }
}

int main()
{
    // Sample params
    const std::size_t length = 4096u;

    try
    {
        std::cout << "SYCL runtime using cl::sycl::default_selector..." << std::endl;

        cl::sycl::device dev{ cl::sycl::default_selector{} };

        std::cout << "Selected " <<
            dev.get_info<cl::sycl::info::device::name>() <<
            " on platform " <<
            dev.get_info<cl::sycl::info::device::platform>().get_info<cl::sycl::info::platform::name>() <<
            std::endl;

        auto async_error_handler = [](cl::sycl::exception_list errors)
        {
            for (auto error : errors)
            {
                try { std::rethrow_exception(error); }
                catch (cl::sycl::runtime_error e)
                {
                    std::cerr << e.what() << std::endl;
                    std::cerr << "Triggered in " << e.get_file_name() << ":" << e.get_line_number() << std::endl;
                    std::exit(e.get_cl_code());
                }
                catch (cl::sycl::exception e)
                {
                    std::cerr << e.what() << std::endl;
                    std::exit(e.get_cl_code());
                }
            }
        }; 
        
        cl::sycl::context ctx{ dev, async_error_handler };

        cl::sycl::queue queue{ dev };

        cl::sycl::buffer<std::uint32_t> iota_buf{ cl::sycl::range<1>{ length }, cl::sycl::property::buffer::context_bound{ ctx } };
        cl::sycl::buffer<std::uint32_t> max_buf{ cl::sycl::range<1>{ 1 }, cl::sycl::property::buffer::context_bound { ctx } };

        // Initialize buffer
        {
            auto access = iota_buf.get_access<cl::sycl::access::mode::write>();

            std::iota(access.get_pointer(), access.get_pointer() + access.get_count(), 1);
        }

        reduce<SYCL_Reduce>(queue,
                            std::numeric_limits<std::uint32_t>::max(),
                            cl::sycl::min<std::uint32_t>,
                            iota_buf,
                            max_buf,
                            dev.get_info<cl::sycl::info::device::max_work_group_size>());

        // Verify
        //
        // NOTE: host access implicitly synchronizes, meaning all operations pending on the
        //       buffer object will complete.
        //
        // SYCL 1.2:   Queue DTOR is a synchronization point, but here we use host accessor as
        //             as a sync point. (See: sycl-1.2.pdf: p.78, section 3.4.6)
        //
        // SYCL 1.2.1: Queue DTOR is no longer a synchronization point! (For a list of implicit
        //             sync points, and rationale for this change, see:
        //             sycl-1.2.1.pdf: p.30, section 3.6.5.1()
        {
            auto access = max_buf.get_access<cl::sycl::access::mode::read>();

            if (access[0] != length)
                throw std::runtime_error{ "Wrong result computed in kernel." };
        }

        std::cout << "Result verification passed!" << std::endl;
    }
    catch (cl::sycl::exception e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(e.get_cl_code());
    }
    catch (std::exception e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    return 0;
}
