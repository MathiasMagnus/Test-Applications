// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <typeinfo>


// Kernel name is complete type to be able to obtain it's name via typeid()
namespace kernels { class SYCL_KernelFunctorQuery; }

struct obscenely_large_object
{
    cl::sycl::vec<double, 16> m_data;
};

int main()
{
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
                catch (cl::sycl::exception e)
                {
                    std::cerr << e.what() << std::endl;
                    std::exit(e.get_cl_code());
                }
            }
        };
        
        cl::sycl::context ctx{ dev, async_error_handler };

        std::vector< obscenely_large_object> vec(1);

        // Buffers exist only to prevent consant folding and allowing proper kernel invocation
        cl::sycl::buffer<double> buf_res{ cl::sycl::range<1>{ 1 } };//,
                                          //cl::sycl::property::buffer::context_bound{ ctx } };
        cl::sycl::buffer<obscenely_large_object> buf_obj{ vec.begin(), vec.end() };//,
                                                          //cl::sycl::property::buffer::context_bound { ctx } };

        cl::sycl::program prog{ ctx };
        prog.build_with_kernel_type<kernels::SYCL_KernelFunctorQuery>();
        cl::sycl::kernel krn{ prog.get_kernel<kernels::SYCL_KernelFunctorQuery>() };

        std::cout << "Maximum work-group size for device " <<
            dev.get_info<cl::sycl::info::device::name>() << ": " <<
            dev.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl;
        // Maximum WGS for given kernel on given device. (See: sycl-1.2.1.pdf: p.177, table 4.85)
        std::cout << "Maximum work-group size for " <<
            "kernels::SYCL_KernelFunctorQuery" << " on device " <<
            dev.get_info<cl::sycl::info::device::name>() << ": " <<
            krn.get_work_group_info<cl::sycl::info::kernel_work_group::work_group_size>(dev) << std::endl;

        cl::sycl::queue queue{ dev };

        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto res = buf_res.get_access<cl::sycl::access::mode::discard_write>(cgh);
            auto obj = buf_obj.get_access<cl::sycl::access::mode::read>(cgh);

            cgh.single_task<kernels::SYCL_KernelFunctorQuery>([=]()
            {
                obscenely_large_object o = obj[0];

                res[0] = o.m_data.s0() +
                         o.m_data.s1() +
                         o.m_data.s2() +
                         o.m_data.s3() +
                         o.m_data.s4() +
                         o.m_data.s5() +
                         o.m_data.s6() +
                         o.m_data.s7() +
                         o.m_data.s8() +
                         o.m_data.s9() +
                         o.m_data.sA() +
                         o.m_data.sB() +
                         o.m_data.sC() +
                         o.m_data.sD() +
                         o.m_data.sE();
            });
        });
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
