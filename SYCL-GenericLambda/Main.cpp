// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <string>
#include <algorithm>


int main()
{
    // Sample params
    const std::size_t plat_index = std::numeric_limits<std::size_t>::max();
    const std::size_t dev_index = std::numeric_limits<std::size_t>::max();
    const auto dev_type = cl::sycl::info::device_type::gpu;
    const std::size_t length = 4096u;

    try
    {
        // Platform selection
        auto plats = cl::sycl::platform::get_platforms();

        if (plats.empty()) throw std::runtime_error{ "No OpenCL platform found." };

        std::cout << "Found platforms:" << std::endl;
        for (const auto plat : plats) std::cout << "\t" << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;

        auto plat = plats.at(plat_index == std::numeric_limits<std::size_t>::max() ? 0 : plat_index);

        std::cout << "\n" << "Selected platform: " << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;

        // Device selection
        auto devs = plat.get_devices(dev_type);

        if (devs.empty()) throw std::runtime_error{ "No OpenCL device of specified type found on selected platform." };

        auto dev = devs.at(dev_index == std::numeric_limits<std::size_t>::max() ? 0 : dev_index);

        std::cout << "Selected device: " << dev.get_info<cl::sycl::info::device::name>() << "\n" << std::endl;

        // Context, queue, buffer creation
        //
        // NOTE: while explicit context creation may be omitted at the developers discretion, it is
        //       deemed both instructive and useful to manually handle the context. Rationale follows
        //       excerpt from sycl-1.2.1.pdf: p.32, section 3.6.9)
        //
        // There is no global state speciﬁed to be required in SYCL implementations. This means, for example,
        // that if the user creates two queues without explicitly constructing a common context, then a SYCL
        // implementation does not have to create a shared context for the two queues. Implementations are free
        // to share or cache state globally for performance, but it is not required.
        //
        //       After getting used to writing single-device/single-queue code, when one ventures into the realm
        //       of multi-device or single-device but multi-queue (concurrent computation and data movement)
        //       code, the optional nature of this facility may cause arcane errors.

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

        cl::sycl::queue queue{ dev };

        cl::sycl::buffer<float> buf{ cl::sycl::range<1>{length} };

        // Initialize buffer
        // 
        // NOTE: host accessor lifetime provides the SYCL runtime with the necessary information
        //       about the scope of possible host access. (Map-unmap in OpenCL terms.)
        {
            auto access = buf.get_access<cl::sycl::access::mode::write>();

            std::fill_n(access.get_pointer(), access.get_count(), 1.f);
        }

        // Nested lambda to be captured by kernel
        const auto f = [](const auto x)
        {
            return [=](const auto y)
            {
                return x + y;
            };
        };

        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto v = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class SYCL_Test>(v.get_range(), [=](cl::sycl::item<1> i)
            {
                v[i] += f(1.f)(2.f);
            });
        });

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
            auto access = buf.get_access<cl::sycl::access::mode::read>();

            if (std::any_of(access.get_pointer(),
                            access.get_pointer() + access.get_count(),
                            [res = 1.f + 1.f + 2.f](const float& val) { return val != res; }))
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
