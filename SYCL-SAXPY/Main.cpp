#include <Options.hpp>

// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <string>
#include <algorithm>
#include <valarray>
#include <random>


namespace util
{
    template <cl::sycl::info::event_profiling From,
              cl::sycl::info::event_profiling To,
              typename Dur = std::chrono::nanoseconds>
    auto get_duration(const cl::sycl::event& ev)
    {
        using namespace std::chrono;
        using cl::sycl::info::event_profiling;
        
        return duration_cast<Dur>(nanoseconds{ ev.get_profiling_info<event_profiling::command_end>() -
                                               ev.get_profiling_info<event_profiling::command_start>() } );
    }
}

namespace kernels { class saxpy; }

int main(int argc, char* argv[])
{
    try
    {
        const std::string banner = "SYCL-SAXPY sample";
        const cli::options opts = cli::parse(argc, argv, banner);

        if (!opts.quiet) std::cout << banner << std::endl << std::endl;

        // Platform selection
        auto plats = cl::sycl::platform::get_platforms();
        
        if (plats.empty()) throw std::runtime_error{ "No OpenCL platform found." };
        if (!opts.quiet)
        {
            std::cout << "Found platform" << (plats.size() > 1 ? "s:" : ":") << std::endl;
            for (const auto plat : plats) std::cout << "\t" << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;
        }
        auto plat = plats.at(opts.plat_id);
        
        if (!opts.quiet) std::cout << "\n" << "Selected platform: " << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;
        
        // Device selection
        auto devs = plat.get_devices(opts.dev_type);
        
        if (devs.empty()) throw std::runtime_error{ "No OpenCL device of specified type found on selected platform." };
        auto dev = devs.at(opts.dev_id);
        
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
        
        cl::sycl::context ctx{ dev, [](cl::sycl::exception_list errors)
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
        } };

        auto dev_supports_profiling = dev.get_info<cl::sycl::info::device::queue_profiling>();

        cl::sycl::queue queue{ dev, dev_supports_profiling ?
                                    cl::sycl::property::queue::enable_profiling{} :
                                    cl::sycl::property_list{} };

        cl::sycl::buffer<float> buf_x{ cl::sycl::range<1>{opts.length} },
                                buf_y{ cl::sycl::range<1>{opts.length} };

        std::valarray<float> arr_x(opts.length),
                             arr_y(opts.length);
        float a = 2.f;

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<float>{ -100.0, 100.0 }]() mutable { return dist(engine); };

        std::generate_n(std::begin(arr_x), opts.length, prng);
        std::generate_n(std::begin(arr_y), opts.length, prng);

        // Initialize buffer
        {
            auto x = buf_x.get_access<cl::sycl::access::mode::write>();
            auto y = buf_y.get_access<cl::sycl::access::mode::write>();

            std::copy(std::begin(arr_x), std::end(arr_x), x.get_pointer());
            std::copy(std::begin(arr_y), std::end(arr_y), y.get_pointer());
        }

        // Compute on device
        auto event = queue.submit([&](cl::sycl::handler& cgh)
        {
            auto x = buf_x.get_access<cl::sycl::access::mode::read>(cgh);
            auto y = buf_y.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<kernels::saxpy>(x.get_range(), [=](cl::sycl::item<1> i)
            {
                y[i] = a * x[i] + y[i];
            });
        });

        event.wait_and_throw(); // May use CPU as device

        // Compute validation set on host
        auto start = std::chrono::high_resolution_clock::now();

        arr_y = a * arr_x + arr_y;

        auto finish = std::chrono::high_resolution_clock::now();

        if (!opts.quiet) std::cout <<
            "Host (validation) execution took: " <<
            std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() <<
            " us." << std::endl;

        if (!opts.quiet && dev_supports_profiling) std::cout <<
            "Device (kernel) execution took: " <<
            util::get_duration<cl::sycl::info::event_profiling::command_start,
                               cl::sycl::info::event_profiling::command_end,
                               std::chrono::microseconds>(event).count() <<
            " us." << std::endl;

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
            auto acc_y = buf_y.get_access<cl::sycl::access::mode::read>();
            auto acc_y_begin = acc_y.get_pointer(),
                 acc_y_end = acc_y.get_pointer() + acc_y.get_count();

            auto markers = std::mismatch(std::begin(arr_y), std::end(arr_y),
                                         acc_y_begin, acc_y_end);

            if (markers.first != std::end(arr_y) || markers.second != acc_y_end)
                throw std::runtime_error{ "Validation failed." };
        }

        if (!opts.quiet) std::cout << "Result verification passed!" << std::endl;
    }
    catch (cli::error& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (cl::sycl::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(e.get_cl_code());
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
