#include <Options.hpp>
#include <CUDA.hpp>
#include <Kernel.hpp>

// Standard C++ includes
#include <string>
#include <iostream>
#include <algorithm>
#include <valarray>
#include <random>
#include <iterator>
#include <valarray>


int main(int argc, char* argv[])
{
    try
    {
        const std::string banner = "CUDA-SAXPY sample";
        const cli::options opts = cli::parse(argc, argv, banner);

        if (!opts.quiet) std::cout << banner << std::endl << std::endl;

        // Device selection
        int device_count = cuda::get_device_count();
        if (device_count == 0) throw std::runtime_error{ "No CUDA device found." };
        
        if (!opts.quiet)
        {
            std::cout << "Found device" << (device_count > 1 ? "s:\n" : ":\n") << std::endl;
            std::generate_n(std::ostream_iterator<std::string>{std::cout, "\n"},
                            device_count,
                            [&, device = 0]() mutable
            {
                using namespace std::string_literals;
                return "\t"s + std::string{ cuda::get_device_properties(device++).name };
            });
        }

        if (opts.dev_id >= (size_t)device_count) throw std::runtime_error{ "No CUDA device of specified id found." };
        cuda::set_device(opts.dev_id);

        if (!opts.quiet) std::cout << "Selected device: " << cuda::get_device_properties(opts.dev_id).name << "\n" << std::endl;

        cuda::stream dispatch_stream,
                     compute_stream,
                     fetch_stream;

        cuda::array<float> arr_x{opts.length},
                           arr_y{opts.length};
        
        std::valarray<float> varr_x(opts.length),
                             varr_y(opts.length);
        float a = 2.f;

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<float>{ -100.0, 100.0 }]() mutable { return dist(engine); };

        std::generate_n(std::begin(varr_x), opts.length, prng);
        std::generate_n(std::begin(varr_y), opts.length, prng);

        // Initialize arrays
        auto dispatch = dispatch_stream.record([&]()
        {
            cuda::copy(dispatch_stream, arr_x, &(*std::begin(varr_x)), varr_x.size());
            cuda::copy(dispatch_stream, arr_y, &(*std::begin(varr_y)), varr_y.size());
        });

        compute_stream.wait(dispatch.second);

        auto compute = saxpy(compute_stream, opts.length, a, arr_x, arr_y);
        
        // Compute validation set on host
        auto start = std::chrono::high_resolution_clock::now();

        varr_y = a * varr_x + varr_y;

        auto finish = std::chrono::high_resolution_clock::now();

        compute_stream.synchronize();

        if (!opts.quiet) std::cout <<
            "Host (validation) execution took: " <<
            std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() <<
            " us." << std::endl;

        if (!opts.quiet) std::cout <<
            "Device (kernel) execution took: " <<
            std::chrono::duration<float, std::chrono::microseconds::period>{cuda::get_duration(compute)}.count() <<
            " us." << std::endl;

        // Verify
        {
            fetch_stream.wait(compute.second);
            fetch_stream.record([&]()
            {
                // Reuse varr_x as storage for device results
                cuda::copy(fetch_stream, &(*std::begin(varr_x)), arr_y, varr_x.size());
            });
            fetch_stream.synchronize();

            auto markers = std::mismatch(std::begin(varr_y), std::end(varr_y),
                                         std::begin(varr_x), std::end(varr_x));

            if (markers.first != std::end(varr_y) || markers.second != std::end(varr_x))
                throw std::runtime_error{ "Validation failed." };
        }

        if (!opts.quiet) std::cout << "Result verification passed!" << std::endl;
    }
    catch (cli::error& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (cuda::error& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
