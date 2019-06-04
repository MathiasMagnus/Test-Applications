// CL-CPP includes
#include <CL-Include.hpp>
#include <Options.hpp>


int main(int argc, char* argv[])
{
	try // Any error results in program termination
	{
		const std::string banner = "SYCL-SAXPY sample";
        const cli::options opts = cli::parse(argc, argv, banner);

        if (!opts.quiet) std::cout << banner << std::endl << std::endl;

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if (platforms.empty()) throw std::runtime_error{ "No OpenCL platforms found." };

		std::cout << "Found platform" << (platforms.size() > 1 ? "s:\n" : ":");
		for (const auto& platform : platforms)
			std::cout << (platforms.size() > 1 ? "\t" : " ") << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		cl::Platform plat = platforms.at(opts.plat_id);
		std::cout << "Selected platform: " << plat.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		std::vector<cl::Device> devices;
		plat.getDevices(opts.dev_type, &devices);

		std::cout << "Found device" << (devices.size() > 1 ? "s:\n" : ":");
		for (const auto& device : devices)
			std::cout << (devices.size() > 1 ? "\t" : " ") << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		cl::Device device = devices.at(opts.dev_id);
		std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		// Create context and queue
		std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((plat)()), 0 };
		cl::Context context{ devices, props.data() };

		cl::CommandQueue queue{ context, device, cl::QueueProperties::Profiling };

		// Load program source
		std::ifstream source_file{ kernels_path + "/kernel.cl" };
		if (!source_file.is_open())
			throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + (kernels_path + "/kernel.cl") };

		// Create program and kernel
		cl::Program program{ context, std::string{ std::istreambuf_iterator<char>{ source_file },
			                                       std::istreambuf_iterator<char>{} } };

		program.build({ device }, "-cl-std=CL1.1");

		auto vecAdd = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(program, "saxpy");

		// Init computation
		const std::size_t chainlength = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
																		//     promises no data is lost, silences compiler warning
		std::valarray<cl_float> vec_x(chainlength),
			                    vec_y(chainlength);
		cl_float a = 2.0;

		// Fill arrays with random values between 0 and 100
		auto prng = [engine = std::default_random_engine{},
			         distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable { return distribution(engine); };

		std::generate_n(std::begin(vec_x), chainlength, prng);
		std::generate_n(std::begin(vec_y), chainlength, prng);

		cl::Buffer buf_x{ context, std::begin(vec_x), std::end(vec_x), true },
			       buf_y{ context, std::begin(vec_x), std::end(vec_x), false };

		// Explicit (blocking) dispatch of data before launch
		cl::copy(queue, std::cbegin(vec_x), std::cend(vec_x), buf_x);
		cl::copy(queue, std::cbegin(vec_x), std::cend(vec_x), buf_y);

		// Launch kernels
		cl::Event kernel_event{ vecAdd(cl::EnqueueArgs{ queue, cl::NDRange{ chainlength } }, a, buf_x, buf_y) };

		kernel_event.wait();

		std::cout <<
			"Device (kernel) execution took: " <<
			cl::util::get_duration<CL_PROFILING_COMMAND_START,
			                       CL_PROFILING_COMMAND_END,
			                       std::chrono::microseconds>(kernel_event).count() <<
			" us." << std::endl;

		// Compute validation set on host
		auto start = std::chrono::high_resolution_clock::now();

		std::valarray<cl_float> ref = a * vec_x + vec_y;

		auto finish = std::chrono::high_resolution_clock::now();

		std::cout <<
			"Host (validation) execution took: " <<
			std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() <<
			" us." << std::endl;

		// (Blocking) fetch of results
		cl::copy(queue, buf_y, std::begin(vec_y), std::end(vec_y));

		// Validate (compute saxpy on host and match results)
		auto markers = std::mismatch(std::cbegin(vec_y), std::cend(vec_y),
			                         std::cbegin(ref), std::cend(ref));

		if (markers.first != std::cend(vec_y) ||
			markers.second != std::cend(ref)) throw std::runtime_error{ "Validation failed." };

	}
	catch (TCLAP::ArgException &e) // If cli parsing error occurs
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	catch (cl::BuildError error) // If kernel failed to build
	{
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

		for (const auto& log : error.getBuildLog())
		{
			std::cerr <<
				"\tBuild log for device: " <<
				log.first.getInfo<CL_DEVICE_NAME>() <<
				std::endl << std::endl <<
				log.second <<
				std::endl << std::endl;
		}

		std::exit(error.err());
	}
	catch (cl::Error error) // If any OpenCL error occurs
	{
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

		std::exit(error.err());
	}
	catch (std::exception error) // If STL/CRT error occurs
	{
		std::cerr << error.what() << std::endl;

		std::exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}