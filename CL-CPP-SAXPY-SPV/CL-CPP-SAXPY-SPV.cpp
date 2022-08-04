// OpenCL includes
#include <CL/opencl.hpp>

// C++ Standard includes
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ios>
#include <chrono>
#include <random>
#include <filesystem>
#include <execution>

namespace cl
{
	namespace util
	{
		template <cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
		auto get_duration(cl::Event& ev)
		{
			return std::chrono::duration_cast<Dur>(std::chrono::nanoseconds{ ev.getProfilingInfo<To>() - ev.getProfilingInfo<From>() });
		}
	}
}

int main(int argc, char* argv[])
{
	try // Any error results in program termination
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if (platforms.empty()) throw std::runtime_error{ "No OpenCL platforms found." };

		std::cout << "Found platform" << (platforms.size() > 1 ? "s" : "") << ":\n";
		for (const auto& platform : platforms)
			std::cout << "\t" << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		cl::Platform platform = platforms.at(argc > 1 ? std::atoi(argv[1]) : 0);
		std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		std::cout << "Found device" << (devices.size() > 1 ? "s" : "") << ":\n";
		for (const auto& device : devices)
			std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		cl::Device device = devices.at(argc > 2 ? std::atoi(argv[2]) : 0);
		std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		if (platform.getInfo<CL_PLATFORM_VERSION>().find("OpenCL 2.1") != std::string::npos ||
		    platform.getInfo<CL_PLATFORM_VERSION>().find("OpenCL 2.2") != std::string::npos)
		{
			if (device.getInfo<CL_DEVICE_IL_VERSION>().empty())
				throw std::runtime_error{ "Selected device doesn't support any IL formats." };

			std::cout << "Supported IL formats: " << device.getInfo<CL_DEVICE_IL_VERSION>() << std::endl;
		}
		else
		{
			if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_il_program") == std::string::npos)
				throw std::runtime_error{ "Selected device doesn't support any IL formats." };

			std::cout << "Supported IL formats: " << device.getInfo<CL_DEVICE_IL_VERSION_KHR>() << std::endl;
		}

		// Create context and queue
		std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((platform)()), 0 };
		cl::Context context{ devices, props.data() };

		cl::CommandQueue queue{ context, device, cl::QueueProperties::Profiling };

		// Load program source
		std::ifstream source_file{ std::filesystem::canonical(argv[0]).parent_path().append("saxpy.spv"), std::ios::binary };
		if (!source_file.is_open())
			throw std::runtime_error{ std::string{ "Cannot open kernel intermediate." } };

		// Create program and kernel
		cl::Program program(
			context,
			std::vector<char>(
				std::istreambuf_iterator<char>{ source_file },
				std::istreambuf_iterator<char>{}
			)
		);

		program.build({ device });

		auto saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(program, "saxpy");

		// Init computation
		const std::size_t length = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
		                                                           //     promises no data is lost, silences compiler warning
		std::vector<cl_float> vec_x(length),
		                      vec_y(length);
		cl_float a = 2.0;

		// Fill arrays with random values between 0 and 100
		auto prng = [engine = std::default_random_engine{},
		             distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable { return distribution(engine); };

		std::generate_n(std::begin(vec_x), length, prng);
		std::generate_n(std::begin(vec_y), length, prng);

		cl::Buffer buf_x{ queue, std::begin(vec_x), std::end(vec_x), true },
		           buf_y{ queue, std::begin(vec_y), std::end(vec_y), false };

		// Launch kernels
		cl::Event kernel_event{ saxpy(cl::EnqueueArgs{ queue, cl::NDRange{ length }, cl::NullRange }, a, buf_x, buf_y) };
		kernel_event.wait();

		// Compute validation set on host
		auto start = std::chrono::high_resolution_clock::now();

		std::transform(std::execution::par_unseq,
		               vec_x.cbegin(), vec_x.cend(),
		               vec_y.cbegin(),
		               vec_y.begin(),
		               [=](const cl_float& x, const cl_float& y)
		{
			return a * x + y;
		});

		auto finish = std::chrono::high_resolution_clock::now();

		std::cout <<
			"Host (validation) execution took: " <<
			std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() <<
			" us." << std::endl;

		std::cout <<
			"Device (kernel) execution took: " <<
			cl::util::get_duration<CL_PROFILING_COMMAND_START,
			                       CL_PROFILING_COMMAND_END,
			                       std::chrono::microseconds>(kernel_event).count() <<
			" us." << std::endl;

		// (Blocking) fetch of results (reuse storage of vec_x)
		cl::copy(queue, buf_y, std::begin(vec_x), std::end(vec_x));

		// Validate (compute saxpy on host and match results)
		auto markers = std::mismatch(std::begin(vec_x), std::end(vec_x),
		                             std::begin(vec_y), std::end(vec_y));

		if (markers.first != std::end(vec_x) ||
		    markers.second != std::end(vec_y)) throw std::runtime_error{ "Validation failed." };
		else
			std::cout << "Validation passed." << std::endl;

	}
	catch (cl::BuildError& error) // If kernel failed to build
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
	catch (cl::Error& error) // If any OpenCL error occurs
	{
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		std::exit(error.err());
	}
	catch (std::exception& error) // If STL/CRT error occurs
	{
		std::cerr << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}