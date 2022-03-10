#pragma once

// OpenCL includes
#include <CL/opencl.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>

// C++ Standard includes
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ios>
#include <chrono>
#include <random>
#include <filesystem>

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
