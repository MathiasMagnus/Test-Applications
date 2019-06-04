#pragma once

// OpenCL behavioral defines
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

// C++ Standard includes
#include <string>

static std::string kernels_path{ "${Path_KRNS}" };
