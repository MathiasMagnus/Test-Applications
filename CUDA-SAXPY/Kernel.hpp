#pragma once

#include <CUDA.hpp>


std::pair<cuda::event, cuda::event> saxpy(cuda::stream stream, std::size_t N, float a, cuda::array<float> x, cuda::array<float> y);
