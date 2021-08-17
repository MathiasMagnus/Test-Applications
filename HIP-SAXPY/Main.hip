#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

void checkError(hipError_t err, const char* name)
{
	if (err != hipSuccess)
	{
		std::cerr << name << "(" << hipGetErrorString(err) << ")" << std::endl;
		std::exit(err);
	}
}

__global__
void saxpy(float a, float* x, float* y)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    y[tid] = a * x[tid] + y[tid];
}

int main()
{
    constexpr std::size_t num_threads = 128;
    constexpr std::size_t num_blocks = 32;
    constexpr std::size_t N = num_threads * num_blocks;

    hipDeviceProp_t prop;
    hipError_t err;
    err = hipGetDeviceProperties(&prop, 0);
    checkError(err, "hipGetDeviceProperties");
    std::cout << "Device name: " << prop.name << std::endl;

    const float a = 2.f;
    std::vector<float> x(N);
    std::vector<float> y(N);

    auto prng = [engine = std::default_random_engine{},
                 distribution = std::uniform_real_distribution<float>{ -1.f, 1.f }]() mutable {
        return distribution(engine);
    };

    std::generate_n(x.begin(), N, prng);
    std::generate_n(y.begin(), N, prng);

    float* x_dev;
    float* y_dev;

    err = hipMalloc((void**)&x_dev, sizeof(float) * N); checkError(err, "hipMalloc");
    err = hipMalloc((void**)&y_dev, sizeof(float) * N); checkError(err, "hipMalloc");

    err = hipMemcpy(x_dev, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice); checkError(err, "hipMemcpy");
    err = hipMemcpy(y_dev, y.data(), y.size() * sizeof(float), hipMemcpyHostToDevice); checkError(err, "hipMemcpy");

    hipLaunchKernelGGL(saxpy, dim3(num_blocks), dim3(num_threads), 0, 0, a, x_dev, y_dev); checkError(hipGetLastError(), "hipKernelLaunchGGL");

    std::transform(x.cbegin(), x.cbegin(), y.cbegin(), x.begin(),
        [=](const float& x, const float& y){ return a * x + y; }
    );

    err = hipMemcpy(y.data(), y_dev, y.size() * sizeof(float), hipMemcpyDeviceToHost); checkError(err, "hipMemcpy");

    if (std::equal(x.cbegin(), x.cend(), y.cbegin()))
        std::cerr << "Validation failed.";
    else
        std::cout << "Validation passed.";

    err = hipFree(x_dev); checkError(err, "hipFree");
    err = hipFree(y_dev); checkError(err, "hipFree");

    return 0;
}
