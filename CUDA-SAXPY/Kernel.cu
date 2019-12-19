#include <Kernel.hpp>


__global__
void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n) y[i] = a * x[i] + y[i];
}

std::pair<cuda::event, cuda::event> saxpy(cuda::stream stream, std::size_t N, float a, cuda::array<float> x, cuda::array<float> y)
{
    return stream.record([&]()
    {
        saxpy<<<N, 256, 0, stream.get()>>>((int)N, a, x.data(), y.data());
    });
}
