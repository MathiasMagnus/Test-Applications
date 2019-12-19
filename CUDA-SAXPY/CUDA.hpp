#pragma once

// CUDA includes
#include <cuda_runtime.h>

// Standard C++ includes
#include <string>
#include <memory>
#include <type_traits>
#include <chrono>


namespace cuda
{
    class event;
    class stream;
    class error;

    class error
    {
    public:

        error() = default;
        error(const error&) = default;
        error(error&&) = default;
        ~error() = default;

        error(const cudaError_t);

        static std::string cudaError_to_string(const cudaError_t);

        const char* what();

    private:

        std::string m_message;
    };

    int get_device_count();

    cudaDeviceProp get_device_properties(int);

    void set_device(int);

    class event
    {
    public:

        event();

        cudaEvent_t get();

    private:

        static void deleter(cudaEvent_t* p);

        std::shared_ptr<cudaEvent_t> m_event;
    };

    class stream
    {
    public:

        stream();

        cudaStream_t get();

        void synchronize();

        template<typename T>
        std::pair<event, event> record(T&& t)
        {
            std::pair<event, event> result;

            cudaError_t err = cudaEventRecord(result.first.get(), *m_stream);

            if (err != cudaSuccess)
                throw cuda::error{ err };

            t();

            err = cudaEventRecord(result.second.get(), *m_stream);

            if (err != cudaSuccess)
                throw cuda::error{ err };

            return result;
        }

        void wait(event ev);

    private:

        static void deleter(cudaStream_t* p);

        std::shared_ptr<cudaStream_t> m_stream;
    };

    template <typename T>
    class array
    {
    public:

        array(std::size_t count) : m_dev_ptr{ new T* }
        {
            static_assert(std::is_trivially_constructible<T>::value, "cuda::array can only hold types satisfying TriviallyConstructible");

            cudaError_t err = cudaMalloc(m_dev_ptr.get(), count * sizeof(T));

            if (err != cudaSuccess)
                throw cuda::error{ err };
        }

        T* data()
        {
            return *m_dev_ptr;
        }

    private:

        static void deleter(T** p)
        {
            cudaFree(*p);
        }

        std::shared_ptr<T*> m_dev_ptr;
    };

    template<typename T>
    void copy(void* dst, array<T> src, size_t count)
    {
        cudaError_t err = cudaMemcpy(dst, src.data(), count * sizeof(T), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    template<typename T>
    void copy(array<T> dst, const void* src, size_t count)
    {
        cudaError_t err = cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    template<typename T>
    void copy(stream s, void* dst, array<T> src, size_t count)
    {
        cudaError_t err = cudaMemcpyAsync(dst, src.data(), count * sizeof(T), cudaMemcpyDeviceToHost, s.get());

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    template<typename T>
    void copy(stream s, array<T> dst, const void* src, size_t count)
    {
        cudaError_t err = cudaMemcpyAsync(dst.data(), src, count * sizeof(T), cudaMemcpyHostToDevice, s.get());

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    std::chrono::duration<float, std::chrono::milliseconds::period> get_duration(std::pair<event, event> recording);
}
