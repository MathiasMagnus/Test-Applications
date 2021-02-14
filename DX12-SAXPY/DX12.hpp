#pragma once

// DirectX 12 includes
#include <d3d12.h>

// C++/WinRT includes
#include <winrt/Windows.Graphics.h>

// Standard C++ includes
#include <iterator>
#include <functional>


namespace dx12
{
    template <typename Adapter>
    struct adapter_traits;

    template <> struct adapter_traits<IDXGIAdapter>
    {
        using factory_type = IDXGIFactory;
        static inline HRESULT (*factory_func)(REFIID, _COM_Outptr_ void **) = &CreateDXGIFactory;
    };
    template <> struct adapter_traits<IDXGIAdapter1>
    {
        using factory_type = IDXGIFactory1;
    };

    template <typename Adapter>
    using factory_t = typename adapter_traits<Adapter>::factory_type;

    template <typename Adapter>
    class dxgi_iterator
    {
    public:

        using factory_t = winrt::com_ptr<factory_t<Adapter>>;
        using value_type = winrt::com_ptr<Adapter>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        

        constexpr dxgi_iterator()
            : _factory{}
            , _adapter{}
            , _a{-1}
            , _end{true}
        {}
        constexpr dxgi_iterator(const dxgi_iterator&) = default;
        ~dxgi_iterator() = default;
        constexpr dxgi_iterator(factory_t factory)
            : _factory{factory}
            , _adapter{}
            , _a{0}
            , _end{false}
        {
            if (!_factory) winrt::check_hresult(std::invoke(adapter_traits<Adapter>::factory_func, __uuidof(_factory), _factory.put_void()));
            if (_factory->EnumAdapters(_a, _adapter.put()) == DXGI_ERROR_NOT_FOUND)
                _end = true;
        }

        constexpr bool operator==(const dxgi_iterator& other) const
        {
            if (_end == other._end)
                return true;
            else
                return _a == other._a;
        }

        constexpr bool operator!=(const dxgi_iterator& other) const
        {
            return !(*this == other);
        }

        constexpr reference operator*()
        {
            return _adapter;
        }

        constexpr dxgi_iterator& operator++()
        {
            _adapter = nullptr;
            if (_factory->EnumAdapters(++_a, _adapter.put()) == DXGI_ERROR_NOT_FOUND)
                _end = true;

            return *this;
        }

        constexpr dxgi_iterator operator++(int)
        {
            dxgi_iterator result = *this;

            _adapter = nullptr;
            if (_factory->EnumAdapters(++_a, _adapter.put()) == DXGI_ERROR_NOT_FOUND)
                _end = true;

            return result;
        }

    private:

        factory_t _factory;
        value_type _adapter;
        int _a;
        bool _end;
    };
}

namespace std
{
    template <>
    struct std::iterator_traits<dx12::dxgi_iterator<IDXGIAdapter>>
    {
        using difference_type = dx12::dxgi_iterator<IDXGIAdapter>::difference_type;
        using value_type = dx12::dxgi_iterator<IDXGIAdapter>::value_type;
        using pointer = void;
        using reference = dx12::dxgi_iterator<IDXGIAdapter>::reference;
        using iterator_category = dx12::dxgi_iterator<IDXGIAdapter>::iterator_category;
    };
}
