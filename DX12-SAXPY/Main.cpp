#include <Options.hpp>
#include <DX12.hpp>

// Windows includes
#include <dxgi.h>

// DirectX 12 includes
#include <d3d12.h>
#include <d3dx12.h>

// C++/WinRT includes
#include <winrt/Windows.Graphics.h>

// Standard C++ includes
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <valarray>
#include <random>
#include <cstddef>
#include <fstream>
#include <filesystem>


template <D3D_FEATURE_LEVEL FL>
winrt::com_ptr<ID3D12Device> adapter_to_device(const winrt::com_ptr<IDXGIAdapter>& adapter)
{
    winrt::com_ptr<ID3D12Device> tmp;

    D3D12CreateDevice(adapter.as<IUnknown>().get(), FL, _uuidof(ID3D12Device), tmp.put_void());

    return tmp;
}

#include <tuple>

int main(int argc, char* argv[])
{
    try
    {
        const std::string banner = "DX12-SAXPY sample";
        const cli::options opts = cli::parse(argc, argv, banner);

        if (!opts.quiet) std::cout << banner << std::endl << std::endl;

        winrt::com_ptr<ID3D12Debug> debug_layer;
        if (opts.debug)
        {
            winrt::check_hresult(D3D12GetDebugInterface(_uuidof(ID3D12Debug), debug_layer.put_void()));
            debug_layer->EnableDebugLayer();
        }

        // Enumerate adapters
        std::vector<winrt::com_ptr<IDXGIAdapter>> adapters(
            dx12::dxgi_iterator<IDXGIAdapter>{ winrt::com_ptr<dx12::factory_t<IDXGIAdapter>>{} },
            dx12::dxgi_iterator<IDXGIAdapter>{}
        ); 
        
        std::cout << "Found adapters:\n" << std::endl;
        for (const auto& adapter : adapters)
        {
            DXGI_ADAPTER_DESC desc;
            winrt::check_hresult(adapter->GetDesc(&desc));

            std::wcout << L"\t" << std::wstring_view{ desc.Description } << std::endl;
        }

        // Create devices
        std::vector<winrt::com_ptr<ID3D12Device>> devices;

        std::transform(adapters.cbegin(), adapters.cend(),
                       std::back_inserter(devices),
                       adapter_to_device<D3D_FEATURE_LEVEL_12_0>);
/*
    devices.erase(devices.begin(),
                  std::remove_if(devices.begin(), devices.end(),
                                 [](const winrt::com_ptr<ID3D12Device>& device)
    {
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_12_0;
        D3D12_FEATURE_DATA_FEATURE_LEVELS info;
        info.NumFeatureLevels = 1;
        info.pFeatureLevelsRequested = &fl;
        winrt::check_hresult(device->CheckFeatureSupport(D3D12_FEATURE::D3D12_FEATURE_FEATURE_LEVELS,
                                                         &info,
                                                         sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS)));

        return info.MaxSupportedFeatureLevel >= D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_12_0;
    }));
*/
        // Select device
        auto device = devices.at(opts.dev_id);

        // Compile shader (IILE)
        auto binary = [](std::filesystem::path path)
        {
            std::vector<std::byte> result(std::filesystem::file_size(path));
            std::ifstream fs{ path, std::ios::binary };
            fs.read(reinterpret_cast<char*>(result.data()), result.size());

            return result;
        }("./SAXPY.cso");

        // Create compute root signature
        std::array<D3D12_ROOT_PARAMETER1, 3> root_params;

        root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        root_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params[0].Constants = D3D12_ROOT_CONSTANTS{ 0, 0, 1 };

        root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        root_params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params[1].Descriptor = D3D12_ROOT_DESCRIPTOR1{1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};

        root_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        root_params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params[2].Descriptor = D3D12_ROOT_DESCRIPTOR1{2, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
        compute_root_signature_desc.Desc_1_1.NumParameters = 3;
        compute_root_signature_desc.Desc_1_1.pParameters = root_params.data();
        compute_root_signature_desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        compute_root_signature_desc.Desc_1_1.NumStaticSamplers = 0;
        compute_root_signature_desc.Desc_1_1.pStaticSamplers = nullptr;
        compute_root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;

        winrt::com_ptr<ID3D12RootSignature> root_signature;
        winrt::com_ptr<ID3DBlob> root_signature_blob, error;
        winrt::check_hresult(D3DX12SerializeVersionedRootSignature(
            &compute_root_signature_desc,
            D3D_ROOT_SIGNATURE_VERSION_1_1,
            root_signature_blob.put(),
            error.put())
        );
        winrt::check_hresult(device->CreateRootSignature(
            0,
            root_signature_blob->GetBufferPointer(),
            root_signature_blob->GetBufferSize(),
            __uuidof(root_signature),
            root_signature.put_void())
        );

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc;
        pso_desc.CS.BytecodeLength = binary.size();
        pso_desc.CS.pShaderBytecode = binary.data();
        pso_desc.NodeMask = 0;
        pso_desc.pRootSignature = root_signature.get();
        pso_desc.Flags = opts.debug ? D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG : D3D12_PIPELINE_STATE_FLAG_NONE;

        winrt::com_ptr<ID3D12PipelineState> pso;
        winrt::check_hresult(device->CreateComputePipelineState(
            &pso_desc,
            __uuidof(pso),
            pso.put_void())
        );

        // Create command queue
        winrt::com_ptr<ID3D12CommandQueue> compute_queue;
        winrt::com_ptr<ID3D12CommandAllocator> compute_comm_alloc;
        winrt::com_ptr<ID3D12GraphicsCommandList> compute_comm_list;

        D3D12_COMMAND_QUEUE_DESC compute_queue_desc = {
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            0,
            D3D12_COMMAND_QUEUE_FLAG_NONE
        };
        winrt::check_hresult(device->CreateCommandQueue(&compute_queue_desc,
            __uuidof(compute_queue),
            compute_queue.put_void())
        );
        winrt::check_hresult(device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE::D3D12_COMMAND_LIST_TYPE_COMPUTE,
            __uuidof(compute_comm_alloc),
            compute_comm_alloc.put_void())
        );
        winrt::check_hresult(device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            compute_comm_alloc.get(),
            nullptr,
            __uuidof(compute_comm_list),
            compute_comm_list.put_void())
        );
        //m_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_threadFences[threadIndex])));

        // Allocate & initialize host-side arrays with random values between 0 and 100
        std::valarray<float> arr_x(opts.length),
                             arr_y(opts.length);
        float a = 2.f;

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<float>{ -100.0, 100.0 }]() mutable { return dist(engine); };

        std::generate_n(std::begin(arr_x), opts.length, prng);
        std::generate_n(std::begin(arr_y), opts.length, prng);

        // Allocate device-side resources from host arrays
        winrt::com_ptr<ID3D12Resource> res_a, res_x, res_y,
                                       res_up, res_back;

        for (auto [res, count]: { std::tuple{ res_a, (size_t)1 },
                                  std::tuple{ res_x, opts.length },
                                  std::tuple{ res_y, opts.length } })
        {
            D3D12_HEAP_PROPERTIES props;
            props.Type = D3D12_HEAP_TYPE::D3D12_HEAP_TYPE_DEFAULT;
            props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            props.CreationNodeMask = 1;
            props.VisibleNodeMask = 1;

            D3D12_RESOURCE_DESC desc;
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Alignment = 0;
            desc.Width = count * sizeof(float);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 0;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.SampleDesc.Quality = 0;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;

            // Allocate
            winrt::check_hresult(device->CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                __uuidof(res),
                res.put_void()
            ));
        }

        // Allocate transfer resources
        for (auto [res, heap, state]: { std::tuple{ res_up, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ },
                                        std::tuple{ res_back, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_SOURCE } })
        {
            D3D12_HEAP_PROPERTIES props;
            props.Type = heap;
            props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            props.CreationNodeMask = 1;
            props.VisibleNodeMask = 1;

            D3D12_RESOURCE_DESC desc;
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Alignment = 0;
            desc.Width = opts.length * sizeof(float);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 0;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.SampleDesc.Quality = 0;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;

            // Allocate
            winrt::check_hresult(device->CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                nullptr,
                __uuidof(res),
                res.put_void()
            ));
        }

        // Initialize device-side resources
        for (auto [res, tmp, count, ptr]: { std::tuple{ res_a, res_up, (size_t)1, &a },
                                            std::tuple{ res_x, res_up, opts.length, std::begin(arr_x) },   // std::begin may not be ptr
                                            std::tuple{ res_y, res_up, opts.length, std::begin(arr_y) } }) // std::begin may not be ptr
        {
            D3D12_SUBRESOURCE_DATA res_data;
            res_data.pData = ptr;
            res_data.RowPitch = count * sizeof(float);
            res_data.SlicePitch = res_data.RowPitch;

            UpdateSubresources<1>(compute_comm_list.get(), res.get(), tmp.get(), 0, 0, 1, &res_data);
            //m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(res.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));
        }

        UINT srv_a, srv_x, srv_y,
             uav_a, uav_x, uav_y;
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = res_x.get();
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            compute_comm_list->ResourceBarrier(1, &barrier);
        }

        compute_comm_list->SetPipelineState(pso.get());
        compute_comm_list->SetComputeRootSignature(m_computeRootSignature.Get());
/*
    ID3D12DescriptorHeap* ppHeaps[] = { m_srvUavHeap.Get() };
    compute_comm_list->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), srvIndex + threadIndex, m_srvUavDescriptorSize);
    CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(m_srvUavHeap->GetGPUDescriptorHandleForHeapStart(), uavIndex + threadIndex, m_srvUavDescriptorSize);

    compute_comm_list->SetComputeRootConstantBufferView(ComputeRootCBV, m_constantBufferCS->GetGPUVirtualAddress());
    compute_comm_list->SetComputeRootDescriptorTable(ComputeRootSRVTable, srvHandle);
    compute_comm_list->SetComputeRootDescriptorTable(ComputeRootUAVTable, uavHandle);

    compute_comm_list->Dispatch(static_cast<int>(ceil(ParticleCount / 128.0f)), 1, 1);

    compute_comm_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pUavResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
*/
    }
    catch (cli::error& e)
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
