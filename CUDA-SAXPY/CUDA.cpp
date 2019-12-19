#include <CUDA.hpp>

namespace cuda
{
    error::error(const cudaError_t err) : m_message{cudaError_to_string(err)} {}

    std::string error::cudaError_to_string(const cudaError_t err)
    {
        switch (err)
        {
            case cudaSuccess:
                return "cudaSuccess";
            case cudaErrorMissingConfiguration:
                return "cudaErrorMissingConfiguration";
            case cudaErrorMemoryAllocation:
                return "MemoryAllocation";
            case cudaErrorInitializationError:
                return "InitializationError";
            case cudaErrorLaunchFailure:
                return "LaunchFailure";
            case cudaErrorPriorLaunchFailure:
                return "PriorLaunchFailure";
            case cudaErrorLaunchTimeout:
                return "LaunchTimeout";
            case cudaErrorLaunchOutOfResources:
                return "LaunchOutOfResources";
            case cudaErrorInvalidDeviceFunction:
                return "InvalidDeviceFunction";
            case cudaErrorInvalidConfiguration:
                return "InvalidConfiguration";
            case cudaErrorInvalidDevice:
                return "InvalidDevice";
            case cudaErrorInvalidValue:
                return "InvalidValue";
            case cudaErrorInvalidPitchValue:
                return "InvalidPitchValue";
            case cudaErrorInvalidSymbol:
                return "InvalidSymbol";
            case cudaErrorMapBufferObjectFailed:
                return "MapBufferObjectFailed";
            case cudaErrorUnmapBufferObjectFailed:
                return "UnmapBufferObjectFailed";
            case cudaErrorInvalidHostPointer:
                return "InvalidHostPointer";
            case cudaErrorInvalidDevicePointer:
                return "InvalidDevicePointer";
            case cudaErrorInvalidTexture:
                return "InvalidTexture";
            case cudaErrorInvalidTextureBinding:
                return "InvalidTextureBinding";
            case cudaErrorInvalidChannelDescriptor:
                return "InvalidChannelDescriptor";
            case cudaErrorInvalidMemcpyDirection:
                return "InvalidMemcpyDirection";
            case cudaErrorAddressOfConstant:
                return "AddressOfConstant";
            case cudaErrorTextureFetchFailed:
                return "TextureFetchFailed";
            case cudaErrorTextureNotBound:
                return "TextureNotBound";
            case cudaErrorSynchronizationError:
                return "SynchronizationError";
            case cudaErrorInvalidFilterSetting:
                return "InvalidFilterSetting";
            case cudaErrorInvalidNormSetting:
                return "InvalidNormSetting";
            case cudaErrorMixedDeviceExecution:
                return "MixedDeviceExecution";
            case cudaErrorCudartUnloading:
                return "CudartUnloading";
            case cudaErrorUnknown:
                return "Unknown";
            case cudaErrorNotYetImplemented:
                return "NotYetImplemented";
            case cudaErrorMemoryValueTooLarge:
                return "MemoryValueTooLarge";
            case cudaErrorInvalidResourceHandle:
                return "InvalidResourceHandle";
            case cudaErrorNotReady:
                return "NotReady";
            case cudaErrorInsufficientDriver:
                return "InsufficientDriver";
            case cudaErrorSetOnActiveProcess:
                return "SetOnActiveProcess";
            case cudaErrorInvalidSurface:
                return "InvalidSurface";
            case cudaErrorNoDevice:
                return "NoDevice";
            case cudaErrorECCUncorrectable:
                return "ECCUncorrectable";
            case cudaErrorSharedObjectSymbolNotFound:
                return "SharedObjectSymbolNotFound";
            case cudaErrorSharedObjectInitFailed:
                return "SharedObjectInitFailed";
            case cudaErrorUnsupportedLimit:
                return "UnsupportedLimit";
            case cudaErrorDuplicateVariableName:
                return "DuplicateVariableName";
            case cudaErrorDuplicateTextureName:
                return "DuplicateTextureName";
            case cudaErrorDuplicateSurfaceName:
                return "DuplicateSurfaceName";
            case cudaErrorDevicesUnavailable:
                return "DevicesUnavailable";
            case cudaErrorInvalidKernelImage:
                return "InvalidKernelImage";
            case cudaErrorNoKernelImageForDevice:
                return "NoKernelImageForDevice";
            case cudaErrorIncompatibleDriverContext:
                return "IncompatibleDriverContext";
            case cudaErrorPeerAccessAlreadyEnabled:
                return "PeerAccessAlreadyEnabled";
            case cudaErrorPeerAccessNotEnabled:
                return "PeerAccessNotEnabled";
            case cudaErrorDeviceAlreadyInUse:
                return "DeviceAlreadyInUse";
            case cudaErrorProfilerDisabled:
                return "ProfilerDisabled";
            case cudaErrorProfilerNotInitialized:
                return "ProfilerNotInitialized";
            case cudaErrorProfilerAlreadyStarted:
                return "ProfilerAlreadyStarted";
            case cudaErrorProfilerAlreadyStopped:
                return "ProfilerAlreadyStopped";
            case cudaErrorAssert:
                return "Assert";
            case cudaErrorTooManyPeers:
                return "TooManyPeers";
            case cudaErrorHostMemoryAlreadyRegistered:
                return "HostMemoryAlreadyRegistered";
            case cudaErrorHostMemoryNotRegistered:
                return "HostMemoryNotRegistered";
            case cudaErrorOperatingSystem:
                return "OperatingSystem";
            case cudaErrorPeerAccessUnsupported:
                return "PeerAccessUnsupported";
            case cudaErrorLaunchMaxDepthExceeded:
                return "LaunchMaxDepthExceeded";
            case cudaErrorLaunchFileScopedTex:
                return "LaunchFileScopedTex";
            case cudaErrorLaunchFileScopedSurf:
                return "LaunchFileScopedSurf";
            case cudaErrorSyncDepthExceeded:
                return "SyncDepthExceeded";
            case cudaErrorLaunchPendingCountExceeded:
                return "LaunchPendingCountExceeded";
            case cudaErrorNotPermitted:
                return "NotPermitted";
            case cudaErrorNotSupported:
                return "NotSupported";
            case cudaErrorHardwareStackError:
                return "HardwareStackError";
            case cudaErrorIllegalInstruction:
                return "IllegalInstruction";
            case cudaErrorMisalignedAddress:
                return "MisalignedAddress";
            case cudaErrorInvalidAddressSpace:
                return "InvalidAddressSpace";
            case cudaErrorInvalidPc:
                return "InvalidPc";
            case cudaErrorIllegalAddress:
                return "IllegalAddress";
            case cudaErrorInvalidPtx:
                return "InvalidPtx";
            case cudaErrorInvalidGraphicsContext:
                return "InvalidGraphicsContext";
            case cudaErrorNvlinkUncorrectable:
                return "NvlinkUncorrectable";
            case cudaErrorJitCompilerNotFound:
                return "JitCompilerNotFound";
            case cudaErrorCooperativeLaunchTooLarge:
                return "CooperativeLaunchTooLarge";
            case cudaErrorStartupFailure:
                return "StartupFailure";
            case cudaErrorApiFailureBase:
                return "ApiFailureBase";
            default:
                return "UnkownError";
        }
    }

    const char* error::what() { return m_message.c_str(); }

    int get_device_count()
    {
        int result = 0;
        cudaError_t err = cudaGetDeviceCount(&result);

        if (err != cudaSuccess)
            throw cuda::error{ err };
        else
            return result;
    }

    cudaDeviceProp get_device_properties(int dev)
    {
        cudaDeviceProp result;
        cudaError_t err = cudaGetDeviceProperties(&result, dev);

        if (err != cudaSuccess)
            throw cuda::error{ err };
        else
            return result;
    }

    void set_device(int dev)
    {
        cudaError_t err = cudaSetDevice(dev);

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    event::event() : m_event{ new cudaEvent_t, deleter }
    {
        cudaError_t err = cudaEventCreate(m_event.get());

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    cudaEvent_t event::get() { return *m_event; }

    void event::deleter(cudaEvent_t* p)
    {
        cudaEventDestroy(*p);
        delete p;
    }

    stream::stream() : m_stream{ new cudaStream_t, deleter }
    {
        cudaError_t err = cudaStreamCreate(m_stream.get());

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    cudaStream_t stream::get() { return *m_stream; }

    void stream::synchronize()
    {
        cudaError_t err = cudaStreamSynchronize(*m_stream);

        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    void stream::wait(event ev)
    {
        cudaError_t err = cudaStreamWaitEvent( *m_stream, ev.get(), 0);
            
        if (err != cudaSuccess)
            throw cuda::error{ err };
    }

    void stream::deleter(cudaStream_t* p)
    {
        cudaStreamDestroy(*p);
        delete p;
    }

    std::chrono::duration<float, std::chrono::milliseconds::period> get_duration(std::pair<event, event> recording)
    {
        float ms = 0;
        cudaError_t err = cudaEventElapsedTime(&ms, recording.first.get(), recording.second.get());

        if (err != cudaSuccess)
            throw cuda::error{ err };

        return std::chrono::duration<float, std::chrono::milliseconds::period>{ ms };
    }
}
