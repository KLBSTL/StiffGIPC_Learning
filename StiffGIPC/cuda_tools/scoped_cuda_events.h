#pragma once
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <string>

namespace gipc
{
// Own timing events across normal, terminal and exception exits. Destroying
// an event does not introduce a device wait or change its recorded interval.
template <size_t Count>
class ScopedCudaEvents
{
    std::array<cudaEvent_t, Count> events{};
    void release() noexcept
    {
        for(auto& event : events)
        {
            if(event) cudaEventDestroy(event);
            event = nullptr;
        }
    }
public:
    ScopedCudaEvents()
    {
        for(auto& event : events)
        {
            const auto status = cudaEventCreate(&event);
            if(status != cudaSuccess)
            {
                release();
                throw std::runtime_error(std::string("Timing event creation failed: ") +
                                         cudaGetErrorString(status));
            }
        }
    }
    ~ScopedCudaEvents() noexcept { release(); }
    ScopedCudaEvents(const ScopedCudaEvents&) = delete;
    ScopedCudaEvents& operator=(const ScopedCudaEvents&) = delete;
    cudaEvent_t operator[](size_t index) const noexcept { return events[index]; }
};
} // namespace gipc
