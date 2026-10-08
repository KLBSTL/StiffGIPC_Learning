#pragma once
#include <linear_system/utils/linear_structure_probe.h>
#include <cuda_tools/cuda_buffer_view.h>
#include <cuda_tools/scoped_cuda_events.h>
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <string>

namespace gipc
{
namespace linear_structure_probe_detail
{
inline void check(cudaError_t status, const char* operation)
{
    if(status != cudaSuccess)
        throw std::runtime_error(std::string("Linear structure probe ") + operation + ": "
                                 + cudaGetErrorString(status));
}

using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

template<class = void>
__global__ void compare_raw(const int* rows, const int* cols,
                            const std::uint64_t* snapshot, std::size_t count,
                            unsigned int* mismatch)
{
    const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if(i >= count) return;
    const auto key = (std::uint64_t{static_cast<std::uint32_t>(rows[i])} << 32)
        | std::uint64_t{static_cast<std::uint32_t>(cols[i])};
    if(key != snapshot[i]) atomicOr(mismatch, 1u);
}

template<class = void>
__global__ void snapshot_raw(const int* rows, const int* cols,
                             std::uint64_t* snapshot, std::size_t count)
{
    const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if(i >= count) return;
    snapshot[i] = (std::uint64_t{static_cast<std::uint32_t>(rows[i])} << 32)
        | std::uint64_t{static_cast<std::uint32_t>(cols[i])};
}

template<class = void>
__global__ void compare_converted(const std::uint32_t* mapping,
                                  const std::uint32_t* partition,
                                  const std::uint32_t* mapping_snapshot,
                                  const std::uint32_t* partition_snapshot,
                                  std::size_t count, unsigned int* mismatch)
{
    const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if(i >= count) return;
    unsigned int bits = 0;
    if(mapping[i] != mapping_snapshot[i]) bits |= 1u;
    if(partition[i] != partition_snapshot[i]) bits |= 2u;
    if(bits) atomicOr(mismatch, bits);
}
} // namespace linear_structure_probe_detail

// Owned by one Converter. Buffers contain only diagnostic indices; none are
// exposed as a production cache, and no Hessian/RHS/numerical values are saved.
class LinearStructureProbe
{
    LinearStructureProbeConfig config_;
    LinearStructureShape raw_shape_, converted_shape_, pair_shape_;
    LinearStructureProbeResult result_;
    bool raw_valid_ = false, converted_valid_ = false, gap_pending_ = false;
    bool pair_valid_ = false, pair_gap_pending_ = false;
    std::size_t converted_unique_count_ = 0;
    std::size_t observation_count_ = 0;
    int resource_device_ = -1;
    cudatool::DeviceBuffer<std::uint64_t> raw_snapshot_;
    cudatool::DeviceBuffer<std::uint64_t> pair_snapshot_;
    cudatool::DeviceBuffer<std::uint32_t> mapping_snapshot_, partition_snapshot_;
    cudatool::DeviceBuffer<unsigned int> mismatch_;
    std::unique_ptr<ScopedCudaEvents<2>> events_;

    void prepare_events()
    {
        if(!config_.gpu_events || events_) return;
        const auto begin = linear_structure_probe_detail::Clock::now();
        events_ = std::make_unique<ScopedCudaEvents<2>>();
        result_.event_setup_cpu_ms += linear_structure_probe_detail::elapsed(begin);
    }

    template<class T> void resize_workspace(cudatool::DeviceBuffer<T>& buffer, std::size_t count)
    {
        const auto begin = linear_structure_probe_detail::Clock::now();
        buffer.resize_discard(count);
        result_.workspace_cpu_ms += linear_structure_probe_detail::elapsed(begin);
    }

    template<class Launch> LinearStructureStageCost measure(Launch launch, unsigned int* host_bits = nullptr)
    {
        using namespace linear_structure_probe_detail;
        prepare_events();
        LinearStructureStageCost cost;
        const auto begin = Clock::now();
        if(config_.gpu_events) check(cudaEventRecord((*events_)[0], cudaStreamPerThread), "event begin");
        launch();
        check(cudaGetLastError(), "kernel launch");
        if(config_.gpu_events) check(cudaEventRecord((*events_)[1], cudaStreamPerThread), "event end");
        if(host_bits)
        {
            const auto readback_begin = Clock::now();
            check(cudaMemcpyAsync(host_bits, mismatch_.data(), sizeof(*host_bits),
                                  cudaMemcpyDeviceToHost, cudaStreamPerThread), "flag readback");
            check(cudaStreamSynchronize(cudaStreamPerThread), "flag completion");
            cost.readback_cpu_ms = elapsed(readback_begin);
            cost.completion_waited = true;
        }
        if(config_.gpu_events)
        {
            check(cudaEventSynchronize((*events_)[1]), "event completion");
            float milliseconds = 0;
            check(cudaEventElapsedTime(&milliseconds, (*events_)[0], (*events_)[1]), "event elapsed");
            cost.gpu_ms = milliseconds;
            cost.gpu_valid = true;
            cost.completion_waited = true;
        }
        cost.cpu_ms = elapsed(begin);
        return cost;
    }

public:
    bool enabled() const noexcept { return config_.enabled; }
    const LinearStructureProbeConfig& config() const noexcept { return config_; }
    const LinearStructureProbeResult* last_result() const noexcept
    {
        return config_.enabled && result_.observed ? &result_ : nullptr;
    }
    // Reset only metadata. Disabled/configuration changes do not issue CUDA
    // frees, comparisons, copies, event creation or other diagnostic work.
    void reset() noexcept
    {
        gap_pending_ = raw_valid_ || converted_valid_ || pair_valid_ || gap_pending_;
        pair_gap_pending_ = pair_valid_ || pair_gap_pending_;
        raw_valid_ = converted_valid_ = false;
        pair_valid_ = false;
        result_ = {};
    }
    void configure(const LinearStructureProbeConfig& config) noexcept
    {
        if(!config.enabled || (!config_.enabled && config.enabled)) reset();
        if(!config.include_converted_structure)
        {
            converted_valid_ = false;
            pair_gap_pending_ = pair_valid_ || pair_gap_pending_;
            pair_valid_ = false;
        }
        config_ = config;
    }

    void observe_raw(std::uintptr_t owner, std::size_t input_offset,
                     std::size_t count, std::size_t output_offset,
                     const int* rows, const int* cols,
                     int block_rows = 0, int block_cols = 0)
    {
        if(!config_.enabled) return;
        using namespace linear_structure_probe_detail;
        const auto begin = Clock::now();
        LinearStructureShape current{owner, input_offset, count, output_offset, 0,
                                     block_rows, block_cols};
        validate_linear_structure_shape(current);
        if(count && (!rows || !cols)) throw std::runtime_error("Linear structure probe null raw indices");
        check(cudaGetDevice(&current.device), "get device");
        if(resource_device_ != -1 && resource_device_ != current.device)
            throw std::runtime_error("Linear structure probe cannot migrate its owned CUDA resources between devices");
        resource_device_ = current.device;
        // A caller that omits a converted/pair observation must not compare
        // across the skipped system on a later call.
        if(result_.observed && !result_.converted_observed) converted_valid_ = false;
        if(result_.observed && !result_.pair_observed)
        {
            pair_gap_pending_ = pair_valid_ || pair_gap_pending_;
            pair_valid_ = false;
        }
        result_ = {};
        result_.shape = current;
        result_.observed = true;
        result_.observation_index = ++observation_count_;
        result_.gap_before = gap_pending_;
        gap_pending_ = false;
        result_.row_storage = reinterpret_cast<std::uintptr_t>(rows);
        result_.col_storage = reinterpret_cast<std::uintptr_t>(cols);
        result_.raw_state = linear_structure_plan(true, raw_valid_, raw_shape_, current);
        result_.raw_snapshot_bytes = count * sizeof(std::uint64_t);
        if(result_.raw_state == LinearStructureState::compare)
        {
            resize_workspace(mismatch_, 1);
            unsigned int bits = 0;
            result_.raw_compare = measure([&]()
            {
                check(cudaMemsetAsync(mismatch_.data(), 0, sizeof(unsigned int), cudaStreamPerThread), "raw flag zero");
                compare_raw<><<<static_cast<unsigned int>((count + 255) / 256), 256,
                                0, cudaStreamPerThread>>>(
                    rows, cols, raw_snapshot_.data(), count, mismatch_.data());
            }, &bits);
            result_.raw_comparison_performed = true;
            result_.raw_equal = bits == 0;
            result_.raw_state = bits == 0 ? LinearStructureState::hit : LinearStructureState::content_changed;
        }
        else result_.raw_equal = result_.raw_state == LinearStructureState::hit;
        if(!result_.raw_equal)
        {
            resize_workspace(raw_snapshot_, count);
            if(count)
            {
                result_.raw_snapshot = measure([&]()
                {
                    snapshot_raw<><<<static_cast<unsigned int>((count + 255) / 256), 256,
                                     0, cudaStreamPerThread>>>(
                        rows, cols, raw_snapshot_.data(), count);
                });
                result_.raw_snapshot_updated = true;
            }
            raw_shape_ = current;
            raw_valid_ = true;
        }
        result_.raw_capacity_bytes = raw_snapshot_.capacity() * sizeof(std::uint64_t);
        result_.probe_cpu_ms += elapsed(begin);
    }

    void observe_converted(const std::uint32_t* mapping, const std::uint32_t* partition,
                           std::size_t unique_count, const int* compact_rows = nullptr,
                           const int* compact_cols = nullptr, bool observe_pairs = true)
    {
        if(!config_.enabled || !config_.include_converted_structure) return;
        if(!result_.observed) throw std::runtime_error("Converted structure probe requires raw observation");
        if(result_.converted_observed) throw std::runtime_error("Converted structure probe observed twice for one system");
        using namespace linear_structure_probe_detail;
        const auto begin = Clock::now();
        const auto current = result_.shape;
        const auto count = current.item_count;
        if(unique_count > count || (count && (!mapping || !partition)))
            throw std::runtime_error("Invalid converted structure probe shape");
        if(observe_pairs && unique_count && (!compact_rows || !compact_cols))
            throw std::runtime_error("Linear structure probe null compact pairs");
        result_.converted_observed = true;
        result_.unique_count = unique_count;
        result_.converted_state = linear_structure_plan(true, converted_valid_, converted_shape_, current);
        if(converted_valid_ && converted_unique_count_ != unique_count
           && result_.converted_state != LinearStructureState::owner_changed)
            result_.converted_state = LinearStructureState::shape_changed;
        result_.converted_snapshot_bytes = 2 * count * sizeof(std::uint32_t);
        if(result_.converted_state == LinearStructureState::compare)
        {
            resize_workspace(mismatch_, 1);
            unsigned int bits = 0;
            result_.converted_compare = measure([&]()
            {
                check(cudaMemsetAsync(mismatch_.data(), 0, sizeof(unsigned int), cudaStreamPerThread), "converted flag zero");
                compare_converted<><<<static_cast<unsigned int>((count + 255) / 256), 256,
                                      0, cudaStreamPerThread>>>(
                    mapping, partition, mapping_snapshot_.data(), partition_snapshot_.data(), count, mismatch_.data());
            }, &bits);
            result_.converted_comparison_performed = true;
            result_.mapping_equal = (bits & 1u) == 0;
            result_.partition_equal = (bits & 2u) == 0;
            result_.converted_state = bits == 0 ? LinearStructureState::hit : LinearStructureState::content_changed;
        }
        else result_.mapping_equal = result_.partition_equal = result_.converted_state == LinearStructureState::hit;
        if(!result_.mapping_equal || !result_.partition_equal)
        {
            resize_workspace(mapping_snapshot_, count);
            resize_workspace(partition_snapshot_, count);
            if(count)
            {
                result_.converted_snapshot = measure([&]()
                {
                    check(cudaMemcpyAsync(mapping_snapshot_.data(), mapping, count * sizeof(std::uint32_t),
                                          cudaMemcpyDeviceToDevice, cudaStreamPerThread), "mapping snapshot");
                    check(cudaMemcpyAsync(partition_snapshot_.data(), partition, count * sizeof(std::uint32_t),
                                          cudaMemcpyDeviceToDevice, cudaStreamPerThread), "partition snapshot");
                });
                result_.converted_snapshot_updated = true;
            }
            converted_shape_ = current;
            converted_unique_count_ = unique_count;
            converted_valid_ = true;
        }
        result_.converted_capacity_bytes = (mapping_snapshot_.capacity() + partition_snapshot_.capacity())
            * sizeof(std::uint32_t);
        if(observe_pairs)
        {
            result_.pair_observed = true;
            result_.pair_gap_before = pair_gap_pending_;
            pair_gap_pending_ = false;
            // Canonical pairs have a separate shape. Raw multiplicity and
            // staging offsets do not change this symbolic pair identity.
            auto pair_current = current;
            pair_current.input_offset = pair_current.output_offset = 0;
            pair_current.item_count = unique_count;
            result_.pair_state = linear_structure_plan(true, pair_valid_, pair_shape_, pair_current);
            result_.pair_snapshot_bytes = unique_count * sizeof(std::uint64_t);
            if(result_.pair_state == LinearStructureState::compare)
            {
                resize_workspace(mismatch_, 1);
                unsigned int bits = 0;
                result_.pair_compare = measure([&]()
                {
                    check(cudaMemsetAsync(mismatch_.data(), 0, sizeof(unsigned int), cudaStreamPerThread), "pair flag zero");
                    compare_raw<><<<static_cast<unsigned int>((unique_count + 255) / 256), 256,
                                    0, cudaStreamPerThread>>>(
                        compact_rows, compact_cols, pair_snapshot_.data(), unique_count, mismatch_.data());
                }, &bits);
                result_.pair_comparison_performed = true;
                result_.pair_equal = bits == 0;
                result_.pair_state = bits == 0 ? LinearStructureState::hit : LinearStructureState::content_changed;
            }
            else result_.pair_equal = result_.pair_state == LinearStructureState::hit;
            if(!result_.pair_equal)
            {
                resize_workspace(pair_snapshot_, unique_count);
                if(unique_count)
                {
                    result_.pair_snapshot = measure([&]()
                    {
                        snapshot_raw<><<<static_cast<unsigned int>((unique_count + 255) / 256), 256,
                                         0, cudaStreamPerThread>>>(
                            compact_rows, compact_cols, pair_snapshot_.data(), unique_count);
                    });
                    result_.pair_snapshot_updated = true;
                }
                pair_shape_ = pair_current;
                pair_valid_ = true;
            }
        }
        else
        {
            pair_gap_pending_ = pair_valid_ || pair_gap_pending_;
            pair_valid_ = false;
        }
        result_.pair_capacity_bytes = pair_snapshot_.capacity() * sizeof(std::uint64_t);
        result_.probe_cpu_ms += elapsed(begin);
    }
};
} // namespace gipc
