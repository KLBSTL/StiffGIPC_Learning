#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace gipc
{
struct LinearStructureProbeConfig
{
    // Observations are diagnostic only. None of these settings bypass sorting,
    // assembly, numeric reduction, preconditioning or the linear solve.
    bool enabled = false;
    bool gpu_events = true;
    bool include_converted_structure = true;
};

struct LinearStructureShape
{
    std::uintptr_t owner = 0;
    std::size_t input_offset = 0, item_count = 0, output_offset = 0;
    int device = 0;
    int block_rows = 0, block_cols = 0;
};

enum class LinearStructureState
{
    disabled, first, owner_changed, shape_changed, compare, content_changed, hit
};

constexpr const char* linear_structure_state_name(LinearStructureState state)
{
    switch(state)
    {
    case LinearStructureState::disabled: return "disabled";
    case LinearStructureState::first: return "first";
    case LinearStructureState::owner_changed: return "owner_changed";
    case LinearStructureState::shape_changed: return "shape_changed";
    case LinearStructureState::compare: return "compare";
    case LinearStructureState::content_changed: return "content_changed";
    case LinearStructureState::hit: return "hit";
    }
    return "unknown";
}

constexpr bool linear_structure_same_shape(const LinearStructureShape& a,
                                           const LinearStructureShape& b)
{
    return a.input_offset == b.input_offset && a.item_count == b.item_count
        && a.output_offset == b.output_offset && a.device == b.device
        && a.block_rows == b.block_rows && a.block_cols == b.block_cols;
}

// A bijective encoding of two int32 indices, not a probabilistic hash.
constexpr std::uint64_t linear_structure_key(std::int32_t row, std::int32_t col)
{
    return (std::uint64_t{static_cast<std::uint32_t>(row)} << 32)
        | std::uint64_t{static_cast<std::uint32_t>(col)};
}

constexpr LinearStructureState linear_structure_plan(
    bool enabled, bool previous_valid, const LinearStructureShape& previous,
    const LinearStructureShape& current)
{
    if(!enabled) return LinearStructureState::disabled;
    if(!previous_valid) return LinearStructureState::first;
    if(previous.owner != current.owner) return LinearStructureState::owner_changed;
    if(!linear_structure_same_shape(previous, current)) return LinearStructureState::shape_changed;
    return current.item_count == 0 ? LinearStructureState::hit : LinearStructureState::compare;
}

inline void validate_linear_structure_shape(const LinearStructureShape& shape)
{
    const auto max_int = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(shape.item_count > max_int || shape.input_offset > max_int || shape.output_offset > max_int)
        throw std::runtime_error("Linear structure probe exceeds converter int32 range");
    if(shape.block_rows < 0 || shape.block_cols < 0)
        throw std::runtime_error("Linear structure probe negative matrix dimension");
    if(shape.item_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
        throw std::runtime_error("Linear structure probe snapshot size overflow");
}

struct LinearStructureStageCost
{
    // CPU intervals can include waiting for prior stream work. gpu_valid is
    // false when event timing is disabled or this stage did not run. Snapshot
    // CPU cost without events is submission cost (completion_waited=false).
    double cpu_ms = 0, gpu_ms = 0, readback_cpu_ms = 0;
    bool gpu_valid = false, completion_waited = false;
};

struct LinearStructureProbeResult
{
    LinearStructureShape shape;
    LinearStructureState raw_state = LinearStructureState::disabled;
    LinearStructureState converted_state = LinearStructureState::disabled;
    LinearStructureState pair_state = LinearStructureState::disabled;
    bool observed = false, gap_before = false;
    bool raw_equal = false, raw_comparison_performed = false, raw_snapshot_updated = false;
    bool converted_observed = false, converted_comparison_performed = false;
    bool mapping_equal = false, partition_equal = false, converted_snapshot_updated = false;
    bool pair_observed = false, pair_equal = false, pair_comparison_performed = false;
    bool pair_snapshot_updated = false, pair_gap_before = false;
    std::size_t unique_count = 0; // Meaningful only if converted_observed.
    std::size_t observation_index = 0;
    std::size_t raw_snapshot_bytes = 0, converted_snapshot_bytes = 0;
    std::size_t raw_capacity_bytes = 0, converted_capacity_bytes = 0;
    std::size_t pair_snapshot_bytes = 0, pair_capacity_bytes = 0;
    std::uintptr_t row_storage = 0, col_storage = 0;
    double event_setup_cpu_ms = 0, workspace_cpu_ms = 0, probe_cpu_ms = 0;
    LinearStructureStageCost raw_compare, raw_snapshot, converted_compare, converted_snapshot;
    LinearStructureStageCost pair_compare, pair_snapshot;
};
} // namespace gipc
