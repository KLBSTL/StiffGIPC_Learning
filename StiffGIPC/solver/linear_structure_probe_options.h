#pragma once
#include <linear_system/utils/linear_structure_probe.h>
#include <gipc/cost_trace.h>

namespace gipc
{
inline const LinearStructureProbeConfig& linear_structure_probe_options()
{
    static const auto config=[] {
        LinearStructureProbeConfig value;
        if(const char* raw=std::getenv("GIPC_LINEAR_STRUCTURE_PROBE"))
        {
            const std::string flag(raw);
            if(flag!="0" && flag!="1")
                throw std::runtime_error("GIPC_LINEAR_STRUCTURE_PROBE must be 0 or 1");
            value.enabled=flag=="1";
        }
        if(value.enabled)
        {
            if(cost_trace_config().path.empty())
                throw std::runtime_error("Linear structure probe requires independent cost diagnostics");
            const char* backend=std::getenv("GIPC_CONTACT_BACKEND");
            if(backend && std::string(backend)!="ipc")
                throw std::runtime_error("Linear structure probe supports IPC only");
            value.gpu_events=cost_trace_config().gpu_events;
        }
        return value;
    }();
    return config;
}

inline Json linear_structure_probe_options_json()
{
    const auto& config=linear_structure_probe_options();
    return {{"requested",config.enabled},{"effective",config.enabled},
        {"gpu_events",config.enabled && config.gpu_events},
        {"include_converted_structure",config.include_converted_structure},
        {"selection","GIPC_COST_FRAMES"},{"numerical_path","unchanged_full_conversion"}};
}

inline Json linear_structure_probe_json(const LinearStructureProbeResult& value)
{
    auto stage=[](const LinearStructureStageCost& cost) {
        return Json{{"cpu_ms",cost.cpu_ms},{"gpu_ms",cost.gpu_ms},
            {"gpu_valid",cost.gpu_valid},{"completion_waited",cost.completion_waited},
            {"readback_cpu_ms",cost.readback_cpu_ms}};
    };
    Json result={{"observed",value.observed},{"gap_before",value.gap_before},
        {"observation_index",value.observation_index},
        {"raw_state",linear_structure_state_name(value.raw_state)},
        {"converted_state",linear_structure_state_name(value.converted_state)},
        {"raw_equal",value.raw_equal},{"raw_comparison_performed",value.raw_comparison_performed},
        {"raw_snapshot_updated",value.raw_snapshot_updated},
        {"converted_observed",value.converted_observed},
        {"converted_comparison_performed",value.converted_comparison_performed},
        {"mapping_equal",value.mapping_equal},{"partition_equal",value.partition_equal},
        {"converted_snapshot_updated",value.converted_snapshot_updated},
        {"pair_state",linear_structure_state_name(value.pair_state)},
        {"pair_observed",value.pair_observed},{"pair_equal",value.pair_equal},
        {"pair_comparison_performed",value.pair_comparison_performed},
        {"pair_snapshot_updated",value.pair_snapshot_updated},{"pair_gap_before",value.pair_gap_before},
        {"owner",value.shape.owner},{"input_offset",value.shape.input_offset},
        {"item_count",value.shape.item_count},{"output_offset",value.shape.output_offset},
        {"block_rows",value.shape.block_rows},{"block_cols",value.shape.block_cols},
        {"device",value.shape.device},{"unique_count",value.unique_count},
        {"row_storage",value.row_storage},{"col_storage",value.col_storage},
        {"raw_snapshot_bytes",value.raw_snapshot_bytes},
        {"converted_snapshot_bytes",value.converted_snapshot_bytes},
        {"raw_capacity_bytes",value.raw_capacity_bytes},
        {"converted_capacity_bytes",value.converted_capacity_bytes},
        {"pair_snapshot_bytes",value.pair_snapshot_bytes},{"pair_capacity_bytes",value.pair_capacity_bytes},
        {"event_setup_cpu_ms",value.event_setup_cpu_ms},
        {"workspace_cpu_ms",value.workspace_cpu_ms},{"probe_cpu_ms",value.probe_cpu_ms},
        {"raw_compare",stage(value.raw_compare)},{"raw_snapshot",stage(value.raw_snapshot)},
        {"converted_compare",stage(value.converted_compare)},
        {"converted_snapshot",stage(value.converted_snapshot)},
        {"pair_compare",stage(value.pair_compare)},{"pair_snapshot",stage(value.pair_snapshot)}};
    attach_solve_context(result);
    return result;
}
} // namespace gipc
