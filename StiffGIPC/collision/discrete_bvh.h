#pragma once

// Host-only policy/state for the experimental ordinary (not swept) BVH path.
// Geometry/topology buffers remain owned by lbvh/GIPC. This module owns no GPU
// memory. In-place topology edits must call lbvh::invalidate_discrete_topology().
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <gipc/utils/json.h>

namespace gipc
{
struct DiscreteBVHConfig
{
    bool enabled = false;
    bool validate = false;
    uint32_t rebuild_interval = 8;
};

inline bool discrete_bvh_bool(const char* name)
{
    const char* value = std::getenv(name);
    if(!value) return false;
    if(std::string(value) == "0") return false;
    if(std::string(value) == "1") return true;
    throw std::runtime_error(std::string(name) + " must be exactly 0 or 1");
}

inline const DiscreteBVHConfig& discrete_bvh_config()
{
    static const DiscreteBVHConfig config = [] {
        DiscreteBVHConfig result;
        result.enabled = discrete_bvh_bool("GIPC_DISCRETE_BVH_REFIT");
        result.validate = discrete_bvh_bool("GIPC_DISCRETE_BVH_VALIDATE");
        if(result.validate && !result.enabled)
            throw std::runtime_error("GIPC_DISCRETE_BVH_VALIDATE requires GIPC_DISCRETE_BVH_REFIT=1");
        if(const char* raw = std::getenv("GIPC_DISCRETE_BVH_REBUILD_INTERVAL"))
        {
            const std::string value(raw);
            if(value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("GIPC_DISCRETE_BVH_REBUILD_INTERVAL must be a positive integer");
            const auto interval = std::stoull(value);
            if(interval == 0 || interval > 1024)
                throw std::runtime_error("GIPC_DISCRETE_BVH_REBUILD_INTERVAL must be in [1,1024]");
            result.rebuild_interval = static_cast<uint32_t>(interval);
        }
        return result;
    }();
    return config;
}

enum class DiscreteBVHTreeMode { invalid, discrete, swept };

struct DiscreteBVHState
{
    DiscreteBVHTreeMode mode = DiscreteBVHTreeMode::invalid;
    uint32_t builds_since_rebuild = 0;
    std::array<uintptr_t, 40> signature{};
};

struct DiscreteBVHTreeStats
{
    uint64_t construct_calls = 0;
    uint64_t production_rebuilds = 0;
    uint64_t production_refits = 0;
    uint64_t disabled_rebuilds = 0;
    uint64_t invalid_rebuilds = 0;
    uint64_t swept_rebuilds = 0;
    uint64_t signature_rebuilds = 0;
    uint64_t interval_rebuilds = 0;
    uint64_t storage_switches = 0;
    uint64_t ordinary_cache_restores = 0;
    uint64_t swept_full_builds = 0;
    uint64_t swept_refits = 0;
    uint64_t swept_refit_fallbacks = 0;
    uint64_t swept_cache_capacity_bytes_peak = 0;
    uint64_t validation_calls = 0;
    uint64_t validation_passed = 0;
    uint64_t validation_failed = 0;
    uint64_t diagnostic_refits = 0;
    uint64_t diagnostic_rebuilds = 0;
    uint64_t diagnostic_query_passes = 0;
    uint64_t diagnostic_overflow_retries = 0;
    uint64_t diagnostic_pairs_compared = 0;
    uint64_t diagnostic_peak_pair_capacity = 0;
    uint64_t diagnostic_max_tree_depth = 0;
    double diagnostic_host_ms = 0;
};

inline std::array<DiscreteBVHTreeStats, 2>& discrete_bvh_stats()
{
    static thread_local std::array<DiscreteBVHTreeStats, 2> stats{};
    return stats;
}

// Counters are per host thread (the GIPC owner thread), cumulative since reset.
// Resetting counters never resets a live tree's interval or validity state.
inline void discrete_bvh_reset_stats() { discrete_bvh_stats() = {}; }

inline Json discrete_bvh_stats_json()
{
    const auto& config = discrete_bvh_config();
    Json result{{"enabled", config.enabled}, {"validate", config.validate},
                {"rebuild_interval", config.rebuild_interval},
                {"independent_ordinary_swept_storage", config.enabled},
                {"diagnostic_changes_tree_to_rebuilt", config.validate},
                {"diagnostic_cost_included_in_production_counters", false}};
    for(size_t i = 0; i < 2; ++i)
    {
        const auto& s = discrete_bvh_stats()[i];
        result[i == 0 ? "face" : "edge"] = {
            {"construct_calls", s.construct_calls},
            {"production_rebuilds", s.production_rebuilds},
            {"production_refits", s.production_refits},
            {"disabled_rebuilds", s.disabled_rebuilds},
            {"invalid_rebuilds", s.invalid_rebuilds},
            {"swept_rebuilds", s.swept_rebuilds},
            {"signature_rebuilds", s.signature_rebuilds},
            {"interval_rebuilds", s.interval_rebuilds},
            {"storage_switches", s.storage_switches},
            {"ordinary_cache_restores", s.ordinary_cache_restores},
            {"swept_full_builds", s.swept_full_builds},
            {"swept_refits", s.swept_refits},
            {"swept_refit_fallbacks", s.swept_refit_fallbacks},
            {"swept_cache_capacity_bytes_peak", s.swept_cache_capacity_bytes_peak},
            {"validation_calls", s.validation_calls},
            {"validation_passed", s.validation_passed},
            {"validation_failed", s.validation_failed},
            {"diagnostic_refits", s.diagnostic_refits},
            {"diagnostic_rebuilds", s.diagnostic_rebuilds},
            {"diagnostic_query_passes", s.diagnostic_query_passes},
            {"diagnostic_overflow_retries", s.diagnostic_overflow_retries},
            {"diagnostic_pairs_compared", s.diagnostic_pairs_compared},
            {"diagnostic_peak_pair_capacity", s.diagnostic_peak_pair_capacity},
            {"diagnostic_max_tree_depth", s.diagnostic_max_tree_depth},
            {"diagnostic_host_ms", s.diagnostic_host_ms}};
    }
    return result;
}
} // namespace gipc
