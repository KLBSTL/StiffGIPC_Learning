#pragma once
#include <collision/ipc_bounded_ccd.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace gipc
{
// Host counters only. Production CCD has no per-pair diagnostic atomics,
// copies, allocations or new timing events.
struct IpcBoundedCCDStats
{
    uint64_t second_queries=0,bounded_queries=0,unsupported_queries=0;
    uint64_t pairs=0,validation_calls=0,validation_failed=0,cut_pairs=0;
    uint64_t old_iterations=0,new_iterations=0;
    double diagnostic_host_ms=0;
};
inline IpcBoundedCCDStats& ipc_bounded_ccd_stats()
{static thread_local IpcBoundedCCDStats s;return s;}
inline void ipc_bounded_ccd_reset_stats(){ipc_bounded_ccd_stats()={};}
inline bool ipc_bounded_ccd_same_bits(double a,double b)
{
    uint64_t aa,bb;std::memcpy(&aa,&a,sizeof(aa));std::memcpy(&bb,&b,sizeof(bb));
    return aa==bb;
}
inline Json ipc_bounded_ccd_stats_json()
{
    const auto& c=ipc_bounded_ccd_config();const auto& s=ipc_bounded_ccd_stats();
    return {{"requested",c.enabled},{"validate",c.validate},
        {"scope","second_ipc_swept_query_only"},{"all_candidate_pairs_retained",true},
        {"second_queries",s.second_queries},{"bounded_queries",s.bounded_queries},
        {"unsupported_queries",s.unsupported_queries},{"pairs",s.pairs},
        {"validation_calls",s.validation_calls},{"validation_failed",s.validation_failed},
        {"cut_pairs",s.cut_pairs},{"old_iterations",s.old_iterations},
        {"new_iterations",s.new_iterations},{"diagnostic_host_ms",s.diagnostic_host_ms},
        {"iteration_counters_effective",c.validate}};
}
}
