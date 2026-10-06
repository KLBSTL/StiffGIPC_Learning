#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <gipc/utils/json.h>

namespace gipc
{
struct QueryEligibilityConfig { bool enabled=false,validate=false; };
inline bool query_eligibility_bool(const char* name)
{
    const char* value=std::getenv(name);
    if(!value || std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
}
inline const QueryEligibilityConfig& query_eligibility_config()
{
    static const QueryEligibilityConfig config=[] {
        QueryEligibilityConfig c;
        c.enabled=query_eligibility_bool("GIPC_BVH_ELIGIBILITY");
        c.validate=query_eligibility_bool("GIPC_BVH_ELIGIBILITY_VALIDATE");
        if(c.validate && !c.enabled)
            throw std::runtime_error("GIPC_BVH_ELIGIBILITY_VALIDATE requires GIPC_BVH_ELIGIBILITY=1");
        return c;
    }();
    return config;
}

// Body values have no reserved sentinel. Homogeneity is a separate flag.
// A leaf key is BodyID[element.x], exactly matching the existing leaf test.
struct QueryEligibilitySummary
{
    int body=0;
    uint32_t homogeneous=0,all_fixed=0,max_original_id=0;
};
static_assert(sizeof(QueryEligibilitySummary)==16,"Eligibility summary layout");

struct QueryEligibilityStats
{
    uint64_t query_calls=0,enabled_queries=0,prepare_calls=0,empty_queries=0;
    uint64_t summary_nodes=0,scratch_bytes_peak=0;
    uint64_t validation_calls=0,validation_passed=0,validation_failed=0;
    uint64_t old_query_passes=0,new_query_passes=0,old_overflow_retries=0,new_overflow_retries=0;
    uint64_t old_peak_capacity=0,new_peak_capacity=0,pairs_compared=0;
    uint64_t diagnostic_nodes_tested=0,diagnostic_pruned_body=0;
    uint64_t diagnostic_pruned_fixed=0,diagnostic_pruned_id=0;
    double diagnostic_host_ms=0;
};
// 0=ordinary VF, 1=ordinary EE, 2=swept VF, 3=swept EE.
inline std::array<QueryEligibilityStats,4>& query_eligibility_stats()
{static thread_local std::array<QueryEligibilityStats,4> value{};return value;}
inline void query_eligibility_reset_stats(){query_eligibility_stats()={};}
inline Json query_eligibility_stats_json()
{
    const auto& c=query_eligibility_config();
    Json result={{"enabled",c.enabled},{"validate",c.validate},
        {"summary_refreshed_every_query",true},{"production_prune_atomics",false},
        {"diagnostic_counters_include_retry_passes",true}};
    const char* names[]={"ordinary_vf","ordinary_ee","swept_vf","swept_ee"};
    for(size_t i=0;i<4;++i)
    {
        const auto& s=query_eligibility_stats()[i];
        result[names[i]]={{"query_calls",s.query_calls},{"enabled_queries",s.enabled_queries},
            {"prepare_calls",s.prepare_calls},{"empty_queries",s.empty_queries},
            {"summary_nodes",s.summary_nodes},{"scratch_bytes_peak",s.scratch_bytes_peak},
            {"validation_calls",s.validation_calls},{"validation_passed",s.validation_passed},
            {"validation_failed",s.validation_failed},{"old_query_passes",s.old_query_passes},
            {"new_query_passes",s.new_query_passes},{"old_overflow_retries",s.old_overflow_retries},
            {"new_overflow_retries",s.new_overflow_retries},{"old_peak_capacity",s.old_peak_capacity},
            {"new_peak_capacity",s.new_peak_capacity},{"pairs_compared",s.pairs_compared},
            {"diagnostic_nodes_tested",s.diagnostic_nodes_tested},
            {"diagnostic_pruned_body",s.diagnostic_pruned_body},
            {"diagnostic_pruned_fixed",s.diagnostic_pruned_fixed},
            {"diagnostic_pruned_id",s.diagnostic_pruned_id},{"diagnostic_host_ms",s.diagnostic_host_ms}};
    }
    return result;
}
int query_eligibility_fixture(const char* output);
}
