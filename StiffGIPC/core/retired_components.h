#pragma once
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace gipc
{
// Historical configuration files remain readable by the runner. A direct native
// invocation must not silently turn an explicitly requested retired method off.
// Call before any fixture dispatch, scene loading, or GPU initialization.
inline void reject_retired_components()
{
    if(const char* raw=std::getenv("GIPC_TOI_INNER_EXIT"))
        if(std::string(raw)=="velocity_only")
            throw std::runtime_error(
                "GIPC_TOI_INNER_EXIT=velocity_only is retired after failed convergence/workload experiments; "
                "use commit 2befec75b217fdcc424d922719453959f3b212cc for the historical implementation");
    // This old diagnostic accepted frame:outer, not a Boolean. Even an empty
    // value or 0 is an explicit unsupported request; do not silently ignore it.
    if(std::getenv("GIPC_TOI_FULL_STEP_EXIT_PROBE"))
        throw std::runtime_error(
            "GIPC_TOI_FULL_STEP_EXIT_PROBE is retired after its bounded experiment; "
            "use commit 2befec75b217fdcc424d922719453959f3b212cc for the historical implementation");
    for(const char* name : {"GIPC_BOUNDED_CCD", "GIPC_BOUNDED_CCD_VALIDATE",
                            "GIPC_BVH_ELIGIBILITY", "GIPC_BVH_ELIGIBILITY_VALIDATE",
                            "GIPC_ELIGIBILITY", "GIPC_ELIGIBILITY_VALIDATE",
                            "GIPC_MAS_FUSED_DOT", "GIPC_FIXED_MAS_DOT_STUDY",
                            "GIPC_SPMV_FUSED_QUADRATIC", "GIPC_FIXED_SPMV_QUADRATIC_STUDY",
                            "GIPC_FIXED_LEGACY_RESTRICT_STUDY"})
    {
        const char* value=std::getenv(name);
        if(!value || std::string(value)=="0")continue;
        if(std::string(value)!="1")
            throw std::runtime_error(std::string(name)+" must be exactly 0 or 1; this component is retired");
        throw std::runtime_error(std::string(name)+
            " is retired after failing performance promotion and is unavailable in this build; "
            "use archive/pre-cleanup-20261006 to reproduce the historical experiment");
    }
    if(const char* raw=std::getenv("GIPC_LEGACY_RESTRICT"))
    {
        const std::string mode=raw;
        if(mode=="ordered")
            throw std::runtime_error(
                "GIPC_LEGACY_RESTRICT=ordered is retired after failing performance promotion; "
                "use archive/pre-cleanup-20261006 to reproduce the historical experiment");
        if(mode!="atomic")
            throw std::runtime_error(
                "GIPC_LEGACY_RESTRICT must be exactly atomic; ordered is retired");
    }
    for(const char* name : {"GIPC_BOUNDED_CCD_FIXTURE", "GIPC_BVH_ELIGIBILITY_FIXTURE",
                            "GIPC_ELIGIBILITY_FIXTURE"})
        if(std::getenv(name))
            throw std::runtime_error(std::string(name)+
                " is a retired fixture and is unavailable in this build; "
                "use archive/pre-cleanup-20261006 for the historical fixture");
}
}
