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
    for(const char* name : {"GIPC_BOUNDED_CCD", "GIPC_BOUNDED_CCD_VALIDATE",
                            "GIPC_BVH_ELIGIBILITY", "GIPC_BVH_ELIGIBILITY_VALIDATE",
                            "GIPC_ELIGIBILITY", "GIPC_ELIGIBILITY_VALIDATE"})
    {
        const char* value=std::getenv(name);
        if(!value || std::string(value)=="0")continue;
        if(std::string(value)!="1")
            throw std::runtime_error(std::string(name)+" must be exactly 0 or 1; this component is retired");
        throw std::runtime_error(std::string(name)+
            " is retired after negative performance screening and is unavailable in this build; "
            "use archive/pre-cleanup-20261006 to reproduce the historical experiment");
    }
    for(const char* name : {"GIPC_BOUNDED_CCD_FIXTURE", "GIPC_BVH_ELIGIBILITY_FIXTURE",
                            "GIPC_ELIGIBILITY_FIXTURE"})
        if(std::getenv(name))
            throw std::runtime_error(std::string(name)+
                " is a retired fixture and is unavailable in this build; "
                "use archive/pre-cleanup-20261006 for the historical fixture");
}
}
