#pragma once
#include <cstdlib>
#include <string>
#include <stdexcept>
namespace gipc
{
inline int& legacy_restrict_override(){thread_local static int value=-1;return value;}
inline bool legacy_ordered_restrict()
{
    static const bool enabled=[](){
        const char* raw=std::getenv("GIPC_LEGACY_RESTRICT");
        const std::string mode=raw?raw:"atomic";
        if(mode!="atomic" && mode!="ordered")throw std::runtime_error("Invalid legacy restriction mode");
        return mode=="ordered";
    }();
    return legacy_restrict_override()<0?enabled:legacy_restrict_override()!=0;
}
inline bool legacy_restrict_study()
{
    const char* raw=std::getenv("GIPC_FIXED_LEGACY_RESTRICT_STUDY");
    return raw && std::string(raw)=="1";
}
struct LegacyRestrictionArm
{
    int saved=legacy_restrict_override();
    explicit LegacyRestrictionArm(bool ordered){legacy_restrict_override()=ordered?1:0;}
    ~LegacyRestrictionArm(){legacy_restrict_override()=saved;}
};
}
