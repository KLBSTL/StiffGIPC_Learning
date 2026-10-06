#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc
{
// The production choice is process-constant. Only a protected fixed-system
// experiment may override it BETWEEN solves, never during a PCG solve.
inline int& mas_restrict_diagnostic_override()
{
    thread_local static int mode=-1;
    return mode;
}
inline bool mas_warp_restrict()
{
    static const bool enabled=[]() {
        const char* raw=std::getenv("GIPC_MAS_RESTRICT_MODE");
        const std::string mode=raw?raw:"serial";
        if(mode!="serial"&&mode!="warp")
            throw std::runtime_error("GIPC_MAS_RESTRICT_MODE must be serial or warp");
        return mode=="warp";
    }();
    return mas_restrict_diagnostic_override()<0?enabled:mas_restrict_diagnostic_override()!=0;
}
struct MasRestrictionStudyArm
{
    int saved=mas_restrict_diagnostic_override();
    explicit MasRestrictionStudyArm(bool warp){mas_restrict_diagnostic_override()=warp?1:0;}
    ~MasRestrictionStudyArm(){mas_restrict_diagnostic_override()=saved;}
};
}
