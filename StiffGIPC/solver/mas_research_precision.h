#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <gipc/statistics.h>
// Bounded X6 intervention, separate from the retired production option.
// A solve never changes its restriction arithmetic during PCG.
namespace gipc {
inline int& research_restriction_override() { static int mode=-1;return mode; }
inline int research_restriction_mode() {
    if(research_restriction_override()>=0)return research_restriction_override();
    const char* raw=std::getenv("GIPC_RESEARCH_RESTRICTION");
    if(!raw||std::string(raw)=="native")return 0;
    const std::string value(raw);
    int mode=value=="ordered_fp32"?1:value=="ordered_fp64"?2:-1;
    if(mode<0)throw std::runtime_error("Invalid research restriction mode");
    const char* first=std::getenv("GIPC_RESEARCH_FIRST_DIRECTION_ONLY");
    if(first&&std::string(first)=="1" &&
       Statistics::instance().at_current_frame()["newton"].size()!=1)return 0;
    return mode;
}
inline bool research_restriction_map_needed() {
    const char* fixed=std::getenv("GIPC_RESEARCH_FIXED_PRECISION");
    return research_restriction_mode()!=0 || (fixed&&std::string(fixed)=="1");
}
}
