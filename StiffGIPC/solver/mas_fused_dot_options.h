#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gipc
{
inline bool mas_dot_strict_bool(const char* name)
{
    const char* value=std::getenv(name);
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
}
inline bool mas_fused_dot_requested(){return mas_dot_strict_bool("GIPC_MAS_FUSED_DOT");}
inline bool mas_fused_dot_study_requested(){return mas_dot_strict_bool("GIPC_FIXED_MAS_DOT_STUDY");}
}
