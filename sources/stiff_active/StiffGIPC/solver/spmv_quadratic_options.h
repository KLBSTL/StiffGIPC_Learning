#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gipc
{
inline bool spmv_quadratic_strict_bool(const char* name)
{
    const char* value=std::getenv(name);
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
}
inline bool spmv_fused_quadratic_requested()
{return spmv_quadratic_strict_bool("GIPC_SPMV_FUSED_QUADRATIC");}
inline bool spmv_quadratic_study_requested()
{return spmv_quadratic_strict_bool("GIPC_FIXED_SPMV_QUADRATIC_STUDY");}
}
