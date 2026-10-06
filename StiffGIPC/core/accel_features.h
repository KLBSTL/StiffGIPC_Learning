#pragma once
#include <cstdlib>
#include <cstring>
inline bool gipc_accel_feature(const char* name)
{
    if(const char* v=std::getenv(name)) return std::strcmp(v,"1")==0;
    if(const char* v=std::getenv("GIPC_ACCEL_SUITE")) return std::strcmp(v,"1")==0;
    return false;
}
