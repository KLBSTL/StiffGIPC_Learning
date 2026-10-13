#pragma once
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc {
inline bool frame_recording_enabled() {
    static const bool enabled=[]() {
        const char* raw=std::getenv("GIPC_RECORD_TIMING");
        if(!raw || std::string(raw)=="0")return false;
        if(std::string(raw)=="1")return true;
        throw std::runtime_error("GIPC_RECORD_TIMING must be 0 or 1");
    }();
    return enabled;
}
inline double recording_elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
}
