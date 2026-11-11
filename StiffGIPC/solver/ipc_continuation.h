#pragma once
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <gipc/utils/json.h>

namespace gipc {
inline std::string continuation_env(const char* key,const char* fallback="") {
    const char* p=std::getenv(key);return p?p:fallback;
}
inline thread_local bool continuation_hold_kappa=false;
struct ContinuationControl {
    std::string mode=continuation_env("GIPC_CONTINUATION_MODE");
    std::string load=continuation_env("GIPC_CONTINUATION_LOAD");
    std::string save=continuation_env("GIPC_CONTINUATION_SAVE");
    std::string output=continuation_env("GIPC_CONTINUATION_DIR");
    bool observe=continuation_env("GIPC_CONTINUATION_OBSERVE")=="1";
    bool fixed=continuation_env("GIPC_CONTINUATION_FIXED","1")=="1";
    bool marked=false;
    double monitor_ms=0;
    std::chrono::steady_clock::time_point tail_start;
    Json marker;
    bool enabled() const {return !mode.empty();}
    bool continues() const {return mode!="stop"&&enabled();}
    double newton_tolerance() const {
        if(mode=="strict")return .001;
        if(mode=="reference")return .0001;
        if(mode=="ultra")return .00005;
        if(mode=="ultra2")return .000025;
        return .01;
    }
    double pcg_tolerance() const {
        if(mode=="strict")return 1e-6;
        if(mode=="reference")return 1e-8;
        if(mode=="ultra"||mode=="ultra2")return 1e-10;
        return 1e-4;
    }
    ~ContinuationControl(){continuation_hold_kappa=false;}
};
}
