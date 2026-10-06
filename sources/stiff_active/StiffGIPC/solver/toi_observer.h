#pragma once
#include <solver/solve_context.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>

namespace gipc
{
// An opt-in observer. Synchronization is supplied by the CUDA caller so that
// configuration serialization can also be used by non-CUDA host tooling.
class ToiObserver
{
public:
    static bool outer_enabled(int frame)
    {
        const char* path=std::getenv("GIPC_TOI_OUTER_PROBE");
        if(!path||!path[0])return false;
        const char* raw=std::getenv("GIPC_TOI_OUTER_PROBE_FRAMES");
        std::stringstream ranges(raw?raw:"1-3,24-26,33-35");std::string token;
        while(std::getline(ranges,token,','))
        {
            const auto dash=token.find('-');
            const int from=std::stoi(token.substr(0,dash));
            const int to=dash==std::string::npos?from:std::stoi(token.substr(dash+1));
            if(from<1||to<from)throw std::runtime_error("Invalid outer probe range");
            if(frame>=from&&frame<=to)return true;
        }
        return false;
    }

    static void outer_event(Json entry)
    {
        attach_solve_context(entry);
        static std::ofstream log(std::getenv("GIPC_TOI_OUTER_PROBE"),std::ios::app);
        entry["scope"]="diagnostic only; complete objective and solved affine contact model";
        log<<entry.dump()<<std::endl;
        if(!log)throw std::runtime_error("Failed to write TOI outer probe");
    }

    static bool stage_enabled(int frame)
    {
        const char* path=std::getenv("GIPC_STAGE_LOG");
        const char* from=std::getenv("GIPC_STAGE_FROM_FRAME");
        return path&&path[0]&&frame>=(from?std::stoi(from):1);
    }

    template<class Synchronize>
    static void stage_event(const char* stage,Json details,int frame,Synchronize synchronize)
    {
        if(!stage_enabled(frame))return;
        synchronize();
        static std::ofstream log(std::getenv("GIPC_STAGE_LOG"),std::ios::app);
        static uint64_t sequence=0;
        attach_solve_context(details);
        details["stage"]=stage;details["frame"]=frame;details["sequence"]=++sequence;
        details["steady_seconds"]=std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        log<<details.dump()<<std::endl;
        if(!log)throw std::runtime_error("Failed to persist stage log");
    }

    static void event_progress(const char* directory,Json entry)
    {
        attach_solve_context(entry);
        std::ofstream file(std::string(directory)+"/progress.jsonl",std::ios::app);
        file<<entry.dump()<<std::endl;
        if(!file)throw std::runtime_error("Failed to persist direction event progress");
    }
};

inline void write_resolved_config(const Json& resolved)
{
    const char* path=std::getenv("GIPC_RESOLVED_CONFIG");
    if(!path||!path[0])return;
    // Repeated frames do not repeatedly write identical configuration. If a
    // caller deliberately changes a parameter, the emitted document follows.
    static std::string previous_path,previous_contents;
    const std::string contents=resolved.dump(2);
    if(previous_path==path&&previous_contents==contents)return;
    std::ofstream file(path,std::ios::trunc);
    file<<contents<<'\n';file.close();
    if(!file)throw std::runtime_error("Failed to write resolved simulation configuration");
    previous_path=path;previous_contents=contents;
}
} // namespace gipc
