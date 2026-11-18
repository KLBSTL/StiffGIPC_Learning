#pragma once
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace gipc {
struct PcgReuseOptions {
    std::string mode="off", audit="A", directory, phase="P1A";
    int target_solve=2, repeats=5;
    bool decision_study() const {return phase=="DG1" || phase=="DG2";}
    bool enabled()const{return mode!="off" || !directory.empty();}
    static PcgReuseOptions read() {
        PcgReuseOptions o;
        if(const char* v=std::getenv("GIPC_PCG_REUSE"))o.mode=v;
        if(const char* v=std::getenv("GIPC_PCG_REUSE_AUDIT"))o.audit=v;
        if(const char* v=std::getenv("GIPC_PCG_REUSE_STUDY_DIR"))o.directory=v;
        if(const char* v=std::getenv("GIPC_PCG_REUSE_STUDY_PHASE"))o.phase=v;
        if(const char* v=std::getenv("GIPC_PCG_REUSE_TARGET"))o.target_solve=std::stoi(v);
        if(const char* v=std::getenv("GIPC_PCG_REUSE_REPEATS"))o.repeats=std::stoi(v);
        if(o.mode!="off"&&o.mode!="b0"&&o.mode!="w1"&&o.mode!="g1"&&o.mode!="g4")
            throw std::runtime_error("Invalid GIPC_PCG_REUSE");
        if(o.audit!="L"&&o.audit!="A")throw std::runtime_error("Invalid reuse audit mode");
        if(o.phase!="P1L"&&o.phase!="P1A"&&o.phase!="P1M"&&o.phase!="P3"&&o.phase!="D0"&&o.phase!="D3"&&o.phase!="DA"&&o.phase!="DT"&&o.phase!="DB"&&!o.decision_study())throw std::runtime_error("Invalid reuse study phase");
        if(o.target_solve<1 || o.repeats<1 || o.repeats>(o.decision_study()?16:o.phase=="DB"?14:o.phase=="DT"?12:10))
            throw std::runtime_error("Invalid reuse study target/repetitions");
        return o;
    }
};
}
