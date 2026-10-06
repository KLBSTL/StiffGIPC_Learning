#pragma once
#include <gipc/utils/json.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace gipc {
struct IpcOptions {
    std::string termination="legacy";
    double cumulative_tol=.01,residual_tol=.03,residual_floor=1e-30;
    int min_updates=6,terminal_frame=0;
    bool shadow=false,cpu_audit=false;
    static IpcOptions read(double movement_tol) {
        IpcOptions o;o.cumulative_tol=movement_tol;
        if(auto v=std::getenv("GIPC_IPC_TERMINATION"))o.termination=v;
        if(o.termination!="legacy"&&o.termination!="movement_only"&&o.termination!="gated"&&o.termination!="compensated")throw std::runtime_error("Invalid IPC termination");
        auto number=[](const char* name,double fallback){auto v=std::getenv(name);if(!v)return fallback;size_t used=0;double x=std::stod(v,&used);if(used!=std::string(v).size()||!std::isfinite(x)||x<=0)throw std::runtime_error(std::string("Invalid ")+name);return x;};
        o.cumulative_tol=number("GIPC_IPC_CUMULATIVE_TOL",o.cumulative_tol);
        o.residual_tol=number("GIPC_IPC_RESIDUAL_REL_TOL",o.residual_tol);
        o.residual_floor=number("GIPC_IPC_RESIDUAL_FLOOR",o.residual_floor);
        auto integer=[](const char* name,int fallback){
            const char* v=std::getenv(name);if(!v)return fallback;
            const std::string text(v);size_t used=0;int x=std::stoi(text,&used);
            if(used!=text.size()||text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)
                throw std::runtime_error(std::string("Invalid ")+name);
            return x;
        };
        auto flag=[](const char* name,bool fallback){
            const char* v=std::getenv(name);if(!v)return fallback;
            if(std::string(v)!="0"&&std::string(v)!="1")throw std::runtime_error(std::string("Invalid ")+name);
            return std::string(v)=="1";
        };
        o.min_updates=integer("GIPC_IPC_MIN_UPDATES",o.min_updates);
        o.terminal_frame=integer("GIPC_IPC_TERMINAL_AUDIT_FRAME",o.terminal_frame);
        o.shadow=flag("GIPC_IPC_RESIDUAL_SHADOW",false);
        o.cpu_audit=flag("GIPC_IPC_RESIDUAL_CPU_AUDIT",false);
        if(o.cumulative_tol>=1||(o.residual_enabled()&&o.residual_tol<o.cumulative_tol)||o.min_updates<1||o.min_updates>10000||o.terminal_frame<0)throw std::runtime_error("Invalid IPC budget options");
        if(o.cpu_audit&&!o.residual_enabled())throw std::runtime_error("CPU residual audit requires observation");
        if(o.terminal_frame&&(!o.shadow||o.termination!="legacy"))throw std::runtime_error("Terminal audit requires legacy shadow observations");
        return o;
    }
    bool residual_enabled()const{return shadow||termination=="gated"||termination=="compensated";}
    Json json()const{return {{"termination",termination},{"cumulative_tol",cumulative_tol},{"min_updates",min_updates},
        {"residual_rel_tol",residual_tol},{"residual_floor",residual_floor},{"residual_shadow",shadow},
        {"residual_cpu_audit",cpu_audit},
        {"terminal_audit_frame",terminal_frame},{"movement_exit_retained",true},{"legacy_cumulative_retained",termination=="legacy"},
        {"residual_family","ipc_cumulative_residual"},{"residual_weight",cumulative_tol/residual_tol},
        {"residual_exit_requires_animation_finished",true},{"reference_freeze","before_minimum_effective_accepted_update"},
        {"residual_coordinate_units","native ABD generalized / FEM Cartesian; no physical error certificate"}};}
};
}
