#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace gipc {
struct IpcPcgDirectionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct IpcLineSearchFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct IpcPcgForcingOptions {
    std::string mode="off";
    double fixed_eta=.03;
    static constexpr double baseline_eta=.01;
    static constexpr double maximum_eta=.1;
    static constexpr double near_reference_ratio=.03;
    static IpcPcgForcingOptions read() {
        IpcPcgForcingOptions o;
        if(const char* value=std::getenv("GIPC_IPC_PCG_FORCING"))o.mode=value;
        if(o.mode!="off"&&o.mode!="monitor"&&o.mode!="fixed"&&o.mode!="adaptive")
            throw std::runtime_error("Invalid GIPC_IPC_PCG_FORCING");
        if(const char* value=std::getenv("GIPC_IPC_PCG_ETA")) {
            size_t used=0;const std::string text(value);o.fixed_eta=std::stod(text,&used);
            if(o.mode!="fixed"||used!=text.size()||!std::isfinite(o.fixed_eta)
                ||o.fixed_eta<baseline_eta||o.fixed_eta>maximum_eta)
                throw std::runtime_error("GIPC_IPC_PCG_ETA requires fixed mode and eta in [.01,.1]");
        }
        return o;
    }
    bool enabled()const{return mode!="off";}
    bool changes_precision()const{return mode=="fixed"||mode=="adaptive";}
};

struct IpcPcgForcingDecision {
    double eta=.01,residual=0,previous=0,reference=0;
    int epoch=0,accepted_update=0;
    const char* reason="disabled";
};

// Observe only the full gradient at the current accepted state. Contact-list
// changes do not reset the same IPC barrier objective; kappa/target changes do.
class IpcPcgForcingController {
    IpcPcgForcingOptions options;
    double previous=0,reference=0,kappa=0,target=0;
    int last_update=-1,epoch=0;
    bool initialized=false;
public:
    explicit IpcPcgForcingController(IpcPcgForcingOptions o):options(std::move(o)){}
    IpcPcgForcingDecision observe(double residual,int accepted_update,double new_kappa,double new_target) {
        if(!std::isfinite(residual)||residual<0||!std::isfinite(new_kappa)||!std::isfinite(new_target))
            throw std::runtime_error("Nonfinite full-gradient forcing observation");
        if(initialized&&accepted_update<=last_update)
            throw std::runtime_error("Forcing observation did not advance accepted state");
        const bool reset=!initialized||new_kappa!=kappa||new_target!=target;
        IpcPcgForcingDecision d;d.residual=residual;d.previous=previous;d.accepted_update=accepted_update;
        if(reset){++epoch;reference=residual;d.reason="objective_epoch";}
        else if(residual<=options.near_reference_ratio*reference){d.reason="near_reference";}
        else if(previous<=0||residual>=previous){d.reason="no_progress";}
        else {
            d.eta=std::clamp(.9*std::pow(residual/previous,1.5),options.baseline_eta,options.maximum_eta);
            d.reason="progress";
        }
        if(options.mode=="fixed"){d.eta=options.fixed_eta;d.reason="fixed";}
        if(!options.changes_precision()){d.eta=options.baseline_eta;d.reason="monitor";}
        d.reference=reference;d.epoch=epoch;
        previous=residual;kappa=new_kappa;target=new_target;last_update=accepted_update;initialized=true;
        return d;
    }
};
}
