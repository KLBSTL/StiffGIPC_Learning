#pragma once
#include "ipc_budget.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace gipc {

// Pure host state; no CUDA, JSON, or physical state ownership. The solve owns
// current geometry/objective and feeds already assembled nonlinear gradients.
struct IpcResidualObservation {
    double absolute=0,relative=std::numeric_limits<double>::quiet_NaN();
    double animation_rate=0,audited_alpha=0;
    bool reference_frozen_now=false,had_pending=false,audited=false;
    bool gated_ready=false,compensated_ready=false,animation_ready=false,minimum_updates_ready=false;
    bool gated_exit_allowed=false,compensated_exit_allowed=false;
};

class IpcResidualController {
  public:
    IpcResidualController(double cumulative_tol,double residual_tol,double floor,
                          int min_updates,double initial_kappa)
        : beta_tol_(cumulative_tol),residual_tol_(residual_tol),floor_(floor),
          min_updates_(min_updates),last_kappa_(initial_kappa) {
        if(!std::isfinite(beta_tol_)||beta_tol_<=0||beta_tol_>=1||
           !std::isfinite(residual_tol_)||residual_tol_<beta_tol_||
           !std::isfinite(floor_)||floor_<=0||min_updates_<1)
            throw std::runtime_error("Invalid IPC residual controller options");
    }

    IpcResidualObservation observe(double abd_inf,double fem_inf,
                                  int accepted_updates,double kappa,double animation_rate) {
        IpcResidualObservation o;
        o.animation_rate=animation_rate;
        // The two native coordinate norms remain separate in the caller log.
        // Do not silently discard nonfinite components via std::max/fmax.
        o.absolute=(!std::isfinite(abd_inf)||!std::isfinite(fem_inf)||abd_inf<0||fem_inf<0)
            ?std::numeric_limits<double>::infinity():std::max(abd_inf,fem_inf);
        if(kappa!=last_kappa_){++objective_epoch_;last_kappa_=kappa;}
        if(accepted_updates<0||accepted_updates<last_observed_updates_)budget_.valid=false;
        last_observed_updates_=accepted_updates;
        const bool was_active=budget_.active;
        // Freeze before the sixth effective accepted update, never reset on
        // objective/Kappa change or on an unfavourable residual observation.
        if(accepted_updates+1>=min_updates_)budget_.start(o.absolute,floor_);
        o.reference_frozen_now=!was_active&&budget_.active;
        if(o.reference_frozen_now){reference_update_=accepted_updates;reference_epoch_=objective_epoch_;}
        o.had_pending=budget_.pending;
        o.audited_alpha=budget_.alpha;
        const int previous_audits=budget_.audits;
        o.relative=budget_.observe(o.absolute,weight());
        o.audited=budget_.audits>previous_audits;
        o.gated_ready=budget_.gated(o.relative,beta_tol_,residual_tol_);
        o.compensated_ready=budget_.compensated(o.relative,beta_tol_,residual_tol_);
        o.animation_ready=std::isfinite(animation_rate)&&animation_rate>0.99;
        o.minimum_updates_ready=accepted_updates>=min_updates_;
        o.gated_exit_allowed=o.animation_ready&&o.minimum_updates_ready&&o.gated_ready;
        o.compensated_exit_allowed=o.animation_ready&&o.minimum_updates_ready&&o.compensated_ready;
        return o;
    }

    // Feed only the final alpha after CCD/CFL/energy backtracking and update.
    // Its residual is deliberately audited on the NEXT normal assembly.
    void accept(double final_alpha){budget_.accept(final_alpha);}
    const IpcBudget& budget()const{return budget_;}
    double weight()const{return beta_tol_/residual_tol_;}
    int objective_epoch()const{return objective_epoch_;}
    int reference_update()const{return reference_update_;}
    int reference_epoch()const{return reference_epoch_;}

    // One production selector and one test entry: movement always has priority
    // and is independent of the additional residual/animation conditions.
    static const char* exit_reason(bool movement,const std::string& mode,
                                  const IpcResidualObservation& o) {
        if(movement)return "movement";
        if(mode=="gated"&&o.gated_exit_allowed)return "residual_gated";
        if(mode=="compensated"&&o.compensated_exit_allowed)return "compensated";
        return nullptr;
    }

  private:
    IpcBudget budget_;
    double beta_tol_,residual_tol_,floor_,last_kappa_;
    int min_updates_,objective_epoch_=0,reference_update_=-1,reference_epoch_=-1;
    int last_observed_updates_=0;
};
} // namespace gipc
