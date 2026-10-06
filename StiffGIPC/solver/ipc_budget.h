#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace gipc {
// Accepted-step audit only. This scalar envelope is not a physical error bound.
struct IpcBudget {
    double beta=1,unit=1,budget=1,reference=0,previous=0,alpha=0;
    bool active=false,valid=true,pending=false;
    int audits=0;
    void start(double residual,double floor) {
        if(active||!valid)return;
        if(!std::isfinite(residual)||residual<0||!std::isfinite(floor)||floor<=0){valid=false;return;}
        reference=std::max(residual,floor);previous=residual/reference;
        active=true;
    }
    void accept(double value) {
        if(!valid)return;
        // Invalid inputs disable the additional exit even before activation.
        // An ignored early NaN must never become a fresh valid budget later.
        if(pending||!std::isfinite(value)||value<0||value>1){valid=false;return;}
        if(!active)return;
        alpha=value;pending=true;
    }
    double observe(double residual,double weight) {
        if(!valid)return std::numeric_limits<double>::quiet_NaN();
        if(!std::isfinite(residual)||residual<0||!std::isfinite(weight)||weight<=0||weight>1){valid=false;return std::numeric_limits<double>::quiet_NaN();}
        if(!active)return std::numeric_limits<double>::quiet_NaN();
        const double r=residual/reference;
        if(!std::isfinite(r)){valid=false;return r;}
        if(pending){
            const double q=1-alpha,d=std::max(0.0,r-q*previous);
            beta=q*beta;unit=q*unit+d;budget=q*budget+weight*d;
            previous=r;pending=false;++audits;
            const double scale=std::max({1.0,std::abs(unit),std::abs(budget),r});
            const double eps=64*std::numeric_limits<double>::epsilon()*scale;
            if(!std::isfinite(unit)||!std::isfinite(budget)||
               std::abs(budget-(beta+weight*(unit-beta)))>eps||
               budget+eps<beta||budget+eps<weight*r||budget>unit+eps)valid=false;
        }
        return r;
    }
    bool gated(double r,double beta_tol,double residual_tol)const {
        return active&&valid&&!pending&&audits>0&&std::isfinite(r)&&beta<=beta_tol&&r<=residual_tol;
    }
    bool compensated(double r,double beta_tol,double residual_tol)const {
        return gated(r,beta_tol,residual_tol)&&budget<=beta_tol;
    }
};
}
