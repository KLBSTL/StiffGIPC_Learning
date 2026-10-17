#pragma once
#include <cmath>
#include <optional>
#include <stdexcept>

namespace gipc::research {
// Scheduling only. Acceptance remains the caller's original explicit check.
// Private CPU prototype: not connected to production or the GPU replay yet.
struct RecursiveNorm {double rr,bb;};
class WeightedCheckSchedule {
    int last_=0;
    bool cooldown_=false;
    void indices(int iteration,int limit)const {
        if(iteration<=0||last_<0||iteration<last_||limit<iteration)
            throw std::invalid_argument("Invalid weighted-check indices");
    }
public:
    static constexpr double tolerance=.0095;
    static constexpr int cooldown_period=2;
    int last_check()const{return last_;}
    bool cooling_down()const{return cooldown_;}
    bool needs_recursive_norm(int iteration,int limit)const {
        indices(iteration,limit);return !cooldown_;
    }
    bool due(int iteration,int limit,std::optional<RecursiveNorm> norm)const {
        indices(iteration,limit);
        if(cooldown_) {
            if(norm)throw std::invalid_argument("Unexpected norm during cooldown");
            return iteration==limit||iteration-last_>=cooldown_period;
        }
        if(!norm||!std::isfinite(norm->rr)||norm->rr<0||!std::isfinite(norm->bb)||norm->bb<=0)
            throw std::invalid_argument("Invalid recursive norm");
        // Preserve the original raw rr/bb expression and eight-step rule.
        return last_==0||iteration==limit||iteration-last_>=8||norm->rr<=.012*.012*norm->bb;
    }
    void observe(int iteration,int limit,double true_l2,double common_MAS,bool zero_rho_repeat=false) {
        indices(iteration,limit);
        if(iteration==last_&&!zero_rho_repeat)throw std::invalid_argument("Unexpected repeated check");
        if(!std::isfinite(true_l2)||true_l2<0||!std::isfinite(common_MAS)||common_MAS<0)
            throw std::invalid_argument("Invalid explicit quality");
        const bool next=true_l2<=tolerance&&common_MAS>tolerance;
        last_=iteration;cooldown_=next;
    }
};
}
