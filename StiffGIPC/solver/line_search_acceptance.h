#pragma once
#include <algorithm>
#include <cmath>

namespace gipc
{
// The legacy condition allowed a half step at counts 0..8: nine backtracks.
inline constexpr int ipc_line_search_backtrack_budget = 9;

enum class LineSearchAction { accept, backtrack, fail };

struct LineSearchCheck
{
    LineSearchAction action;
    const char* reason;
    double threshold;
};

inline LineSearchCheck check_line_search_acceptance(double baseline,
                                                  double trial,
                                                  double slope,
                                                  double alpha,
                                                  int backtracks)
{
    if(!std::isfinite(baseline))
        return {LineSearchAction::fail, "nonfinite_baseline_energy", baseline};
    if(!std::isfinite(slope))
        return {LineSearchAction::fail, "nonfinite_armijo_slope", baseline};
    if(!std::isfinite(alpha) || alpha <= 0)
        return {LineSearchAction::fail, "invalid_alpha", baseline};
    if(backtracks < 0 || backtracks > ipc_line_search_backtrack_budget)
        return {LineSearchAction::fail, "invalid_backtrack_count", baseline};
    const double increment = slope * alpha;
    const double threshold = baseline + increment;
    if(!std::isfinite(increment) || !std::isfinite(threshold))
        return {LineSearchAction::fail, "nonfinite_acceptance_threshold", threshold};
    if(!std::isfinite(trial))
        return {LineSearchAction::fail, "nonfinite_trial_energy", threshold};
    if(trial <= threshold)
        return {LineSearchAction::accept, "accepted", threshold};
    if(backtracks == ipc_line_search_backtrack_budget)
        return {LineSearchAction::fail, "energy_backtrack_budget_exhausted", threshold};
    return {LineSearchAction::backtrack, "energy_above_threshold", threshold};
}

// A positive infinite CFL bound means no CFL restriction. Require the actual
// half step to be finite, positive and strictly smaller, including subnormals.
inline bool line_search_half_step(double alpha, double cfl_alpha, double& next)
{
    if(!std::isfinite(alpha) || alpha <= 0 || std::isnan(cfl_alpha) || cfl_alpha <= 0)
        return false;
    next = std::min(cfl_alpha, alpha * 0.5);
    return std::isfinite(next) && next > 0 && next < alpha;
}
} // namespace gipc
