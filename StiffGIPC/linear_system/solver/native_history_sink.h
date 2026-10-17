#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gipc {
// Diagnostic host-PCG only. First 16 iterates plus a rotating tail of 112.
// No allocation or callback executes on the default null-sink path.
struct NativePcgHistory {
    static constexpr int capacity=128,first=16,limit=2000;
    std::function<void(int)> copy_slot;
    std::vector<int> slot_iterations;
    std::vector<std::array<double,3>> rows; // alpha_k, pre-update rho_k, p_k^T A p_k

    explicit NativePcgHistory(std::function<void(int)> copy)
        :copy_slot(std::move(copy)),slot_iterations(capacity,-1) {
        if(!copy_slot)throw std::runtime_error("Native history requires a copy callback");
        rows.reserve(limit);
    }
    void observe(int update,double alpha,double rho,double curvature) {
        if(update!=static_cast<int>(rows.size())||update<0||update>=limit)
            throw std::runtime_error("Native history update/index budget exceeded");
        if(!std::isfinite(alpha)||!std::isfinite(rho)||!std::isfinite(curvature)
           ||alpha<=0||rho<=0||curvature<=0)
            throw std::runtime_error("Native history invalid scalar statistics");
        const int slot=update<first?update:first+(update-first)%(capacity-first);
        copy_slot(slot); // metadata commits only after successful submission
        slot_iterations[slot]=update;rows.push_back({alpha,rho,curvature});
    }
};
inline bool native_history_study_enabled() {
    const char* value=std::getenv("GIPC_FIXED_NATIVE_HISTORY_STUDY");
    if(!value||std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::runtime_error("GIPC_FIXED_NATIVE_HISTORY_STUDY must be 0 or 1");
}
} // namespace gipc
