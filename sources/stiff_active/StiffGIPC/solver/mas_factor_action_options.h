#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc
{
// Production selection is process-constant. A protected experiment may select
// another action only BETWEEN complete solves, with both buffers prepared first.
inline int& mas_factor_action_diagnostic_override()
{
    thread_local static int mode = -1;
    return mode;
}

inline bool mas_factor_action_study_enabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("GIPC_FIXED_FACTOR_STUDY");
        return raw && std::string(raw) == "1";
    }();
    return enabled;
}

inline bool mas_factor_inverse_action()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("GIPC_MAS_FACTOR_ACTION");
        const std::string mode = raw ? raw : "factor_inverse";
        if(mode != "triangular" && mode != "factor_inverse")
            throw std::runtime_error(
                "GIPC_MAS_FACTOR_ACTION must be triangular or factor_inverse");
        return mode == "factor_inverse";
    }();
    const int override = mas_factor_action_diagnostic_override();
    return override < 0 ? enabled : override != 0;
}

inline const char* mas_factor_action_mode()
{
    return mas_factor_inverse_action() ? "factor_inverse" : "triangular";
}

inline bool mas_prepare_factor_inverse()
{
    return mas_factor_inverse_action() || mas_factor_action_study_enabled();
}

struct MasFactorActionStudyArm
{
    int saved = mas_factor_action_diagnostic_override();
    explicit MasFactorActionStudyArm(bool inverse)
    {
        if(!mas_factor_action_study_enabled())
            throw std::runtime_error(
                "MAS factor action override requires GIPC_FIXED_FACTOR_STUDY=1");
        mas_factor_action_diagnostic_override() = inverse ? 1 : 0;
    }
    ~MasFactorActionStudyArm()
    {
        mas_factor_action_diagnostic_override() = saved;
    }
    MasFactorActionStudyArm(const MasFactorActionStudyArm&) = delete;
    MasFactorActionStudyArm& operator=(const MasFactorActionStudyArm&) = delete;
};
}
