#pragma once
#include <cstdint>
#include <gipc/utils/json.h>

namespace gipc
{
// Host-side observational identity only; never used by numerical decisions.
// Zero means that no contact model / linear system has been formed yet.
struct SolveContext
{
    int frame = -1;
    int outer = -1;
    int inner = -1;
    uint64_t contact_model_id = 0;
    uint64_t linear_system_id = 0;
};

inline SolveContext& solve_context()
{
    thread_local static SolveContext context;
    return context;
}

inline void set_solve_frame(int frame)
{
    auto& context = solve_context();
    context = SolveContext{};
    context.frame = frame;
}

inline void set_solve_outer(int outer)
{
    thread_local static uint64_t next_model_id = 0;
    auto& context = solve_context();
    context.outer = outer;
    context.inner = -1;
    context.contact_model_id = ++next_model_id;
    context.linear_system_id = 0;
}

inline void set_solve_inner(int inner)
{
    auto& context = solve_context();
    context.inner = inner;
    context.linear_system_id = 0;
}

// Call exactly once at the production global linear-system solve entrance.
// In-process fixed-system diagnostic replays retain the original system ID.
inline uint64_t begin_linear_system(int frame)
{
    thread_local static uint64_t next_system_id = 0;
    if(solve_context().frame != frame) set_solve_frame(frame);
    return solve_context().linear_system_id = ++next_system_id;
}

inline void attach_solve_context(Json& record)
{
    const auto& context = solve_context();
    record["frame"] = context.frame;
    record["outer"] = context.outer;
    record["inner"] = context.inner;
    record["contact_model_id"] = context.contact_model_id;
    record["linear_system_id"] = context.linear_system_id;
}
} // namespace gipc
