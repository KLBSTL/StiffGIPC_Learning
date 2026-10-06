#pragma once
#include <cuda_tools/cuda_tools.h>
#include <solver/solve_context.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <functional>
#include <cstring>
#include <exception>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GIPC_COST_HAS_NVTX 1
#else
#define GIPC_COST_HAS_NVTX 0
#endif

namespace gipc
{
// Observational only. CUDA intervals include device idle time between host
// submissions, not just kernel execution. Nested intervals must not be summed.
// Events are never injected into a captured graph; use Nsight or the explicitly
// enabled fixed-operator probe to decompose conditional-graph work.
struct CostTraceConfig
{
    std::string path;
    std::vector<std::pair<int,int>> frames;
    bool operator_probe = false;
    bool gpu_events = true;
    CostTraceConfig()
    {
        const char* output = std::getenv("GIPC_COST_TRACE");
        if(!output || !output[0]) return;
        path = output;
        if(const char* raw=std::getenv("GIPC_COST_EVENTS"))
        {
            const std::string flag=raw;
            if(flag!="0" && flag!="1")throw std::runtime_error("GIPC_COST_EVENTS must be 0 or 1");
            gpu_events=flag=="1";
        }
        const char* raw = std::getenv("GIPC_COST_FRAMES");
        std::istringstream input(raw && raw[0] ? raw : "1-3,24-26,33-35");
        std::string item;
        while(std::getline(input,item,','))
        {
            const auto dash = item.find('-');
            auto parse = [](const std::string& s) {
                size_t used=0; const int value=std::stoi(s,&used);
                if(used!=s.size() || value<1) throw std::runtime_error("Invalid GIPC_COST_FRAMES");
                return value;
            };
            const int first=parse(item.substr(0,dash));
            const int last=dash==std::string::npos ? first : parse(item.substr(dash+1));
            if(last<first) throw std::runtime_error("Reversed GIPC_COST_FRAMES range");
            frames.emplace_back(first,last);
        }
        if(frames.empty()) throw std::runtime_error("Empty GIPC_COST_FRAMES");
        const char* probe=std::getenv("GIPC_COST_OPERATOR_PROBE");
        operator_probe=probe && std::string(probe)=="1";
    }
};
inline const CostTraceConfig& cost_trace_config()
{
    static const CostTraceConfig config;
    return config;
}
inline bool cost_trace_selected()
{
    const auto& config=cost_trace_config();
    if(config.path.empty()) return false;
    const int frame=solve_context().frame;
    for(const auto& range:config.frames)
        if(frame>=range.first && frame<=range.second) return true;
    return false;
}
struct CostTraceRecord
{
    uint64_t id=0,parent=0;
    SolveContext context;
    const char* stage=nullptr;
    const char* sample_kind="production";
    int iteration=-1,index=-1;
    double cpu_submit_ms=0;
    cudaEvent_t begin=nullptr,end=nullptr;
};
struct CostTraceState
{
    uint64_t next_id=0;
    int capture_depth=0,iteration=-1;
    const char* sample_kind="production";
    std::vector<uint64_t> stack;
    std::vector<CostTraceRecord> pending;
    std::function<Json()> persistent_preconditioner_audit;
};
inline CostTraceState& cost_trace_state()
{
    thread_local static CostTraceState state;
    return state;
}
struct CostCaptureGuard
{
    CostCaptureGuard(){++cost_trace_state().capture_depth;}
    ~CostCaptureGuard(){--cost_trace_state().capture_depth;}
};
struct CostSampleGuard
{
    const char* previous;
    explicit CostSampleGuard(const char* kind):previous(cost_trace_state().sample_kind)
    {cost_trace_state().sample_kind=kind;}
    ~CostSampleGuard(){cost_trace_state().sample_kind=previous;}
};
inline void cost_trace_iteration(int iteration)
{if(!cost_trace_config().path.empty()) cost_trace_state().iteration=iteration;}
inline void cost_trace_cuda(cudaError_t status,const char* operation)
{
    if(status!=cudaSuccess)
        throw std::runtime_error(std::string("Cost trace ")+operation+": "+cudaGetErrorString(status));
}
inline void cost_trace_destroy_events(CostTraceRecord& record) noexcept
{
    if(record.begin) cudaEventDestroy(record.begin);
    if(record.end) cudaEventDestroy(record.end);
    record.begin=nullptr; record.end=nullptr;
}
inline void cost_trace_discard() noexcept
{
    auto& state=cost_trace_state();
    for(auto& record:state.pending) cost_trace_destroy_events(record);
    state.pending.clear(); state.stack.clear();
}

class CostScope
{
    bool active=false,nvtx=false;
    CostTraceRecord record;
    int exceptions_on_entry=0;
    std::chrono::steady_clock::time_point cpu_start;
public:
    explicit CostScope(const char* stage,bool gpu=true,int index=-1)
    {
        if(!cost_trace_selected()) return;
        auto& state=cost_trace_state();
        if(state.capture_depth) return;
        exceptions_on_entry=std::uncaught_exceptions();
        active=true;
        record.id=++state.next_id;
        record.parent=state.stack.empty()?0:state.stack.back();
        record.context=solve_context();
        record.stage=stage; record.sample_kind=state.sample_kind;
        record.iteration=state.iteration; record.index=index;
        if(gpu && cost_trace_config().gpu_events)
        {
            try
            {
                cost_trace_cuda(cudaEventCreate(&record.begin),"event create begin");
                cost_trace_cuda(cudaEventCreate(&record.end),"event create end");
                cost_trace_cuda(cudaEventRecord(record.begin,cudaStreamPerThread),"event begin");
            }
            catch(...){cost_trace_destroy_events(record);throw;}
        }
        try {state.stack.push_back(record.id);}
        catch(...){cost_trace_destroy_events(record);throw;}
#if GIPC_COST_HAS_NVTX
        nvtxRangePushA(stage); nvtx=true;
#endif
        cpu_start=std::chrono::steady_clock::now();
    }
    CostScope(const CostScope&)=delete;
    CostScope& operator=(const CostScope&)=delete;
    ~CostScope() noexcept(false)
    {
        if(!active) return;
        auto& state=cost_trace_state();
        // Never launch a new tracing event or throw while a solver exception is
        // unwinding. Completed pending intervals are discarded by the flusher.
        if(std::uncaught_exceptions()>exceptions_on_entry)
        {
#if GIPC_COST_HAS_NVTX
            if(nvtx) nvtxRangePop();
#endif
            state.stack.pop_back();
            cost_trace_destroy_events(record);
            return;
        }
        record.cpu_submit_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-cpu_start).count();
        const auto status=record.end ? cudaEventRecord(record.end,cudaStreamPerThread) : cudaSuccess;
#if GIPC_COST_HAS_NVTX
        if(nvtx) nvtxRangePop();
#endif
        state.stack.pop_back();
        try
        {
            cost_trace_cuda(status,"event end");
            state.pending.push_back(record);
        }
        catch(...){cost_trace_destroy_events(record);throw;}
    }
};

inline void cost_trace_flush()
{
    if(!cost_trace_selected()) return;
    auto& state=cost_trace_state();
    if(!state.stack.empty()) throw std::runtime_error("Cost trace flush inside active scope");
    if(state.pending.empty()) return;
    // Event mode waits once at the outermost scope. NVTX/CPU-only mode adds no
    // CUDA events or stream wait; GPU activity is read independently by Nsight.
    if(cost_trace_config().gpu_events)
        cost_trace_cuda(cudaStreamSynchronize(cudaStreamPerThread),"flush stream wait");
    static std::ofstream output(cost_trace_config().path,std::ios::app);
    if(!output) throw std::runtime_error("Cannot open GIPC_COST_TRACE");
    for(auto& record:state.pending)
    {
        Json row={{"schema","gipc.cost.v1"},{"diagnostic_only",true},
            {"scope_id",record.id},{"parent_scope_id",record.parent},
            {"stage",record.stage},{"sample_kind",record.sample_kind},
            {"frame",record.context.frame},{"outer",record.context.outer},
            {"inner",record.context.inner},{"contact_model_id",record.context.contact_model_id},
            {"linear_system_id",record.context.linear_system_id},
            {"pcg_iteration",record.iteration},{"index",record.index},
            {"cpu_submit_ms",record.cpu_submit_ms},{"inclusive",true},
            {"operator_probe_enabled",cost_trace_config().operator_probe},
            {"gpu_events_enabled",cost_trace_config().gpu_events},
            {"measurement_mode",cost_trace_config().gpu_events?"cuda_events_nvtx_cpu":"nvtx_cpu_only"},
            {"nvtx_available",GIPC_COST_HAS_NVTX!=0},
            {"gpu_interval_semantics","stream elapsed including host submission gaps; nested inclusive"}};
        if(record.begin)
        {
            float elapsed=0;
            cost_trace_cuda(cudaEventElapsedTime(&elapsed,record.begin,record.end),"event elapsed time");
            row["gpu_interval_ms"]=elapsed;
            cost_trace_destroy_events(record);
        }
        else row["gpu_interval_ms"]=nullptr;
        output<<row.dump()<<'\n';
    }
    output.flush(); state.pending.clear();
    if(!output) throw std::runtime_error("Failed to persist cost trace");
}
struct CostFlushAtExit
{
    int exceptions_on_entry=std::uncaught_exceptions();
    ~CostFlushAtExit() noexcept(false)
    {
        // A linear solve can now be nested in a Newton/physical-frame trace.
        // The outermost owner flushes once, after all parent events have ended.
        // Do not discard/pop parent scopes during nested exception unwinding.
        if(!cost_trace_state().stack.empty())return;
        if(std::uncaught_exceptions()>exceptions_on_entry)
        {
            cost_trace_discard();
            return;
        }
        try {cost_trace_flush();}
        catch(...){cost_trace_discard();throw;}
    }
};
} // namespace gipc
