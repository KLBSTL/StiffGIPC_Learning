#pragma once
#include <type_traits>
// Included after the shared fixture helpers. Disabled during capture/setup.
namespace full_apply_probe {
struct NativeCostProfile {
    bool enabled=false;
    int depth=0;
    Json stages=Json::object();
    template<class Operation> decltype(auto) call(const char* name,Operation&& operation) {
        if(!enabled||depth)return operation();
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
        const auto start=Clock::now();++depth;
        try {
            if constexpr(std::is_void_v<decltype(operation())>) {
                operation();balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
                const double ms=elapsed(start);--depth;add(name,ms);return;
            } else {
                auto value=operation();balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
                const double ms=elapsed(start);--depth;add(name,ms);return value;
            }
        } catch(...) {depth=0;throw;}
    }
    void add(const char* name,double ms) {
        balanced_require(std::isfinite(ms)&&ms>=0,"Invalid diagnostic stage time");
        if(!stages.contains(name))stages[name]={{"calls",0},{"host_ms",0.}};
        auto& row=stages[name];row["calls"]=row["calls"].get<int>()+1;
        row["host_ms"]=row["host_ms"].get<double>()+ms;
    }
};
}
