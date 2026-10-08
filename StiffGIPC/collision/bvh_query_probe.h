#pragma once
#include <gipc/utils/json.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gipc
{
struct BvhQueryProbeOptions
{
    bool enabled=false;
    bool raw_records=false;
    std::vector<int> frames;
    std::string file;
    static BvhQueryProbeOptions read()
    {
        BvhQueryProbeOptions o;
        if(const char* p=std::getenv("GIPC_BVH_QUERY_PROBE")){
            const std::string v=p;
            if(v!="0"&&v!="1")throw std::runtime_error("GIPC_BVH_QUERY_PROBE must be 0 or 1");
            o.enabled=v=="1";
        }
        if(!o.enabled)return o;
        if(const char* p=std::getenv("GIPC_BVH_QUERY_PROBE_RECORDS")){
            const std::string v=p;
            if(v!="0"&&v!="1")throw std::runtime_error("GIPC_BVH_QUERY_PROBE_RECORDS must be 0 or 1");
            o.raw_records=v=="1";
        }
        const char* raw=std::getenv("GIPC_BVH_QUERY_PROBE_FRAMES");
        const char* path=std::getenv("GIPC_BVH_QUERY_PROBE_FILE");
        if(!raw||!*raw||!path||!*path)
            throw std::runtime_error("BVH query probe requires explicit frames and output file");
        o.file=path;
        const std::string list=raw;
        size_t begin=0;
        while(begin<list.size()){
            const size_t comma=list.find(',',begin);
            const std::string token=list.substr(begin,comma==std::string::npos?comma:comma-begin);
            if(token.empty()||token.find_first_not_of("0123456789")!=std::string::npos)
                throw std::runtime_error("Invalid BVH query probe frame");
            size_t used=0;const int frame=std::stoi(token,&used);
            if(used!=token.size()||frame<1||std::find(o.frames.begin(),o.frames.end(),frame)!=o.frames.end())
                throw std::runtime_error("BVH query probe frames must be unique positive integers");
            o.frames.push_back(frame);
            if(comma==std::string::npos)break;
            begin=comma+1;
            if(begin==list.size())throw std::runtime_error("Trailing comma in BVH query probe frames");
        }
        return o;
    }
    bool selected(int frame) const
    {return enabled&&std::find(frames.begin(),frames.end(),frame)!=frames.end();}
    void require_ipc(bool ipc) const
    {if(enabled&&!ipc)throw std::runtime_error("BVH query workload probe is IPC-only");}
    Json json() const
    {return {{"enabled",enabled},{"frames",frames},{"file",file},{"raw_records",raw_records},
        {"samples_per_frame_per_query_kind",1},{"production_outputs_untouched",true},
        {"scope","private_same_tree_DCD_and_FullCCD_replay"},{"performance_certified",false}};}
};
inline const BvhQueryProbeOptions& bvh_query_probe_options()
{static const BvhQueryProbeOptions options=BvhQueryProbeOptions::read();return options;}

// Node tests count actual AABB overlap calls, not memory transactions. Leaf
// overlaps precede shared/body/fixed/ID filters. FullCCD has no narrow calls.
struct BvhQueryRecord
{
    uint32_t original_id=0,node_aabb_tests=0,internal_pops=0,leaf_overlaps=0;
    uint32_t narrow_calls=0,max_stack=0,completed=0,reserved=0;
};
static_assert(sizeof(BvhQueryRecord)==32,"BVH workload record ABI");

#if defined(__CUDACC__)
#define GIPC_BVH_PROBE_HD __host__ __device__ __forceinline__
#else
#define GIPC_BVH_PROBE_HD inline
#endif
template<bool Enabled> struct BvhQueryCounter;
template<> struct BvhQueryCounter<false>
{
    GIPC_BVH_PROBE_HD BvhQueryCounter(BvhQueryRecord*,uint32_t,uint32_t* = nullptr,uint32_t = 0) {}
    GIPC_BVH_PROBE_HD void id(uint32_t) {}
    GIPC_BVH_PROBE_HD void node() {}
    GIPC_BVH_PROBE_HD void pop() {}
    GIPC_BVH_PROBE_HD void leaf(uint32_t) {}
    GIPC_BVH_PROBE_HD void narrow() {}
    GIPC_BVH_PROBE_HD void stack(uint32_t) {}
    GIPC_BVH_PROBE_HD void finish() {}
};
template<> struct BvhQueryCounter<true>
{
    BvhQueryRecord value{};BvhQueryRecord* output;uint32_t slot;
    uint32_t* leaf_trace;uint32_t trace_stride;
    GIPC_BVH_PROBE_HD BvhQueryCounter(BvhQueryRecord* p,uint32_t s,uint32_t* trace=nullptr,uint32_t stride=0)
        :output(p),slot(s),leaf_trace(trace),trace_stride(stride) {}
    GIPC_BVH_PROBE_HD void id(uint32_t id) {value.original_id=id;}
    GIPC_BVH_PROBE_HD void node() {++value.node_aabb_tests;}
    GIPC_BVH_PROBE_HD void pop() {++value.internal_pops;}
    GIPC_BVH_PROBE_HD void leaf(uint32_t id) {
        if(leaf_trace){
            if(value.leaf_overlaps<trace_stride)leaf_trace[uint64_t(slot)*trace_stride+value.leaf_overlaps]=id;
            else value.reserved=1;
        }
        ++value.leaf_overlaps;
    }
    GIPC_BVH_PROBE_HD void narrow() {++value.narrow_calls;}
    GIPC_BVH_PROBE_HD void stack(uint32_t n) {if(n>value.max_stack)value.max_stack=n;}
    GIPC_BVH_PROBE_HD void finish() {value.completed=1;output[slot]=value;}
};
#undef GIPC_BVH_PROBE_HD

inline Json bvh_query_distribution(std::vector<uint32_t> values)
{
    if(values.empty())return {{"count",0},{"sum",0},{"mean",0.0},
        {"p50",nullptr},{"p95",nullptr},{"p99",nullptr},{"max",nullptr}};
    uint64_t sum=0;for(auto n:values)sum+=n;
    std::sort(values.begin(),values.end());
    auto percentile=[&](uint64_t p){return values[(p*values.size()+99)/100-1];};
    return {{"count",values.size()},{"sum",sum},{"mean",double(sum)/values.size()},
        {"p50",percentile(50)},{"p95",percentile(95)},{"p99",percentile(99)},
        {"max",values.back()}};
}
inline Json bvh_query_record_summary(const std::vector<BvhQueryRecord>& records,bool swept)
{
    std::vector<uint32_t> nodes,pops,leaves,narrow,stack,warp_max;
    for(const auto& r:records){
        if(r.completed!=1||r.reserved!=0||r.max_stack<1||r.max_stack>65
           ||r.narrow_calls>r.leaf_overlaps||(swept&&r.narrow_calls))
            throw std::runtime_error("Incomplete or invalid BVH query workload record");
        nodes.push_back(r.node_aabb_tests);pops.push_back(r.internal_pops);
        leaves.push_back(r.leaf_overlaps);narrow.push_back(r.narrow_calls);stack.push_back(r.max_stack);
    }
    for(size_t i=0;i<records.size();i+=32){
        uint32_t peak=0;
        for(size_t j=i;j<std::min(i+32,records.size());++j)peak=std::max(peak,records[j].node_aabb_tests);
        warp_max.push_back(peak);
    }
    const auto node_distribution=bvh_query_distribution(nodes);
    const auto warp_distribution=bvh_query_distribution(warp_max);
    const double query_mean=node_distribution["mean"].get<double>();
    const double warp_mean=warp_distribution["mean"].get<double>();
    return {{"query_count",records.size()},{"node_aabb_tests",bvh_query_distribution(nodes)},
        {"internal_pops",bvh_query_distribution(pops)},{"leaf_overlaps",bvh_query_distribution(leaves)},
        {"narrow_calls",bvh_query_distribution(narrow)},{"max_pending_stack",bvh_query_distribution(stack)},
        {"launch_warp_max_node_aabb_tests",warp_distribution},{"query_node_mean",query_mean},
        {"launch_warp_max_node_mean",warp_mean},
        {"warp_max_mean_over_query_mean",query_mean>0?Json(warp_mean/query_mean):Json(nullptr)}};
}
using BvhTypedPair=std::array<int,9>;
inline std::vector<BvhTypedPair> bvh_query_typed_multiset(std::vector<BvhTypedPair> pairs,
    const std::vector<int>& ranks,const std::array<uint32_t,5>& counts)
{
    if(counts[1]!=0||uint64_t(counts[2])+counts[3]+counts[4]!=counts[0]
       ||pairs.size()!=counts[0]||ranks.size()!=pairs.size())
        throw std::runtime_error("BVH probe incomplete typed counts");
    std::array<std::vector<int>,5> typed;
    for(size_t i=0;i<pairs.size();++i){
        const int type=pairs[i][0];
        if(type<2||type>4)throw std::runtime_error("BVH probe invalid pair type");
        typed[type].push_back(ranks[i]);
    }
    for(int type=2;type<=4;++type){
        auto& values=typed[type];std::sort(values.begin(),values.end());
        if(values.size()!=counts[type])throw std::runtime_error("BVH probe type count mismatch");
        for(size_t i=0;i<values.size();++i)
            if(values[i]!=static_cast<int>(i))throw std::runtime_error("BVH probe invalid MatIndex permutation");
    }
    std::sort(pairs.begin(),pairs.end());return pairs; // retain duplicates and tuple orientation
}
int bvh_query_probe_fixture(const char* output);
} // namespace gipc
