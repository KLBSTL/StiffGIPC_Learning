#pragma once
#include <gipc/utils/json.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace gipc
{
// Eight alternating FWWF/WFFW blocks: each pair contains both paths, and
// each of the four positions contains F and W equally often.
constexpr unsigned p3_ee_timing_samples=32;
constexpr bool p3_ee_timing_split(unsigned sample)
{
    return ((sample%4==1)||(sample%4==2))!=bool((sample/4)%2);
}
struct P3EEOptions
{
    std::string mode="fused",file;
    uint32_t queue_capacity=1u<<20; // 8 MiB; exact allocation, no amortized growth
    std::vector<int> frames;
    static P3EEOptions read()
    {
        P3EEOptions o;
        if(const char* p=std::getenv("GIPC_P3_EE_MODE"))o.mode=p;
        if(o.mode!="fused"&&o.mode!="two_stage")throw std::runtime_error("P3 EE mode must be fused or two_stage");
        if(const char* p=std::getenv("GIPC_P3_EE_QUEUE_CAPACITY"))
        {
            const std::string s=p;
            if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)
                throw std::runtime_error("P3 queue capacity must be an unsigned integer");
            const auto n=std::stoull(s);
            if(!n||n>(1u<<23))throw std::runtime_error("P3 queue capacity must be in [1,8388608]");
            o.queue_capacity=static_cast<uint32_t>(n);
        }
        if(const char* p=std::getenv("GIPC_P3_EE_STUDY_FRAMES"))
        {
            const std::string s=p;size_t start=0;
            if(s.empty())throw std::runtime_error("Empty P3 study frames");
            while(start<s.size())
            {
                const auto end=s.find(',',start);const auto token=s.substr(start,end==std::string::npos?end:end-start);
                if(token.empty()||token.find_first_not_of("0123456789")!=std::string::npos)
                    throw std::runtime_error("Invalid P3 study frame");
                const auto n=std::stoull(token);
                if(!n||n>1000000||std::find(o.frames.begin(),o.frames.end(),int(n))!=o.frames.end())
                    throw std::runtime_error("P3 frames must be unique positive integers <=1000000");
                o.frames.push_back(int(n));
                if(end==std::string::npos)break;
                start=end+1;if(start==s.size())throw std::runtime_error("Trailing P3 frame comma");
            }
            const char* path=std::getenv("GIPC_P3_EE_STUDY_FILE");
            if(!path||!*path)throw std::runtime_error("P3 study needs an output path");o.file=path;
        }
        return o;
    }
    bool selected(int frame) const {return std::find(frames.begin(),frames.end(),frame)!=frames.end();}
    bool active() const {return mode!="fused"||!frames.empty();}
    void require_context(bool ipc,bool p2_active,bool contact_pool) const
    {
        if(!active())return;
        if(!ipc||p2_active)throw std::runtime_error("P3 requires IPC with all P2 increments disabled");
        if(contact_pool)throw std::runtime_error("P3 requires the original DCD query, contact pool disabled");
    }
    Json json() const {return {{"mode",mode},{"queue_capacity",queue_capacity},
        {"queue_entry_bytes",8},{"queue_budget_bytes",uint64_t(queue_capacity)*8+16},
        {"study_frames",frames},{"study_file",file},{"overflow_policy","complete_fused_before_any_typed_write"},
        {"vf_and_fullccd_unchanged",true},{"study_inputs_outputs_private",true}};}
};
inline const P3EEOptions& p3_ee_options(){static const auto o=P3EEOptions::read();return o;}
inline void p3_ee_require_edge_count(uint32_t number)
{
    // Both F and W form the signed leaf index slot + number - 1.
    // Checking number <= INT_MAX alone does not bound that sum.
    constexpr uint32_t maximum=uint32_t(std::numeric_limits<int>::max())/2+1;
    if(number>maximum)throw std::runtime_error("P3 edge count exceeds the original signed leaf-index range");
}
struct P3EEStats
{
    uint64_t dispatch_queries=0,w_queries=0,raw_pairs=0,raw_peak=0,fallbacks=0;
};
inline P3EEStats& p3_ee_stats(){static thread_local P3EEStats stats;return stats;}
inline void p3_ee_reset_stats(){p3_ee_stats()={};}
inline Json p3_ee_stats_json()
{
    const auto& s=p3_ee_stats();return {{"mode",p3_ee_options().mode},{"dispatch_queries",s.dispatch_queries},{"w_queries",s.w_queries},
        {"raw_pairs",s.raw_pairs},{"raw_peak",s.raw_peak},{"complete_fused_fallbacks",s.fallbacks},
        {"queue_budget_bytes",uint64_t(p3_ee_options().queue_capacity)*8+16}};
}
int p3_ee_fixture(const char* output);
} // namespace gipc
