#include <collision/bvh_query_probe.h>
#include <iostream>
#include <type_traits>

namespace
{
int checks=0;
void require(bool value,const char* message)
{++checks;if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message)
{bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,message);}
void env(const char* key,const char* value)
{
#if defined(_WIN32)
    _putenv_s(key,value?value:"");
#else
    if(value)setenv(key,value,1);else unsetenv(key);
#endif
}
void clear()
{for(const char* key:{"GIPC_BVH_QUERY_PROBE","GIPC_BVH_QUERY_PROBE_FRAMES","GIPC_BVH_QUERY_PROBE_FILE","GIPC_BVH_QUERY_PROBE_RECORDS"})env(key,nullptr);}
}
int main()
{
    try{
        clear();auto disabled=gipc::BvhQueryProbeOptions::read();
        require(!disabled.enabled&&!disabled.selected(1)&&disabled.frames.empty()&&disabled.file.empty(),"probe default off");
        env("GIPC_BVH_QUERY_PROBE","bad");rejects([]{gipc::BvhQueryProbeOptions::read();},"invalid enable rejected");
        env("GIPC_BVH_QUERY_PROBE","1");rejects([]{gipc::BvhQueryProbeOptions::read();},"unbounded missing selection rejected");
        env("GIPC_BVH_QUERY_PROBE_FILE","private.jsonl");env("GIPC_BVH_QUERY_PROBE_FRAMES","1,41,57");
        auto options=gipc::BvhQueryProbeOptions::read();
        require(options.selected(1)&&options.selected(41)&&options.selected(57)&&!options.selected(2)&&!options.raw_records,"explicit selection and default summary");
        env("GIPC_BVH_QUERY_PROBE_RECORDS","1");require(gipc::BvhQueryProbeOptions::read().raw_records,"explicit raw records");
        env("GIPC_BVH_QUERY_PROBE_RECORDS","bad");rejects([]{gipc::BvhQueryProbeOptions::read();},"invalid record flag rejected");
        env("GIPC_BVH_QUERY_PROBE_RECORDS","0");
        rejects([&]{options.require_ipc(false);},"TOI selection rejected");
        for(const char* value:{"0","-1","1,1","1,","1,,2"," 1","1.0","2147483648"}){
            env("GIPC_BVH_QUERY_PROBE_FRAMES",value);rejects([]{gipc::BvhQueryProbeOptions::read();},"invalid frames rejected");}
        clear();
        require(std::is_empty<gipc::BvhQueryCounter<false>>::value,"off counter has no state");
        gipc::BvhQueryCounter<false> off(nullptr,123);off.id(3);off.node();off.pop();off.leaf(0);off.narrow();off.stack(65);off.finish();
        gipc::BvhQueryRecord records[2]{};gipc::BvhQueryCounter<true> on(records,1);
        on.id(42);on.stack(1);on.node();on.pop();on.node();on.leaf(0);on.narrow();on.stack(3);on.stack(2);on.finish();
        require(records[0].completed==0&&records[1].completed==1&&records[1].original_id==42,"exact slot and original ID");
        require(records[1].node_aabb_tests==2&&records[1].internal_pops==1&&records[1].leaf_overlaps==1
            &&records[1].narrow_calls==1&&records[1].max_stack==3,"local counter values");
        uint32_t leaf_trace[4]={};gipc::BvhQueryRecord traced{};
        gipc::BvhQueryCounter<true> trace(&traced,0,leaf_trace,4);
        trace.stack(1);for(uint32_t id:{2u,3u,0u,1u})trace.leaf(id);trace.finish();
        require(leaf_trace[0]==2&&leaf_trace[1]==3&&leaf_trace[2]==0&&leaf_trace[3]==1
            &&traced.leaf_overlaps==4&&traced.reserved==0,"exact diagnostic leaf trace");
        auto empty=gipc::bvh_query_record_summary({},false);
        require(empty["query_count"]==0&&empty["max_pending_stack"]["max"].is_null(),"empty queries are not fabricated");
        auto summary=gipc::bvh_query_record_summary({records[1]},false);
        require(summary["node_aabb_tests"]["p99"]==2&&summary["max_pending_stack"]["max"]==3,"record summary");
        rejects([&]{gipc::bvh_query_record_summary({records[0]},false);},"unwritten record rejected");
        rejects([&]{gipc::bvh_query_record_summary({records[1]},true);},"FullCCD is not narrow phase");
        auto invalid=records[1];invalid.max_stack=66;
        rejects([&]{gipc::bvh_query_record_summary({invalid},false);},"invalid stack record rejected");
        std::vector<uint32_t> values;for(uint32_t i=1;i<=100;++i)values.push_back(i);
        auto distribution=gipc::bvh_query_distribution(values);
        require(distribution["sum"]==5050&&distribution["p50"]==50&&distribution["p95"]==95
            &&distribution["p99"]==99&&distribution["max"]==100,"nearest rank quantiles");
        auto wide=gipc::bvh_query_distribution({0xffffffffu,0xffffffffu});
        require(wide["sum"]==uint64_t(0x1fffffffeULL),"64 bit sums");
        std::vector<gipc::BvhQueryRecord> warp_records(33,records[1]);
        for(size_t i=0;i<warp_records.size();++i)warp_records[i].node_aabb_tests=uint32_t(i+1);
        auto warp_summary=gipc::bvh_query_record_summary(warp_records,false);
        require(warp_summary["launch_warp_max_node_aabb_tests"]["count"]==2
            &&warp_summary["launch_warp_max_node_mean"]==32.5&&warp_summary["query_node_mean"]==17.0,
            "launch warp tails include final partial warp");
        gipc::BvhTypedPair a={2,-4,0,-1,-1,-4,0,1,2},b={3,-5,1,2,-1,-5,0,1,2};
        const std::array<uint32_t,5> counts={3,0,2,1,0};
        auto first=gipc::bvh_query_typed_multiset({a,b,a},{1,0,0},counts);
        auto second=gipc::bvh_query_typed_multiset({a,a,b},{0,1,0},counts);
        require(first==second&&first.size()==3&&first[0]==first[1],"atomic order ignored, duplicates retained");
        rejects([&]{gipc::bvh_query_typed_multiset({a,b,a},{0,0,0},counts);},"duplicate MatIndex rejected");
        auto bad_counts=counts;bad_counts[1]=1;
        rejects([&]{gipc::bvh_query_typed_multiset({a,b,a},{0,0,1},bad_counts);},"incomplete counts rejected");
        std::cout<<"{\"passed\":true,\"checks\":"<<checks<<"}"<<std::endl;return 0;
    }catch(const std::exception& error){clear();std::cerr<<error.what()<<std::endl;return 1;}
}
