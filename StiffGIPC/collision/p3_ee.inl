#include <filesystem>
#include <gipc/statistics.h>

namespace
{
struct P3Queue
{
    cudatool::DeviceBuffer<uint2> pairs;
    cudatool::DeviceVar<P3QueueState> state;
    void prepare(uint32_t capacity){pairs.resize(capacity);}
    size_t bytes()const{return pairs.capacity()*sizeof(uint2)+sizeof(P3QueueState);}
};
struct P3Result {uint64_t raw=0;bool fallback=false;};
void p3_fused(const lbvh_e& e,double dHat,int4* pairs,int4* ccd,uint32_t* counts,int* matrix,uint32_t capacity)
{
    gipc::p3_ee_require_edge_count(e.edge_number);
    if(e.edge_number>1)selfQuery_ee(e._bodyId,e._btype,e._vertexes,e._rest_vertexes,e._edges,
        e._bvs.data(),e._nodes.data(),pairs,ccd,counts,matrix,dHat,capacity,e.edge_number);
}
P3QueueState p3_collect(const lbvh_e& e,double dHat,P3Queue& q,uint32_t capacity)
{
    gipc::p3_ee_require_edge_count(e.edge_number);
    q.prepare(capacity);
    CUDA_SAFE_CALL(cudaMemsetAsync(q.state.data(),0,sizeof(P3QueueState),cudaStreamPerThread));
    if(e.edge_number>1)
        p3_ee_traverse<<<(e.edge_number+255)/256,256>>>(e._bodyId,e._btype,e._edges,e._bvs.data(),
            e._nodes.data(),dHat,e.edge_number,q.pairs.data(),capacity,q.state.data());
    CUDA_SAFE_CALL(cudaGetLastError());P3QueueState state{};
    CUDA_SAFE_CALL(cudaMemcpy(&state,q.state.data(),sizeof(state),cudaMemcpyDeviceToHost));
    if(state.stack_error)throw std::runtime_error("P3 traversal exceeded the original 65-entry stack");
    return state;
}
void p3_classify(const lbvh_e& e,double dHat,const P3Queue& q,uint32_t raw,
    int4* pairs,int4* ccd,uint32_t* counts,int* matrix,uint32_t capacity)
{
    if(raw)p3_ee_classify<<<(raw+255)/256,256>>>(q.pairs.data(),raw,e._edges,e._vertexes,
        e._rest_vertexes,dHat,e.edge_number,pairs,ccd,counts,matrix,capacity);
    CUDA_SAFE_CALL(cudaGetLastError());
}
P3Result p3_two_stage(const lbvh_e& e,double dHat,P3Queue& q,uint32_t raw_capacity,
    int4* pairs,int4* ccd,uint32_t* counts,int* matrix,uint32_t capacity)
{
    const auto state=p3_collect(e,dHat,q,raw_capacity);
    if(state.count>raw_capacity)
    {
        // Collection has not touched any production output/counter, including
        // an already appended VF prefix. Re-run the COMPLETE original F query.
        p3_fused(e,dHat,pairs,ccd,counts,matrix,capacity);
        return {state.count,true};
    }
    p3_classify(e,dHat,q,static_cast<uint32_t>(state.count),pairs,ccd,counts,matrix,capacity);
    return {state.count,false};
}
void p3_study(const lbvh_e& e,double dHat);
bool p3_dispatch(const lbvh_e& e,double dHat,int4* pairs,int4* ccd,uint32_t* counts,int* matrix,uint32_t capacity)
{
    const auto& o=gipc::p3_ee_options();
    if(o.selected(gipc::solve_context().frame))p3_study(e,dHat);
    ++gipc::p3_ee_stats().dispatch_queries;
    if(o.mode=="fused")return false;
    static thread_local P3Queue queue;
    const auto r=p3_two_stage(e,dHat,queue,o.queue_capacity,pairs,ccd,counts,matrix,capacity);
    auto& s=gipc::p3_ee_stats();++s.w_queries;s.raw_pairs+=r.raw;
    s.raw_peak=std::max(s.raw_peak,r.raw);s.fallbacks+=r.fallback;
    return true;
}
} // namespace
