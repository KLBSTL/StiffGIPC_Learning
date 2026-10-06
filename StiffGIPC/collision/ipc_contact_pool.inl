// Included only by mlbvh.cu, after the unchanged raw query helpers.
// All buffers belong to this host thread and use the existing per-thread CUDA
// stream. No borrowed tree storage is retained across a storage selection.
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
using PoolIdentity=gipc::IpcContactPoolIdentity;
struct PoolOutput
{
    cudatool::DeviceBuffer<int4> dcd,ccd;
    cudatool::DeviceBuffer<int> indices;
    cudatool::DeviceBuffer<uint32_t> device_counts;
    std::array<uint32_t,5> counts{};
    std::vector<std::array<int,9>> records;
};
struct ContactPoolStats
{
    uint64_t attempts=0,reused_queries=0,old_queries=0,prepare_calls=0,capture_passes=0,generations=0;
    uint64_t validation_calls=0,validation_passed=0,validation_failed=0,pairs_compared=0,nonempty_validation_calls=0;
    uint64_t pool_bytes_peak=0,pool_pairs=0,vf_pairs=0,ee_pairs=0,overflow_retries=0;
    uint64_t nonempty_reused_queries=0,validation_peak_capacity=0,production_overflow_queries=0;
    double diagnostic_host_ms=0;
    std::map<std::string,uint64_t> fallback_reasons;
};
struct ContactPoolState
{
    bool active=false,sealed=false,trial_set=false,reference_valid=false;
    bool fixture=false,validate=false;
    uint32_t serial=0,capacity=0,count=0,vertices=0,faces=0,edges=0,surface=0;
    uint64_t generation=0;
    double max_alpha=0,trial_alpha=0,dhat=0;
    lbvh_f* face_tree=nullptr;lbvh_e* edge_tree=nullptr;
    const double3* direction=nullptr;
    double3* positions=nullptr;double3* rest=nullptr;
    int* body=nullptr;int* btype=nullptr;
    uint3* face_map=nullptr;uint2* edge_map=nullptr;uint32_t* surface_map=nullptr;
    cudatool::DeviceBuffer<PoolIdentity> identities;
    cudatool::DeviceBuffer<double3> saved_direction,saved_rest;
    cudatool::DeviceBuffer<int> saved_body,saved_btype;
    cudatool::DeviceBuffer<uint3> saved_faces;
    cudatool::DeviceBuffer<uint2> saved_edges;
    cudatool::DeviceBuffer<uint32_t> saved_surface,guard;
    cudatool::DeviceBuffer<AABB> swept_faces,swept_edges,swept_points,current_faces,current_edges;
    PoolOutput reference,candidate;
};
ContactPoolState& cp_state(){static thread_local ContactPoolState value;return value;}
ContactPoolStats& cp_stats(){static thread_local ContactPoolStats value;return value;}

template<class T> void cp_snapshot(cudatool::DeviceBuffer<T>& dst,const T* src,size_t count)
{
    dst.resize_discard(count);
    if(count)CUDA_SAFE_CALL(cudaMemcpyAsync(dst.data(),src,count*sizeof(T),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
}
__device__ bool cp_finite3(double3 p){return isfinite(p.x)&&isfinite(p.y)&&isfinite(p.z);}
__device__ bool cp_valid_box(const AABB& b)
{return cp_finite3(b.lower)&&cp_finite3(b.upper)&&b.lower.x<=b.upper.x&&b.lower.y<=b.upper.y&&b.lower.z<=b.upper.z;}
__device__ bool cp_contains(const AABB& outer,const AABB& inner)
{
    return cp_valid_box(outer)&&cp_valid_box(inner)
        &&outer.lower.x<=inner.lower.x&&outer.lower.y<=inner.lower.y&&outer.lower.z<=inner.lower.z
        &&outer.upper.x>=inner.upper.x&&outer.upper.y>=inner.upper.y&&outer.upper.z>=inner.upper.z;
}
__device__ bool cp_same3(double3 a,double3 b)
{return __double_as_longlong(a.x)==__double_as_longlong(b.x)
    &&__double_as_longlong(a.y)==__double_as_longlong(b.y)&&__double_as_longlong(a.z)==__double_as_longlong(b.z);}

// One bit per independent reason. No production output is touched by guards.
enum PoolGuard : uint32_t { CP_ATTRIBUTE=1,CP_MAPPING=2,CP_NONFINITE=4,CP_CONTAINMENT=8,CP_IDENTITY=16,CP_DIRECTION=32 };
__global__ void cp_cache_points(const double3* positions,const double3* direction,uint32_t n,
    double alpha,AABB* bounds,uint32_t* error)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double3 x=positions[i],d=direction[i];
    AABB b;b.lower=b.upper=x;b.combines(x.x-d.x*alpha,x.y-d.y*alpha,x.z-d.z*alpha);
    bounds[i]=b;if(!cp_finite3(x)||!cp_finite3(d)||!cp_valid_box(b))atomicOr(error,CP_NONFINITE);
}
template<class Element>
__global__ void cp_cache_leaves(const Node* nodes,const AABB* bounds,const Element* elements,
    uint32_t n,uint32_t vertices,AABB* original,uint32_t* error)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const uint32_t leaf=n-1+i,id=nodes[leaf].element_idx;
    if(id>=n){atomicOr(error,CP_MAPPING);return;}
    const auto e=elements[id];bool valid=e.x<vertices&&e.y<vertices;
    if constexpr(std::is_same_v<Element,uint3>)valid=valid&&e.z<vertices;
    if(!valid){atomicOr(error,CP_MAPPING);return;}
    original[id]=bounds[leaf];if(!cp_valid_box(bounds[leaf]))atomicOr(error,CP_NONFINITE);
}
__global__ void cp_check_identities(const PoolIdentity* identities,uint32_t n,uint32_t epoch,
    uint32_t vertices,uint32_t faces,uint32_t edges,uint32_t* counters)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto p=i<n?identities[i]:PoolIdentity{0,0,2,0};bool valid=p.epoch==epoch;
    if(p.kind==0)valid=valid&&p.first<vertices&&p.second<faces;
    else if(p.kind==1)valid=valid&&p.first<edges&&p.second<edges&&p.first<p.second;
    else valid=false;
    if(i<n&&!valid)atomicOr(counters,CP_IDENTITY);
    // All lanes, including the last block's inactive entries, participate.
    const unsigned vf=__ballot_sync(0xffffffffu,i<n&&p.kind==0);
    const unsigned ee=__ballot_sync(0xffffffffu,i<n&&p.kind==1);
    if((threadIdx.x&31)==0)
    {if(vf)atomicAdd(counters+1,__popc(vf));if(ee)atomicAdd(counters+2,__popc(ee));}
}
__global__ void cp_guard_vertices(const double3* positions,const double3* direction,const double3* saved_direction,
    const double3* rest,const double3* saved_rest,const int* body,const int* saved_body,
    const int* btype,const int* saved_btype,const AABB* swept,uint32_t n,uint32_t* error)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(body[i]!=saved_body[i]||btype[i]!=saved_btype[i])atomicOr(error,CP_ATTRIBUTE);
    if(!cp_same3(direction[i],saved_direction[i]))atomicOr(error,CP_DIRECTION);
    if(!cp_same3(rest[i],saved_rest[i]))atomicOr(error,CP_ATTRIBUTE);
    if(!cp_finite3(rest[i]))atomicOr(error,CP_NONFINITE);
    const double3 x=positions[i];AABB b;b.lower=b.upper=x;
    if(!cp_finite3(x))atomicOr(error,CP_NONFINITE);
    else if(!cp_contains(swept[i],b))atomicOr(error,CP_CONTAINMENT);
}
template<class Element>
__global__ void cp_guard_leaves(const Node* nodes,const AABB* bounds,const Element* elements,
    const Element* saved,const AABB* swept,uint32_t n,AABB* current,uint32_t* error)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const uint32_t leaf=n-1+i,id=nodes[leaf].element_idx;
    if(id>=n){atomicOr(error,CP_MAPPING);return;}
    const auto a=elements[id],b=saved[id];bool same=a.x==b.x&&a.y==b.y;
    if constexpr(std::is_same_v<Element,uint3>)same=same&&a.z==b.z;
    if(!same)atomicOr(error,CP_MAPPING);
    const auto box=bounds[leaf];current[id]=box;
    if(!cp_valid_box(box))atomicOr(error,CP_NONFINITE);
    else if(!cp_contains(swept[id],box))atomicOr(error,CP_CONTAINMENT);
}
__global__ void cp_guard_surface(const uint32_t* current,const uint32_t* saved,uint32_t n,uint32_t vertices,uint32_t* error)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&(current[i]!=saved[i]||current[i]>=vertices))atomicOr(error,CP_MAPPING);
}
__global__ void cp_classify(const PoolIdentity* identities,uint32_t n,const double3* positions,const double3* rest,
    const uint3* faces,const uint2* edges,const AABB* face_bounds,const AABB* edge_bounds,
    double dhat,uint32_t edges_count,int4* dcd,int4* ccd,uint32_t* counts,int* indices,uint32_t capacity)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto p=identities[i];const double gap=sqrt(dhat);
    if(p.kind==0)
    {
        const auto f=faces[p.second];AABB b;b.upper=b.lower=positions[p.first];
        // Repeat the original instantaneous overlap as well as the unchanged
        // narrow-phase helper: no extra swept-only tuple may reach the barrier.
        if(overlap(b,face_bounds[p.second],gap))
            _checkPTintersection(positions,p.first,f.x,f.y,f.z,dhat,counts,indices,dcd,ccd,capacity);
    }
    else
    {
        const auto a=edges[p.first],b=edges[p.second];
        if(overlap(edge_bounds[p.first],edge_bounds[p.second],gap))
            _checkEEintersection(positions,rest,a.x,a.y,b.x,b.y,p.second,dhat,counts,indices,dcd,ccd,edges_count,capacity);
    }
}
inline unsigned cp_blocks(uint32_t n){return (n-1)/256+1;}
void cp_launch_classify(ContactPoolState& s,int4* dcd,int4* ccd,uint32_t* counts,int* indices,uint32_t capacity)
{
    if(s.count)cp_classify<<<cp_blocks(s.count),256>>>(s.identities.data(),s.count,s.positions,s.rest,
        s.face_map,s.edge_map,s.current_faces.data(),s.current_edges.data(),s.dhat,s.edges,dcd,ccd,counts,indices,capacity);
}
uint64_t cp_bytes(const ContactPoolState& s)
{
    return s.identities.capacity()*sizeof(PoolIdentity)
        +(s.saved_direction.capacity()+s.saved_rest.capacity())*sizeof(double3)
        +(s.saved_body.capacity()+s.saved_btype.capacity())*sizeof(int)
        +s.saved_faces.capacity()*sizeof(uint3)+s.saved_edges.capacity()*sizeof(uint2)
        +(s.saved_surface.capacity()+s.guard.capacity())*sizeof(uint32_t)
        +(s.swept_faces.capacity()+s.swept_edges.capacity()+s.swept_points.capacity()
          +s.current_faces.capacity()+s.current_edges.capacity())*sizeof(AABB);
}
void cp_begin_impl(lbvh_f& f,lbvh_e& e,const double3* direction,uint32_t vertices,
    double alpha,double dhat,uint64_t generation,bool validate,bool fixture=false)
{
    auto& s=cp_state();s.active=false;s.sealed=false;s.reference_valid=false;s.trial_set=false;
    s.fixture=fixture;s.validate=validate;s.capacity=0;s.count=0;
    ++cp_stats().generations;
    if(!std::isfinite(alpha)||alpha<0||!std::isfinite(dhat)||dhat<=0)
    {++cp_stats().fallback_reasons["invalid_source_interval"];return;}
    if(f._vertexes!=e._vertexes||f._bodyId!=e._bodyId||f._btype!=e._btype
        ||(vertices&&(!direction||!f._vertexes||!f._bodyId||!f._btype||!e._rest_vertexes)))
    {++cp_stats().fallback_reasons["source_identity"];return;}
    s.face_tree=&f;s.edge_tree=&e;s.direction=direction;s.positions=f._vertexes;s.rest=e._rest_vertexes;
    s.body=f._bodyId;s.btype=f._btype;s.face_map=f._faces;s.edge_map=e._edges;s.surface_map=f._surfVerts;
    s.vertices=vertices;s.faces=f.face_number;s.edges=e.edge_number;s.surface=f.vert_number;
    s.max_alpha=alpha;s.dhat=dhat;s.generation=generation;
    if(++s.serial==0)++s.serial; // all live identity entries are rewritten per pass
    s.active=true;
}
void cp_read_guard(std::array<uint32_t,3>& result)
{CUDA_SAFE_CALL(cudaMemcpy(result.data(),cp_state().guard.data(),sizeof(result),cudaMemcpyDeviceToHost));}
bool cp_fallback(const char* reason)
{++cp_stats().old_queries;++cp_stats().fallback_reasons[reason];cp_state().reference_valid=false;return false;}

void cp_snapshot_query(PoolOutput& out,bool candidate)
{
    auto& s=cp_state();auto& stats=cp_stats();uint32_t capacity=0;
    out.records.clear();out.device_counts.resize_discard(5);
    for(;;)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(out.device_counts.data(),0,5*sizeof(uint32_t),cudaStreamPerThread));
        if(capacity)
        {
            CUDA_SAFE_CALL(cudaMemsetAsync(out.dcd.data(),0xa5,capacity*sizeof(int4),cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(out.ccd.data(),0xa5,capacity*sizeof(int4),cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(out.indices.data(),0xa5,capacity*sizeof(int),cudaStreamPerThread));
        }
        if(candidate)cp_launch_classify(s,out.dcd.data(),out.ccd.data(),out.device_counts.data(),out.indices.data(),capacity);
        else
        {
            const auto& f=*s.face_tree;const auto& e=*s.edge_tree;
            if(s.faces&&s.surface)selfQuery_vf(s.body,s.btype,s.positions,s.face_map,s.surface_map,
                f._bvs.data(),f._nodes.data(),out.dcd.data(),out.ccd.data(),out.device_counts.data(),out.indices.data(),s.dhat,capacity,s.surface);
            if(s.edges>1)selfQuery_ee(s.body,s.btype,s.positions,s.rest,s.edge_map,
                e._bvs.data(),e._nodes.data(),out.dcd.data(),out.ccd.data(),out.device_counts.data(),out.indices.data(),s.dhat,capacity,s.edges);
        }
        CUDA_SAFE_CALL(cudaMemcpy(out.counts.data(),out.device_counts.data(),sizeof(out.counts),cudaMemcpyDeviceToHost));
        if(out.counts[0]<=capacity)break;
        if(out.counts[0]>uint32_t(std::numeric_limits<int>::max()))throw std::runtime_error("contact pool diagnostic pair count overflow");
        capacity=out.counts[0];out.dcd.resize_discard(capacity);out.ccd.resize_discard(capacity);out.indices.resize_discard(capacity);
        ++stats.overflow_retries;stats.validation_peak_capacity=std::max<uint64_t>(stats.validation_peak_capacity,capacity);
    }
    const auto& c=out.counts;
    if(c[1]||uint64_t(c[2])+c[3]+c[4]!=c[0])throw std::runtime_error("contact pool typed count contract");
    std::vector<int4> hd(c[0]),hc(c[0]);std::vector<int> hi(c[0]);
    if(c[0])
    {
        CUDA_SAFE_CALL(cudaMemcpy(hd.data(),out.dcd.data(),hd.size()*sizeof(int4),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hc.data(),out.ccd.data(),hc.size()*sizeof(int4),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hi.data(),out.indices.data(),hi.size()*sizeof(int),cudaMemcpyDeviceToHost));
    }
    std::array<std::vector<int>,5> ranks;
    for(size_t i=0;i<hd.size();++i)
    {
        const auto p=hd[i],q=hc[i];const int type=(p.x>=0||p.y<0)?4:(p.z<0?2:(p.w<0?3:4));
        ranks[type].push_back(hi[i]);out.records.push_back({type,p.x,p.y,p.z,p.w,q.x,q.y,q.z,q.w});
    }
    for(int type=2;type<=4;++type)
    {
        auto& r=ranks[type];std::sort(r.begin(),r.end());
        if(r.size()!=c[type])throw std::runtime_error("contact pool tuple/type count mismatch");
        for(size_t i=0;i<r.size();++i)if(r[i]!=static_cast<int>(i))throw std::runtime_error("contact pool MatIndex not a type-local bijection");
    }
    std::sort(out.records.begin(),out.records.end()); // signed tuples and multiplicity intact
}
void cp_validate()
{
    auto& s=cp_state();auto& stats=cp_stats();++stats.validation_calls;
    const auto start=std::chrono::steady_clock::now();gipc::CostSampleGuard sample("diagnostic_contact_pool");
    gipc::CostScope scope("diagnostic.contact_pool");
    try
    {
        cp_snapshot_query(s.reference,false);cp_snapshot_query(s.candidate,true);
        if(s.reference.counts!=s.candidate.counts||s.reference.records!=s.candidate.records)
            throw std::runtime_error("contact pool differs from legacy typed DCD/CCD multiset");
        ++stats.validation_passed;stats.pairs_compared+=s.reference.counts[0];
        if(s.reference.counts[0])++stats.nonempty_validation_calls;
        s.reference_valid=true;
    }
    catch(...){++stats.validation_failed;s.reference_valid=false;throw;}
    stats.diagnostic_host_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
} // anonymous namespace

namespace gipc
{
void ipc_contact_pool_begin(lbvh_f& f,lbvh_e& e,const double3* direction,uint32_t vertices,
    double max_alpha,double dhat,uint64_t generation)
{
    if(!ipc_contact_pool_config().enabled)return;
    cp_begin_impl(f,e,direction,vertices,max_alpha,dhat,generation,ipc_contact_pool_config().validate);
}
IpcContactPoolIdentity* ipc_contact_pool_capture_pass(uint32_t capacity)
{
    auto& s=cp_state();if(!s.active)return nullptr;
    CostScope scope("collision.contact_pool.prepare");++cp_stats().prepare_calls;++cp_stats().capture_passes;
    s.sealed=false;s.reference_valid=false;s.capacity=capacity;
    // Allocate one marker even for capacity=0 so count-only passes still select
    // the capture kernel. No zero-capacity slot is written.
    s.identities.resize_discard(std::max<uint32_t>(capacity,1));
    CUDA_SAFE_CALL(cudaMemsetAsync(s.identities.data(),0xff,s.identities.size()*sizeof(PoolIdentity),cudaStreamPerThread));
    cp_stats().pool_bytes_peak=std::max(cp_stats().pool_bytes_peak,cp_bytes(s));return s.identities.data();
}
uint32_t ipc_contact_pool_capture_epoch(){return cp_state().serial;}
void ipc_contact_pool_seal(uint32_t count)
{
    auto& s=cp_state();if(!s.active)return;
    CostScope scope("collision.contact_pool.prepare");++cp_stats().prepare_calls;
    if(count>s.capacity)throw std::runtime_error("contact pool sealed an overflowed capture pass");
    s.count=count;s.guard.resize_discard(3);
    CUDA_SAFE_CALL(cudaMemsetAsync(s.guard.data(),0,3*sizeof(uint32_t),cudaStreamPerThread));
    cp_snapshot(s.saved_direction,s.direction,s.vertices);cp_snapshot(s.saved_rest,s.rest,s.vertices);
    cp_snapshot(s.saved_body,s.body,s.vertices);cp_snapshot(s.saved_btype,s.btype,s.vertices);
    cp_snapshot(s.saved_faces,s.face_map,s.faces);cp_snapshot(s.saved_edges,s.edge_map,s.edges);
    cp_snapshot(s.saved_surface,s.surface_map,s.surface);
    s.swept_points.resize_discard(s.vertices);s.swept_faces.resize_discard(s.faces);s.swept_edges.resize_discard(s.edges);
    s.current_faces.resize_discard(s.faces);s.current_edges.resize_discard(s.edges);
    if(s.vertices)cp_cache_points<<<cp_blocks(s.vertices),256>>>(s.positions,s.direction,s.vertices,s.max_alpha,s.swept_points.data(),s.guard.data());
    if(s.faces)cp_cache_leaves<<<cp_blocks(s.faces),256>>>(s.face_tree->_nodes.data(),s.face_tree->_bvs.data(),s.face_map,s.faces,s.vertices,s.swept_faces.data(),s.guard.data());
    if(s.edges)cp_cache_leaves<<<cp_blocks(s.edges),256>>>(s.edge_tree->_nodes.data(),s.edge_tree->_bvs.data(),s.edge_map,s.edges,s.vertices,s.swept_edges.data(),s.guard.data());
    if(count)cp_check_identities<<<cp_blocks(count),256>>>(s.identities.data(),count,s.serial,s.vertices,s.faces,s.edges,s.guard.data());
    std::array<uint32_t,3> guard{};cp_read_guard(guard);
    if(guard[0]&CP_IDENTITY)throw std::runtime_error("contact pool incomplete or stale identity capture");
    if(guard[0]){s.active=false;++cp_stats().fallback_reasons["invalid_source_bounds"];return;}
    s.sealed=true;cp_stats().pool_pairs+=count;cp_stats().vf_pairs+=guard[1];cp_stats().ee_pairs+=guard[2];
    cp_stats().pool_bytes_peak=std::max(cp_stats().pool_bytes_peak,cp_bytes(s));
}
void ipc_contact_pool_set_trial(double alpha)
{auto& s=cp_state();s.reference_valid=false;s.trial_alpha=alpha;s.trial_set=s.active&&s.sealed;}
bool ipc_contact_pool_try_discrete(lbvh_f& f,lbvh_e& e,double dhat,int4* dcd,int4* ccd,
    uint32_t* counts,int* indices,uint32_t capacity)
{
    auto& s=cp_state();if(!s.fixture&&!ipc_contact_pool_config().enabled)return false;
    auto& stats=cp_stats();++stats.attempts;s.reference_valid=false;
    if(!s.active||!s.sealed||!s.trial_set)return cp_fallback("no_sealed_trial");
    if(!std::isfinite(s.trial_alpha)||s.trial_alpha<0||s.trial_alpha>s.max_alpha)return cp_fallback("outside_interval");
    if(dhat!=s.dhat)return cp_fallback("barrier_radius");
    if(&f!=s.face_tree||&e!=s.edge_tree||f.face_number!=s.faces||e.edge_number!=s.edges||f.vert_number!=s.surface
        ||f._vertexes!=s.positions||e._vertexes!=s.positions||e._rest_vertexes!=s.rest
        ||f._bodyId!=s.body||e._bodyId!=s.body||f._btype!=s.btype||e._btype!=s.btype
        ||f._faces!=s.face_map||e._edges!=s.edge_map||f._surfVerts!=s.surface_map)
        return cp_fallback("source_identity");
    // Root retains ordinary builds. Select their public storage before guards,
    // exactly as SelfCollitionDetect would; this is a host ownership swap only.
    if(!s.fixture)
    {if(s.faces)discrete_select_query_storage(f,false);if(s.edges)discrete_select_query_storage(e,false);}
    uint32_t errors=0;
    {
        CostScope scope("collision.contact_pool.guard");
        CUDA_SAFE_CALL(cudaMemsetAsync(s.guard.data(),0,3*sizeof(uint32_t),cudaStreamPerThread));
        if(s.vertices)cp_guard_vertices<<<cp_blocks(s.vertices),256>>>(s.positions,s.direction,s.saved_direction.data(),
            s.rest,s.saved_rest.data(),s.body,s.saved_body.data(),s.btype,s.saved_btype.data(),s.swept_points.data(),s.vertices,s.guard.data());
        if(s.faces)cp_guard_leaves<<<cp_blocks(s.faces),256>>>(f._nodes.data(),f._bvs.data(),s.face_map,s.saved_faces.data(),s.swept_faces.data(),s.faces,s.current_faces.data(),s.guard.data());
        if(s.edges)cp_guard_leaves<<<cp_blocks(s.edges),256>>>(e._nodes.data(),e._bvs.data(),s.edge_map,s.saved_edges.data(),s.swept_edges.data(),s.edges,s.current_edges.data(),s.guard.data());
        if(s.surface)cp_guard_surface<<<cp_blocks(s.surface),256>>>(s.surface_map,s.saved_surface.data(),s.surface,s.vertices,s.guard.data());
        CUDA_SAFE_CALL(cudaMemcpy(&errors,s.guard.data(),sizeof(errors),cudaMemcpyDeviceToHost));
    }
    if(errors)
    {
        ++stats.old_queries;
        if(errors&CP_ATTRIBUTE)++stats.fallback_reasons["attributes_changed"];
        if(errors&CP_MAPPING)++stats.fallback_reasons["mapping_changed"];
        if(errors&CP_NONFINITE)++stats.fallback_reasons["nonfinite_bounds"];
        if(errors&CP_CONTAINMENT)++stats.fallback_reasons["bounds_not_contained"];
        if(errors&CP_DIRECTION)++stats.fallback_reasons["direction_changed"];
        return false;
    }
    if(s.validate)
    {
        cp_validate();
        if(s.reference.counts[0]>capacity)
        {++stats.production_overflow_queries;s.reference_valid=false;}
    }
    {CostScope scope("collision.contact_pool.classify");cp_launch_classify(s,dcd,ccd,counts,indices,capacity);}
    ++stats.reused_queries;if(s.count)++stats.nonempty_reused_queries;return true;
}
IpcContactPoolReference ipc_contact_pool_reference()
{
    const auto& s=cp_state();IpcContactPoolReference r;if(!s.reference_valid)return r;
    r.valid=true;r.count=s.reference.counts[0];std::copy(s.reference.counts.begin(),s.reference.counts.end(),r.counts);
    r.dcd=s.reference.dcd.data();r.ccd=s.reference.ccd.data();r.matrix_indices=s.reference.indices.data();return r;
}
void ipc_contact_pool_end()
{auto& s=cp_state();s.active=false;s.sealed=false;s.trial_set=false;s.reference_valid=false;}
void ipc_contact_pool_reset_stats(){cp_stats()={};cp_state().reference_valid=false;}
Json ipc_contact_pool_stats_json()
{
    const auto& s=cp_stats();const auto& c=ipc_contact_pool_config();
    return {{"requested",c.enabled},{"validate",c.validate},{"attempts",s.attempts},{"reused_queries",s.reused_queries},
        {"old_queries",s.old_queries},{"prepare_calls",s.prepare_calls},{"capture_passes",s.capture_passes},{"generations",s.generations},
        {"validation_calls",s.validation_calls},{"validation_passed",s.validation_passed},{"validation_failed",s.validation_failed},
        {"pairs_compared",s.pairs_compared},{"nonempty_validation_calls",s.nonempty_validation_calls},
        {"pool_pairs",s.pool_pairs},{"vf_pairs",s.vf_pairs},{"ee_pairs",s.ee_pairs},{"pool_bytes_peak",s.pool_bytes_peak},
        {"fallback_reasons",s.fallback_reasons},{"diagnostic_host_ms",s.diagnostic_host_ms},
        {"diagnostic_overflow_retries",s.overflow_retries},{"diagnostic_peak_capacity",s.validation_peak_capacity},
        {"production_overflow_queries",s.production_overflow_queries},{"production_overflow_observed_only_when_validate",true},
        {"nonempty_reused_queries",s.nonempty_reused_queries},{"private_validation_preserves_production",true},
        {"signed_tuples_and_duplicates_preserved",true},{"same_address_metadata_checked_every_trial",true}};
}
} // namespace gipc

namespace
{
std::vector<Node> cp_fixture_nodes(uint32_t count)
{
    if(!count)return {};
    std::vector<Node> nodes(size_t(count)*2-1,Node{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu});
    std::vector<uint32_t> original(count);std::iota(original.begin(),original.end(),0u);
    std::mt19937 random(731u+count);std::shuffle(original.begin(),original.end(),random);
    uint32_t next=0;
    std::function<uint32_t(uint32_t,uint32_t,uint32_t)> build=[&](uint32_t lo,uint32_t hi,uint32_t parent)
    {
        if(hi-lo==1)
        {
            const uint32_t index=count-1+lo;
            nodes[index]={parent,0xffffffffu,0xffffffffu,original[lo]};return index;
        }
        const uint32_t index=next++;
        const uint32_t mid=lo+(hi-lo)/2;
        nodes[index].parent_idx=parent;
        nodes[index].left_idx=build(lo,mid,index);
        nodes[index].right_idx=build(mid,hi,index);
        return index;
    };
    if(build(0,count,0xffffffffu)!=0 || next!=count-1)
        throw std::runtime_error("contact pool fixture invalid generated topology");
    return nodes;
}

// Fixture trees use the same raw production queries and primitive helpers, but
// a CPU-built balanced topology keeps this a finite collision-module fixture.
template<class Element>
void cp_fixture_tree(lbvh& tree,const std::vector<Element>& elements,
    const std::vector<double3>& points,const std::vector<double3>& direction,double alpha)
{
    auto nodes=cp_fixture_nodes(static_cast<uint32_t>(elements.size()));
    std::vector<AABB> boxes(nodes.size());
    for(uint32_t i=0;i<elements.size();++i)
    {
        const uint32_t leaf=static_cast<uint32_t>(elements.size())-1+i;
        const auto e=elements[nodes[leaf].element_idx];AABB b;
        auto add=[&](uint32_t id) {
            const auto x=points[id],d=direction[id];b.combines(x.x,x.y,x.z);
            b.combines(x.x-d.x*alpha,x.y-d.y*alpha,x.z-d.z*alpha);
        };
        add(e.x);add(e.y);if constexpr(std::is_same_v<Element,uint3>)add(e.z);
        boxes[leaf]=b;
    }
    std::function<AABB(uint32_t)> build=[&](uint32_t node) {
        if(nodes[node].element_idx!=0xffffffffu)return boxes[node];
        const auto a=build(nodes[node].left_idx),b=build(nodes[node].right_idx);
        return boxes[node]=merge(a,b);
    };
    if(!nodes.empty())build(0);
    tree._nodes.copy_from(nodes);tree._bvs.copy_from(boxes);
}
void cp_fixture_case(gipc::Json& cases,uint32_t query_count,bool empty,bool one_edge)
{
    std::vector<double3> points={make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0),
        make_double3(.2,.2,.01),make_double3(.5,-.03,0),make_double3(-.03,-.02,0),make_double3(5,5,5),
        make_double3(2,0,0),make_double3(3,0,0),make_double3(2.5,-.5,.01),make_double3(2.5,.5,.01),
        make_double3(2.1,.04,0),make_double3(2.9,.04,0)};
    std::vector<double3> direction(points.size(),make_double3(0,0,0));
    std::vector<uint3> faces=empty?std::vector<uint3>{}:std::vector<uint3>{make_uint3(0,1,2)};
    std::vector<uint2> edges=empty?std::vector<uint2>{}:std::vector<uint2>{make_uint2(7,8),make_uint2(9,10),make_uint2(11,12),make_uint2(7,8)};
    if(one_edge)edges.resize(1);
    std::vector<uint32_t> surface(query_count);
    for(uint32_t i=0;i<query_count;++i)surface[i]=3+i%4;
    if(empty){points.clear();direction.clear();surface.clear();}
    std::vector<int> body(points.size(),-1),fixed(points.size(),0);
    cudatool::DeviceBuffer<double3> dx(points),dr(points),dd(direction);
    cudatool::DeviceBuffer<uint3> df(faces);cudatool::DeviceBuffer<uint2> de(edges);
    cudatool::DeviceBuffer<uint32_t> ds(surface);cudatool::DeviceBuffer<int> db(body),dt(fixed);
    lbvh_f f;lbvh_e e;
    f.face_number=static_cast<uint32_t>(faces.size());f.vert_number=static_cast<uint32_t>(surface.size());
    f._vertexes=dx.data();f._bodyId=db.data();f._btype=dt.data();f._faces=df.data();f._surfVerts=ds.data();
    e.edge_number=static_cast<uint32_t>(edges.size());e.vert_number=static_cast<uint32_t>(points.size());
    e._vertexes=dx.data();e._rest_vertexes=dr.data();e._bodyId=db.data();e._btype=dt.data();e._edges=de.data();
    auto trees=[&]() {cp_fixture_tree(f,faces,points,direction,0);cp_fixture_tree(e,edges,points,direction,0);};
    trees();
    cudatool::DeviceBuffer<uint32_t> count;count.resize_discard(5);
    cudatool::DeviceBuffer<int4> swept;uint32_t capacity=0,n=0;
    auto capture=[&](uint64_t epoch) {
        cp_begin_impl(f,e,dd.data(),static_cast<uint32_t>(points.size()),1,.01,epoch,true,true);
        for(;;)
        {
            auto* sidecar=gipc::ipc_contact_pool_capture_pass(capacity);
            CUDA_SAFE_CALL(cudaMemsetAsync(count.data(),0,5*sizeof(uint32_t),cudaStreamPerThread));
            if(f.face_number&&f.vert_number)fullCCDselfQuery_vf(f._bodyId,f._btype,f._vertexes,dd.data(),1.,f._faces,f._surfVerts,
                f._bvs.data(),f._nodes.data(),swept.data(),count.data(),.01,capacity,f.vert_number,sidecar,gipc::ipc_contact_pool_capture_epoch());
            if(e.edge_number>1)fullCCDselfQuery_ee(e._bodyId,e._btype,e._vertexes,dd.data(),1.,e._edges,
                e._bvs.data(),e._nodes.data(),swept.data(),count.data(),.01,capacity,e.edge_number,sidecar,gipc::ipc_contact_pool_capture_epoch());
            CUDA_SAFE_CALL(cudaMemcpy(&n,count.data(),sizeof(n),cudaMemcpyDeviceToHost));
            if(n<=capacity)break;capacity=n;swept.resize_discard(capacity);
        }
        gipc::ipc_contact_pool_seal(n);
        // Capture may only add a sidecar; independently run the old swept
        // queries and compare their full CCD tuples to the capture output.
        cudatool::DeviceBuffer<int4> old_swept;old_swept.resize_discard(std::max(n,1u));
        cudatool::DeviceBuffer<uint32_t> old_count;old_count.resize_discard(1);
        CUDA_SAFE_CALL(cudaMemsetAsync(old_count.data(),0,sizeof(uint32_t),cudaStreamPerThread));
        if(f.face_number&&f.vert_number)fullCCDselfQuery_vf(f._bodyId,f._btype,f._vertexes,dd.data(),1.,f._faces,f._surfVerts,
            f._bvs.data(),f._nodes.data(),old_swept.data(),old_count.data(),.01,n,f.vert_number);
        if(e.edge_number>1)fullCCDselfQuery_ee(e._bodyId,e._btype,e._vertexes,dd.data(),1.,e._edges,
            e._bvs.data(),e._nodes.data(),old_swept.data(),old_count.data(),.01,n,e.edge_number);
        uint32_t old_n=0;CUDA_SAFE_CALL(cudaMemcpy(&old_n,old_count.data(),sizeof(old_n),cudaMemcpyDeviceToHost));
        if(old_n!=n)throw std::runtime_error("contact pool capture changed FullCCD count");
        std::vector<int4> hnew(n),hold(n);
        if(n)
        {
            CUDA_SAFE_CALL(cudaMemcpy(hnew.data(),swept.data(),n*sizeof(int4),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hold.data(),old_swept.data(),n*sizeof(int4),cudaMemcpyDeviceToHost));
        }
        std::vector<std::array<int,4>> a,b;
        for(const auto p:hnew)a.push_back({p.x,p.y,p.z,p.w});
        for(const auto p:hold)b.push_back({p.x,p.y,p.z,p.w});
        std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        if(a!=b)throw std::runtime_error("contact pool capture changed FullCCD multiset");
        cases.push_back({{"case","capture_preserves_full_ccd"},{"generation",epoch},{"pairs",n},{"pass",true}});
    };
    capture(1);
    cudatool::DeviceBuffer<int4> production_dcd,production_ccd;
    cudatool::DeviceBuffer<int> production_indices;
    auto run=[&](const char* label,double alpha,bool expected_reuse) {
        gipc::ipc_contact_pool_set_trial(alpha);
        // Deliberately poisoned caller outputs/counts: both failed guards and
        // private validation must leave them intact before production classify.
        std::array<uint32_t,5> sentinel={11,0,3,4,4};
        CUDA_SAFE_CALL(cudaMemcpy(count.data(),sentinel.data(),sizeof(sentinel),cudaMemcpyHostToDevice));
        const uint32_t production_capacity=n+11;
        production_dcd.resize_discard(production_capacity+1);production_ccd.resize_discard(production_capacity+1);
        production_indices.resize_discard(production_capacity+1);
        CUDA_SAFE_CALL(cudaMemsetAsync(production_dcd.data(),0xa5,(production_capacity+1)*sizeof(int4),cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemsetAsync(production_ccd.data(),0xa5,(production_capacity+1)*sizeof(int4),cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemsetAsync(production_indices.data(),0xa5,(production_capacity+1)*sizeof(int),cudaStreamPerThread));
        const bool used=gipc::ipc_contact_pool_try_discrete(f,e,.01,production_dcd.data(),production_ccd.data(),count.data(),production_indices.data(),production_capacity);
        if(used!=expected_reuse)throw std::runtime_error(std::string("contact pool fixture unexpected reuse: ")+label);
        std::array<uint32_t,5> after{};CUDA_SAFE_CALL(cudaMemcpy(after.data(),count.data(),sizeof(after),cudaMemcpyDeviceToHost));
        if(!used&&after!=sentinel)throw std::runtime_error("contact pool failed guard changed production counts");
        auto reference=gipc::ipc_contact_pool_reference();
        if(used)
        {
            const auto& expected=cp_state().reference.counts;
            if(!reference.valid)throw std::runtime_error("contact pool reference missing on complete production pass");
            for(int i=0;i<5;++i)if(after[i]!=sentinel[i]+expected[i])
                throw std::runtime_error("contact pool private validation modified production counts");
        }
        else if(reference.valid)throw std::runtime_error("contact pool stale reference escaped fallback");
        std::vector<int4> pd,pc;std::vector<int> pi;
        production_dcd.copy_to(pd);production_ccd.copy_to(pc);production_indices.copy_to(pi);
        std::vector<std::array<int,9>> records;std::array<std::vector<int>,5> ranks;
        for(uint32_t j=0;j<=production_capacity;++j)
        {
            if(!used||j<11||j>=after[0])
            {
                const unsigned char* a=reinterpret_cast<const unsigned char*>(&pd[j]);
                const unsigned char* b=reinterpret_cast<const unsigned char*>(&pc[j]);
                const unsigned char* c=reinterpret_cast<const unsigned char*>(&pi[j]);
                for(size_t k=0;k<sizeof(int4);++k)if(a[k]!=0xa5||b[k]!=0xa5)throw std::runtime_error("contact pool wrote outside caller live slots");
                for(size_t k=0;k<sizeof(int);++k)if(c[k]!=0xa5)throw std::runtime_error("contact pool wrote outside caller live indices");
            }
            else
            {
                const auto p=pd[j],q=pc[j];const int type=(p.x>=0||p.y<0)?4:(p.z<0?2:(p.w<0?3:4));
                records.push_back({type,p.x,p.y,p.z,p.w,q.x,q.y,q.z,q.w});ranks[type].push_back(pi[j]);
            }
        }
        if(used)
        {
            std::sort(records.begin(),records.end());
            if(records!=cp_state().reference.records)throw std::runtime_error("contact pool production differs from private reference");
            for(int t=2;t<=4;++t)
            {std::sort(ranks[t].begin(),ranks[t].end());for(size_t j=0;j<ranks[t].size();++j)
                if(ranks[t][j]!=static_cast<int>(sentinel[t]+j))throw std::runtime_error("contact pool production rank offset");}
        }
        gipc::Json result={{"case",label},{"query_points",surface.size()},{"pool_pairs",n},{"reused",used},
            {"expected_reuse",expected_reuse},{"caller_count_preserved_before_classify",true},{"pass",true}};
        if(used)result["typed_counts"]=cp_state().reference.counts;
        cases.push_back(std::move(result));
    };
    run(empty?"empty":"typed_duplicates_tail",0,true);run("inclusive_interval_end",1,true);
    // A distinct count-only production overflow pass: legacy helpers increment
    // total count before capacity rejection and leave type counts unchanged.
    if(n)
    {
        gipc::ipc_contact_pool_set_trial(0);
        CUDA_SAFE_CALL(cudaMemsetAsync(count.data(),0,5*sizeof(uint32_t),cudaStreamPerThread));
        if(!gipc::ipc_contact_pool_try_discrete(f,e,.01,nullptr,nullptr,count.data(),nullptr,0))
            throw std::runtime_error("contact pool zero-capacity pass unexpectedly fell back");
        std::array<uint32_t,5> c{};CUDA_SAFE_CALL(cudaMemcpy(c.data(),count.data(),sizeof(c),cudaMemcpyDeviceToHost));
        if(c[0]!=cp_state().reference.counts[0]||c[1]||c[2]||c[3]||c[4]||gipc::ipc_contact_pool_reference().valid)
            throw std::runtime_error("contact pool zero-capacity overflow contract");
        cases.push_back({{"case","zero_capacity_overflow"},{"pairs",c[0]},{"reference_invalid",true},{"pass",true}});
        run("complete_retry_after_overflow",0,true);
    }
    run("outside_interval",1.01,false);run("negative_alpha",-.1,false);
    run("nonfinite_alpha",std::numeric_limits<double>::quiet_NaN(),false);
    if(!empty)
    {
        body[3]=9;db.copy_from(body);run("inplace_body_changed",.5,false);body[3]=-1;db.copy_from(body);
        fixed[3]=3;dt.copy_from(fixed);run("inplace_btype_changed",.5,false);fixed[3]=0;dt.copy_from(fixed);
        direction[3].x=1;dd.copy_from(direction);run("inplace_direction_changed",.5,false);direction[3].x=0;dd.copy_from(direction);
        const auto saved=points[3];points[3].x=4;dx.copy_from(points);trees();run("actual_point_outside_swept_box",.5,false);
        points[3]=saved;dx.copy_from(points);trees();
        if(!surface.empty()){const auto id=surface[0];surface[0]=4;ds.copy_from(surface);run("inplace_surface_changed",.5,false);surface[0]=id;ds.copy_from(surface);}
        const auto face=faces[0];faces[0]=make_uint3(0,2,1);df.copy_from(faces);trees();run("inplace_face_mapping_changed",.5,false);
        faces[0]=face;df.copy_from(faces);trees();
        // Execute the concrete double-precision ABD arithmetic counterexample
        // on the host, then feed its actual resulting point to the GPU guard.
        // This is not a full ABD solver/trajectory fixture.
        volatile double q0=1e16,q1=1e16-2,dq0=1,dq1=1,a=1;
        const double origin=q0-q1,world_direction=dq0-dq1;
        const double predicted=origin-a*world_direction;
        const double moved0=q0-a*dq0,moved1=q1-a*dq1,actual=moved0-moved1;
        if(origin!=2||predicted!=2||actual!=4)throw std::runtime_error("contact pool ABD arithmetic fixture assumptions");
        points[3].x=origin;dx.copy_from(points);trees();capture(101);
        points[3].x=actual;dx.copy_from(points);trees();run("ABD_roundoff_actual_point_outside",1,false);
        cases.back()["origin"]=origin;cases.back()["predicted_endpoint"]=predicted;cases.back()["actual_q_step"]=actual;
        points[3]=saved;dx.copy_from(points);trees();
        // New generation may legitimately use new body/fixed values. Restore
        // between calls is not required: the next capture snapshots them.
        for(auto& v:body)v=7;db.copy_from(body);capture(2);run("new_generation_same_body_filtered",.5,true);
        for(auto& v:body)v=-1;db.copy_from(body);for(auto& v:fixed)v=3;dt.copy_from(fixed);
        capture(3);run("new_generation_all_fixed_3",.5,true);
        fixed[3]=1;dt.copy_from(fixed);capture(4);run("new_generation_partially_fixed",.5,true);
    }
    gipc::ipc_contact_pool_end();run("ended_epoch",0,false);
    cp_state().fixture=false;
}
} // anonymous namespace

int gipc::ipc_contact_pool_fixture(const char* output)
{
    Json report={{"fixture","ipc_contact_pool"},{"pass",false},{"cases",Json::array()},
        {"comparison","exact signed typed DCD/CCD multisets and type-local MatIndex bijections"},
        {"performance_claim",false},{"production_scene_validation_required",true},
        {"scope","synthetic raw production queries; no safety CCD or physical simulation skipped"}};
    int code=0;
    try
    {
        ipc_contact_pool_end();ipc_contact_pool_reset_stats();
        cp_fixture_case(report["cases"],0,true,false);
        cp_fixture_case(report["cases"],5,false,true);
        cp_fixture_case(report["cases"],257,false,false);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());report["pass"]=true;
    }
    catch(const std::exception& e){report["error"]=e.what();code=1;}
    report["stats"]=ipc_contact_pool_stats_json();ipc_contact_pool_end();cp_state().fixture=false;
    try
    {
        if(!output||!*output)throw std::runtime_error("contact pool fixture requires output path");
        std::ofstream file(output,std::ios::binary);if(!file)throw std::runtime_error("cannot open contact pool fixture output");
        file<<report.dump(2)<<'\n';if(!file)throw std::runtime_error("cannot write contact pool fixture output");
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    return code;
}
