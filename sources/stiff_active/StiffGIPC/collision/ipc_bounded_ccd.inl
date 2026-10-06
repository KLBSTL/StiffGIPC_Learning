// Included in exactly one CUDA translation unit after ipc_bounded_ccd.h.
// Original arithmetic copied from frozen stiff_perf_v50/collision/ACCD.cu
// (Kemeng Huang / StiffGIPC). The original device functions remain the truth.
#include <collision/ipc_bounded_ccd.h>
#include <collision/ACCD.cuh>
#include <math/gpu_eigen_libs.cuh>
#include <cuda_tools/cuda_buffer_view.h>
#include <cuda_tools/cuda_cub_wrappers.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

// These existing out-of-line device helpers are defined by ACCD.cu.
__device__ double edge_edge_distance_unclassified(const double3&,const double3&,const double3&,const double3&);
__device__ double point_triangle_distance_unclassified(const double3&,const double3&,const double3&,const double3&);

namespace gipc
{
namespace bounded_ccd_detail
{
struct Trace {int iterations=0;unsigned cut=0,fallback=0,prefix_invalid=0;};
__device__ inline bool finite3(const double3& v)
{return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);}
__device__ inline bool supported(const double3& a,const double3& b,const double3& c,const double3& d,
    const double3& da,const double3& db,const double3& dc,const double3& dd,
    double eta,double thickness,double bound)
{
    return isfinite(bound) && bound>0 && bound<1 && isfinite(eta) && eta>0 && eta<1 && thickness==0
        && finite3(a) && finite3(b) && finite3(c) && finite3(d)
        && finite3(da) && finite3(db) && finite3(dc) && finite3(dd);
}
template<bool UseBound,bool Observe>
__device__ inline double edge_edge_ccd_copy(const double3& _ea0,
                                const double3& _ea1,
                                const double3& _eb0,
                                const double3& _eb1,
                                const double3& _dea0,
                                const double3& _dea1,
                                const double3& _deb0,
                                const double3& _deb1,
                                double         eta,
                                double         thickness,double requested_bound,Trace* trace)
{
    if constexpr(UseBound)
    {
        if(!supported(_ea0,_ea1,_eb0,_eb1,_dea0,_dea1,_deb0,_deb1,eta,thickness,requested_bound))
        {
            if constexpr(Observe){trace->fallback=1;trace->iterations=-1;}
            return ::edge_edge_ccd(_ea0,_ea1,_eb0,_eb1,_dea0,_dea1,_deb0,_deb1,eta,thickness);
        }
    }
    double3 ea0 = _ea0, ea1 = _ea1, eb0 = _eb0, eb1 = _eb1, dea0 = _dea0,
            dea1 = _dea1, deb0 = _deb0, deb1 = _deb1;
    double3 temp0 = __GEIGEN__::__add(dea0, dea1);
    double3 temp1 = __GEIGEN__::__add(deb0, deb1);
    double3 mov = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__add(temp0, temp1), -0.25);

    dea0 = __GEIGEN__::__add(dea0, mov);
    dea1 = __GEIGEN__::__add(dea1, mov);
    deb0 = __GEIGEN__::__add(deb0, mov);
    deb1 = __GEIGEN__::__add(deb1, mov);

    double max_disp_mag =
        sqrt(std::max(__GEIGEN__::__squaredNorm(dea0), __GEIGEN__::__squaredNorm(dea1)))
        + sqrt(std::max(__GEIGEN__::__squaredNorm(deb0), __GEIGEN__::__squaredNorm(deb1)));
    if(max_disp_mag == 0)
        return 1.0;

    double dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);

    double dFunc = dist2_cur - thickness * thickness;
    if(dFunc <= 0)
    {
        double dists0 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb0));
        double dists1 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb1));
        double dists2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb0));
        double dists3 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb1));

        dist2_cur = std::min(std::min(dists0, dists1), std::min(dists2, dists3));
        dFunc     = dist2_cur - thickness * thickness;
    }
    double dist_cur = sqrt(dist2_cur);
    double gap      = eta * dFunc / (dist_cur + thickness);
    double toc      = 0.0;
    int    count    = 0;
    bool prefix_valid=true;
    if constexpr(UseBound)
        prefix_valid=isfinite(max_disp_mag) && max_disp_mag>0 && isfinite(dist2_cur) && dist2_cur>=0
            && isfinite(dist_cur) && dist_cur>=0 && isfinite(gap) && gap>=0
            && isfinite(dFunc) && dFunc>=0;
    if constexpr(UseBound && Observe)if(!prefix_valid)trace->prefix_invalid=1;
    while(true)
    {
        count++;
        if constexpr(Observe)trace->iterations=count;
        if(count > 50000)
            return toc;
        //if (count > 5000)
        //    printf("ee  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dea0.x, _dea0.y, _dea0.z, _dea1.x, _dea1.y, _dea1.z,
        //        _deb0.x, _deb0.y, _deb0.z, _deb1.x, _deb1.y, _deb1.z, _ea0.x, _ea0.y, _ea0.z, _ea1.x, _ea1.y, _ea1.z, _eb0.x, _eb0.y, _eb0.z, _eb1.x, _eb1.y, _eb1.z);
        double toc_lower_bound = (1 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag);
        ea0 = __GEIGEN__::__add(ea0, __GEIGEN__::__s_vec_multiply(dea0, toc_lower_bound));
        ea1 = __GEIGEN__::__add(ea1, __GEIGEN__::__s_vec_multiply(dea1, toc_lower_bound));
        eb0 = __GEIGEN__::__add(eb0, __GEIGEN__::__s_vec_multiply(deb0, toc_lower_bound));
        eb1 = __GEIGEN__::__add(eb1, __GEIGEN__::__s_vec_multiply(deb1, toc_lower_bound));

        dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);
        dFunc     = dist2_cur - thickness * thickness;
        if(dFunc <= 0)
        {
            double dists0 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb0));
            double dists1 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb1));
            double dists2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb0));
            double dists3 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb1));

            dist2_cur = std::min(std::min(dists0, dists1), std::min(dists2, dists3));
            dFunc = dist2_cur - thickness * thickness;
        }
        dist_cur = sqrt(dist2_cur);
        if(toc && (dFunc / (dist_cur + thickness) < gap))
        {
            break;
        }
        const double previous_toc=toc;
        toc += toc_lower_bound;
        if constexpr(UseBound)
        {
            prefix_valid=prefix_valid && isfinite(toc_lower_bound) && toc_lower_bound>=0
                && isfinite(toc) && toc>=0 && toc>=previous_toc
                && isfinite(dist2_cur) && dist2_cur>=0 && isfinite(dist_cur) && dist_cur>=0
                && isfinite(dFunc) && dFunc>=0 && finite3(ea0) && finite3(ea1) && finite3(eb0) && finite3(eb1);
            if constexpr(Observe)if(!prefix_valid)trace->prefix_invalid=1;
        }
        if(toc > 1.0)
            return 1.0;
        // Only after the original gap break, toc update, and toc>1 return.
        // Future arithmetic beyond this point is NOT claimed bitwise equivalent.
        if constexpr(UseBound)
        {
            if(prefix_valid && toc>requested_bound)
            {
                volatile double prefix_inverse=1.0/toc;
                const double effective_prefix=1.0/prefix_inverse;
                if(isfinite(effective_prefix) && effective_prefix>=requested_bound)
                {
                    if constexpr(Observe)trace->cut=1;
                    return 1.0; // nonbinding under the caller's unchanged clip
                }
            }
        }
    }
    return toc;
}

template<bool UseBound,bool Observe>
__device__ inline double point_triangle_ccd_copy(const double3& _p,
                                     const double3& _t0,
                                     const double3& _t1,
                                     const double3& _t2,
                                     const double3& _dp,
                                     const double3& _dt0,
                                     const double3& _dt1,
                                     const double3& _dt2,
                                     double         eta,
                                     double         thickness,double requested_bound,Trace* trace)
{
    if constexpr(UseBound)
    {
        if(!supported(_p,_t0,_t1,_t2,_dp,_dt0,_dt1,_dt2,eta,thickness,requested_bound))
        {
            if constexpr(Observe){trace->fallback=1;trace->iterations=-1;}
            return ::point_triangle_ccd(_p,_t0,_t1,_t2,_dp,_dt0,_dt1,_dt2,eta,thickness);
        }
    }
    double3 p = _p, t0 = _t0, t1 = _t1, t2 = _t2, dp = _dp, dt0 = _dt0,
            dt1 = _dt1, dt2 = _dt2;

    double3 temp0 = __GEIGEN__::__add(dt0, dt1);
    double3 temp1 = __GEIGEN__::__add(dt2, dp);
    double3 mov = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__add(temp0, temp1), -0.25);

    dt0 = __GEIGEN__::__add(dt0, mov);
    dt1 = __GEIGEN__::__add(dt1, mov);
    dt2 = __GEIGEN__::__add(dt2, mov);
    dp  = __GEIGEN__::__add(dp, mov);

    double disp_mag2_vec0 = __GEIGEN__::__squaredNorm(dt0);
    double disp_mag2_vec1 = __GEIGEN__::__squaredNorm(dt1);
    double disp_mag2_vec2 = __GEIGEN__::__squaredNorm(dt2);

    double max_disp_mag =
        __GEIGEN__::__norm(dp)
        + sqrt(std::max(disp_mag2_vec0, std::max(disp_mag2_vec1, disp_mag2_vec2)));
    if(max_disp_mag == 0)
        return 1.0;

    double dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
    double dist_cur  = sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc   = 0.0;
    int    count = 0;
    bool prefix_valid=true;
    if constexpr(UseBound)
        prefix_valid=isfinite(max_disp_mag) && max_disp_mag>0 && isfinite(dist2_cur) && dist2_cur>=0
            && isfinite(dist_cur) && dist_cur>=0 && isfinite(gap) && gap>=0;
    if constexpr(UseBound && Observe)if(!prefix_valid)trace->prefix_invalid=1;
    while(true)
    {
        count++;
        if constexpr(Observe)trace->iterations=count;
        if(count > 50000)
            return toc;
        //if (count > 5000)
        //    printf("pt  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dp.x, _dp.y, _dp.z, _dt0.x, _dt0.y, _dt0.z,
        //        _dt1.x, _dt1.y, _dt1.z, _dt2.x, _dt2.y, _dt2.z, _p.x, _p.y, _p.z, _t0.x, _t0.y, _t0.z, _t1.x, _t1.y, _t1.z, _t2.x, _t2.y, _t2.z);
        double toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness)
                                 / ((dist_cur + thickness) * max_disp_mag);

        p = __GEIGEN__::__add(p, __GEIGEN__::__s_vec_multiply(dp, toc_lower_bound));
        t0 = __GEIGEN__::__add(t0, __GEIGEN__::__s_vec_multiply(dt0, toc_lower_bound));
        t1 = __GEIGEN__::__add(t1, __GEIGEN__::__s_vec_multiply(dt1, toc_lower_bound));
        t2 = __GEIGEN__::__add(t2, __GEIGEN__::__s_vec_multiply(dt2, toc_lower_bound));

        dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
        dist_cur  = sqrt(dist2_cur);
        if(toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
        {
            break;
        }

        const double previous_toc=toc;
        toc += toc_lower_bound;
        if constexpr(UseBound)
        {
            prefix_valid=prefix_valid && isfinite(toc_lower_bound) && toc_lower_bound>=0
                && isfinite(toc) && toc>=0 && toc>=previous_toc
                && isfinite(dist2_cur) && dist2_cur>=0 && isfinite(dist_cur) && dist_cur>=0
                && finite3(p) && finite3(t0) && finite3(t1) && finite3(t2);
            if constexpr(Observe)if(!prefix_valid)trace->prefix_invalid=1;
        }
        if(toc > 1.0)
        {
            return 1.0;
        }
        // Only after the original gap break, toc update, and toc>1 return.
        // Future arithmetic beyond this point is NOT claimed bitwise equivalent.
        if constexpr(UseBound)
        {
            if(prefix_valid && toc>requested_bound)
            {
                volatile double prefix_inverse=1.0/toc;
                const double effective_prefix=1.0/prefix_inverse;
                if(isfinite(effective_prefix) && effective_prefix>=requested_bound)
                {
                    if constexpr(Observe)trace->cut=1;
                    return 1.0; // nonbinding under the caller's unchanged clip
                }
            }
        }
    }
    return toc;
}

template<bool UseBound,bool Observe>
__device__ inline double evaluate(const double3* x,int4 pair,const double3* direction,
    double slackness,double bound,Trace* trace)
{
    const double eta=1.0-slackness;
    if(pair.x<0)
    {
        pair.x=-pair.x-1;
        return point_triangle_ccd_copy<UseBound,Observe>(x[pair.x],x[pair.y],x[pair.z],x[pair.w],
            __GEIGEN__::__s_vec_multiply(direction[pair.x],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.y],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.z],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.w],-1),eta,0,bound,trace);
    }
    return edge_edge_ccd_copy<UseBound,Observe>(x[pair.x],x[pair.y],x[pair.z],x[pair.w],
        __GEIGEN__::__s_vec_multiply(direction[pair.x],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.y],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.z],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.w],-1),eta,0,bound,trace);
}
__device__ inline double original(const double3* x,int4 pair,const double3* direction,double slackness)
{
    const double eta=1.0-slackness;
    if(pair.x<0)
    {
        pair.x=-pair.x-1;
        return ::point_triangle_ccd(x[pair.x],x[pair.y],x[pair.z],x[pair.w],
            __GEIGEN__::__s_vec_multiply(direction[pair.x],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.y],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.z],-1),
            __GEIGEN__::__s_vec_multiply(direction[pair.w],-1),eta,0);
    }
    return ::edge_edge_ccd(x[pair.x],x[pair.y],x[pair.z],x[pair.w],
        __GEIGEN__::__s_vec_multiply(direction[pair.x],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.y],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.z],-1),
        __GEIGEN__::__s_vec_multiply(direction[pair.w],-1),eta,0);
}
struct Max
{
    __host__ __device__ double operator()(double a,double b)const{return a>b?a:b;}
};
__global__ void production_kernel(const double3* x,const int4* pairs,const double3* direction,
    double* output,double slackness,int count,double bound)
{
    const int begin=blockIdx.x*blockDim.x,i=begin+threadIdx.x;
    const int valid=count-begin<256?count-begin:256;
    double value=1.0;
    if(i<count)value=1.0/evaluate<true,false>(x,pairs[i],direction,slackness,bound,nullptr);
    using Reduce=cub::BlockReduce<double,256>;
    __shared__ typename Reduce::TempStorage storage;
    value=Reduce(storage).Reduce(value,Max{},valid);
    if(threadIdx.x==0)output[blockIdx.x]=value;
}
struct PairEvidence
{
    double old_return,counted_old_return,new_return,counted_new_return;
    double old_inverse,new_inverse,old_roundtrip,new_roundtrip,old_clipped,new_clipped;
    Trace old_trace,new_trace;
    unsigned pt,finite_inputs;
};
__global__ void debug_pair_kernel(const double3* x,const int4* pairs,const double3* direction,
    double slackness,int count,double bound,PairEvidence* output)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    PairEvidence value{};value.pt=pairs[i].x<0;
    const auto pair=pairs[i];const int first=pair.x<0?-pair.x-1:pair.x;
    value.finite_inputs=finite3(x[first]) && finite3(x[pair.y]) && finite3(x[pair.z]) && finite3(x[pair.w])
        && finite3(direction[first]) && finite3(direction[pair.y]) && finite3(direction[pair.z]) && finite3(direction[pair.w]);
    value.old_return=original(x,pairs[i],direction,slackness);
    value.counted_old_return=evaluate<false,true>(x,pairs[i],direction,slackness,bound,&value.old_trace);
    value.new_return=evaluate<true,false>(x,pairs[i],direction,slackness,bound,nullptr);
    value.counted_new_return=evaluate<true,true>(x,pairs[i],direction,slackness,bound,&value.new_trace);
    value.old_inverse=1.0/value.old_return;value.new_inverse=1.0/value.new_return;
    value.old_roundtrip=1.0/value.old_inverse;value.new_roundtrip=1.0/value.new_inverse;
    value.old_clipped=std::min(bound,value.old_roundtrip);
    value.new_clipped=std::min(bound,value.new_roundtrip);
    output[i]=value;
}
template<bool Old>
__global__ void debug_reduce_kernel(const PairEvidence* values,double* output,int count)
{
    const int begin=blockIdx.x*blockDim.x,i=begin+threadIdx.x;
    const int valid=count-begin<256?count-begin:256;
    double value=1.0;
    if(i<count)value=Old?values[i].old_inverse:values[i].new_inverse;
    using Reduce=cub::BlockReduce<double,256>;
    __shared__ typename Reduce::TempStorage storage;
    value=Reduce(storage).Reduce(value,Max{},valid);
    if(threadIdx.x==0)output[blockIdx.x]=value;
}
inline bool same_bits(double a,double b)
{
    uint64_t aa=0,bb=0;std::memcpy(&aa,&a,sizeof(a));std::memcpy(&bb,&b,sizeof(b));return aa==bb;
}
inline Json number_or_null(double value)
{return std::isfinite(value)?Json(value):Json(nullptr);}
inline double reduce_step(cudatool::DeviceBuffer<double>& partial,cudatool::DeviceBuffer<double>& scalar)
{
    CUDA_SAFE_CALL(cudaMemsetAsync(scalar.data(),0xff,sizeof(double),cudaStreamPerThread));
    cudatool::DeviceReduce().Max(partial.data(),scalar.data(),static_cast<int>(partial.size()));
    double value=0;CUDA_SAFE_CALL(cudaMemcpy(&value,scalar.data(),sizeof(value),cudaMemcpyDeviceToHost));
    return 1.0/value;
}
} // namespace bounded_ccd_detail

void ipc_bounded_ccd_launch(const double3* x,const int4* pairs,const double3* direction,
    double* queue,double slackness,int count,double requested_bound)
{
    if(count<0)throw std::runtime_error("negative bounded CCD pair count");
    if(!count)return;
    if(!x || !pairs || !direction || !queue)throw std::runtime_error("null bounded CCD launch storage");
    bounded_ccd_detail::production_kernel<<<(count-1)/256+1,256>>>(
        x,pairs,direction,queue,slackness,count,requested_bound);
}

Json ipc_bounded_ccd_compare(const double3* x,const int4* pairs,const double3* direction,
    double slackness,int count,double requested_bound)
{
    using namespace bounded_ccd_detail;
    const auto diagnostic_start=std::chrono::steady_clock::now();
    if(count<0)throw std::runtime_error("negative bounded CCD diagnostic pair count");
    Json report={{"passed",true},{"pairs_compared",count},{"block_threads",256},
        {"requested_bound",number_or_null(requested_bound)},
        {"bound_supported",std::isfinite(requested_bound) && requested_bound>0 && requested_bound<1},
        {"old_step",1.0},{"new_step",1.0},{"new_debug_step",1.0},
        {"old_clipped_step",number_or_null(std::min(requested_bound,1.0))},
        {"new_clipped_step",number_or_null(std::min(requested_bound,1.0))},
        {"original_count_clone_bitwise_equal",true},{"new_count_clone_bitwise_equal",true},
        {"per_pair_clipped_bitwise_equal",true},{"final_clipped_bitwise_equal",true},
        {"production_debug_bitwise_equal",true},{"raw_returns_bitwise_equal",true},
        {"partial_outputs_complete",true},{"production_partial_debug_bitwise_equal",true},
        {"finite_returns",true},{"finite_inputs",true},{"cut_pairs",0},{"fallback_pairs",0},{"prefix_invalid_pairs",0},
        {"old_iterations",0},{"new_iterations",0},{"new_iterations_known",0},{"new_iterations_unknown_pairs",0},
        {"fallback_iterations_from_verified_old_clone",0},{"iterations_are_instrumented_clone",true},
        {"diagnostic_host_ms",0.0},
        {"pt_pairs",0},{"ee_pairs",0},{"first_mismatch_pair",nullptr},
        {"input_ownership","read-only; diagnostics own all outputs"}};
    if(!count)
    {
        report["diagnostic_host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-diagnostic_start).count();
        return report;
    }
    if(!x || !pairs || !direction)throw std::runtime_error("null bounded CCD diagnostic input");
    const int blocks=(count-1)/256+1;
    cudatool::DeviceBuffer<PairEvidence> device_values(count);
    cudatool::DeviceBuffer<double> old_partial(blocks),new_partial(blocks),production_partial(blocks),scalar(1);
    // Poison every private output, including tail-block partials.
    for(auto* p:{&old_partial,&new_partial,&production_partial})
        CUDA_SAFE_CALL(cudaMemsetAsync(p->data(),0xff,p->size()*sizeof(double),cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(device_values.data(),0xff,device_values.size()*sizeof(PairEvidence),cudaStreamPerThread));
    debug_pair_kernel<<<blocks,256>>>(x,pairs,direction,slackness,count,requested_bound,device_values.data());
    debug_reduce_kernel<true><<<blocks,256>>>(device_values.data(),old_partial.data(),count);
    debug_reduce_kernel<false><<<blocks,256>>>(device_values.data(),new_partial.data(),count);
    ipc_bounded_ccd_launch(x,pairs,direction,production_partial.data(),slackness,count,requested_bound);
    CUDA_SAFE_CALL(cudaGetLastError());
    const double old_step=reduce_step(old_partial,scalar),new_debug_step=reduce_step(new_partial,scalar);
    const double new_step=reduce_step(production_partial,scalar);
    std::vector<double> op,np,pp;old_partial.copy_to(op);new_partial.copy_to(np);production_partial.copy_to(pp);
    bool partial_complete=true,partial_equal=true;
    for(int i=0;i<blocks;++i)
    {
        // +infinity is a legitimate reciprocal of a zero feasible step.
        partial_complete=partial_complete && !std::isnan(op[i]) && !std::isnan(np[i]) && !std::isnan(pp[i])
            && op[i]>=1 && np[i]>=1 && pp[i]>=1;
        partial_equal=partial_equal && same_bits(np[i],pp[i]);
    }
    const bool clip_valid=std::isfinite(requested_bound) && requested_bound>=0 && requested_bound<=1;
    const double old_clipped=std::min(requested_bound,old_step),new_clipped=std::min(requested_bound,new_step);
    std::vector<PairEvidence> values;device_values.copy_to(values);
    uint64_t old_iterations=0,new_iterations=0,fallback_iterations=0,unknown=0,cuts=0,fallback=0,invalid=0,pt=0;
    bool old_copy=true,new_copy=true,raw_equal=true,clipped_equal=true,finite=true,finite_inputs=true;
    for(size_t i=0;i<values.size();++i)
    {
        const auto& v=values[i];
        const bool a=same_bits(v.old_return,v.counted_old_return),b=same_bits(v.new_return,v.counted_new_return);
        const bool c=clip_valid?same_bits(v.old_clipped,v.new_clipped):same_bits(v.old_return,v.new_return);
        const bool f=std::isfinite(v.old_return) && std::isfinite(v.new_return)
            && v.old_return>=0 && v.old_return<=1 && v.new_return>=0 && v.new_return<=1;
        old_copy=old_copy&&a;new_copy=new_copy&&b;clipped_equal=clipped_equal&&c;finite=finite&&f;
        finite_inputs=finite_inputs&&v.finite_inputs;
        raw_equal=raw_equal&&same_bits(v.old_return,v.new_return);
        if((!a || !b || !c || !f || !v.finite_inputs) && report["first_mismatch_pair"].is_null())report["first_mismatch_pair"]=i;
        old_iterations+=std::max(v.old_trace.iterations,0);
        if(v.new_trace.iterations<0){++unknown;fallback_iterations+=std::max(v.old_trace.iterations,0);}
        else new_iterations+=v.new_trace.iterations;
        cuts+=v.new_trace.cut;fallback+=v.new_trace.fallback;invalid+=v.new_trace.prefix_invalid;pt+=v.pt;
    }
    const bool final_equal=clip_valid?same_bits(old_clipped,new_clipped):same_bits(old_step,new_step);
    const bool production_equal=same_bits(new_step,new_debug_step);
    report["old_step"]=number_or_null(old_step);report["new_step"]=number_or_null(new_step);
    report["new_debug_step"]=number_or_null(new_debug_step);
    report["old_clipped_step"]=number_or_null(old_clipped);report["new_clipped_step"]=number_or_null(new_clipped);
    report["original_count_clone_bitwise_equal"]=old_copy;report["new_count_clone_bitwise_equal"]=new_copy;
    report["raw_returns_bitwise_equal"]=raw_equal;report["per_pair_clipped_bitwise_equal"]=clipped_equal;
    report["final_clipped_bitwise_equal"]=final_equal;report["production_debug_bitwise_equal"]=production_equal;
    report["partial_outputs_complete"]=partial_complete;report["production_partial_debug_bitwise_equal"]=partial_equal;
    report["finite_returns"]=finite;report["cut_pairs"]=cuts;report["fallback_pairs"]=fallback;
    report["finite_inputs"]=finite_inputs;
    report["prefix_invalid_pairs"]=invalid;report["old_iterations"]=old_iterations;
    report["new_iterations_known"]=new_iterations;report["new_iterations_unknown_pairs"]=unknown;
    report["fallback_iterations_from_verified_old_clone"]=fallback_iterations;
    report["new_iterations"]=new_iterations+fallback_iterations;
    report["pt_pairs"]=pt;report["ee_pairs"]=count-pt;
    report["passed"]=old_copy && new_copy && clipped_equal && final_equal && production_equal && finite && finite_inputs && partial_complete && partial_equal
        && std::isfinite(old_step) && std::isfinite(new_step);
    report["diagnostic_host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-diagnostic_start).count();
    return report;
}
int ipc_bounded_ccd_fixture(const char* output)
{
    Json report={{"fixture","ipc_bounded_request_ccd"},{"passed",false},
        {"cases",Json::array()},{"geometry_cases",Json::array({"pt_zero_motion","ee_zero_motion",
            "pt_approach","ee_approach","pt_separate","ee_separate","pt_near_contact","ee_near_contact",
            "pt_grazing","ee_near_parallel","pt_thin_triangle"})},
        {"cut_interval_future_equivalence","not assumed; same-state clipped results required"},
        {"production_scene_validation_required",true},{"performance_claim",false}};
    int status=0;
    try
    {
        uint64_t total_pairs=0,total_cuts=0;
        auto run=[&](int count,double bound,const char* label)
        {
            std::vector<double3> x(size_t(count)*4),direction(size_t(count)*4,make_double3(0,0,0));
            std::vector<int4> pairs(count);
            for(int i=0;i<count;++i)
            {
                const int kind=i%11,base=i*4;const bool point=(kind%2)==0;
                const double height=kind>=6?1e-8:1.0;
                const double displacement=kind<2?0.0:(kind<4?2.0:(kind<6?-0.2:1e-7));
                if(point)
                {
                    x[base]=make_double3(.25,.25,height);
                    x[base+1]=make_double3(0,0,0);x[base+2]=make_double3(1,0,0);x[base+3]=make_double3(0,1,0);
                    direction[base]=make_double3(0,0,displacement);
                    pairs[i]=make_int4(-base-1,base+1,base+2,base+3);
                }
                else
                {
                    x[base]=make_double3(-1,0,0);x[base+1]=make_double3(1,0,0);
                    x[base+2]=make_double3(0,-1,height);x[base+3]=make_double3(0,1,height);
                    direction[base+2]=direction[base+3]=make_double3(0,0,displacement);
                    pairs[i]=make_int4(base,base+1,base+2,base+3);
                }
                if(kind==8)
                {
                    x[base]=make_double3(.25,.25,.02);
                    direction[base]=make_double3(.5,0,0);
                }
                if(kind==9)
                {
                    x[base+2]=make_double3(-1,.1,.02);x[base+3]=make_double3(1,.1000001,.02);
                    direction[base+2]=direction[base+3]=make_double3(.5,0,0);
                }
                if(kind==10)
                {
                    x[base]=make_double3(.2,2e-6,.01);x[base+3]=make_double3(0,1e-5,0);
                    direction[base]=make_double3(0,0,.1);
                }
            }
            cudatool::DeviceBuffer<double3> dx(x),dd(direction);
            cudatool::DeviceBuffer<int4> dp(pairs);
            auto result=ipc_bounded_ccd_compare(dx.data(),dp.data(),dd.data(),.8,count,bound);
            std::vector<double3> after_x,after_direction;std::vector<int4> after_pairs;
            dx.copy_to(after_x);dd.copy_to(after_direction);dp.copy_to(after_pairs);
            const bool unchanged=(x.empty() || std::memcmp(x.data(),after_x.data(),x.size()*sizeof(double3))==0)
                && (direction.empty() || std::memcmp(direction.data(),after_direction.data(),direction.size()*sizeof(double3))==0)
                && (pairs.empty() || std::memcmp(pairs.data(),after_pairs.data(),pairs.size()*sizeof(int4))==0);
            result["case"]=label;result["inputs_bitwise_unchanged"]=unchanged;
            total_pairs+=count;total_cuts+=result["cut_pairs"].get<uint64_t>();
            report["cases"].push_back(result);
            if(!result["passed"].get<bool>() || !unchanged)
                throw std::runtime_error(std::string("bounded CCD fixture failed: ")+label);
        };
        run(11,.05,"short_request_all_geometries");
        run(11,.25,"middle_request_all_geometries");
        run(11,.9,"long_request_all_geometries");
        run(11,0,"zero_request_fallback");
        run(11,1,"unit_request_fallback");
        run(11,-.1,"negative_request_fallback");
        run(11,std::numeric_limits<double>::quiet_NaN(),"nan_request_fallback");
        run(11,std::numeric_limits<double>::infinity(),"infinite_request_fallback");
        for(const int count:{0,1,255,256,257})run(count,.05,"block_boundary_and_tail");
        // Zero relative motion makes the old function return 1 before reading
        // distances. Its benign scalar must not certify invalid geometry.
        for(const double invalid:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
        {
            std::vector<double3> x={make_double3(.25,.25,invalid),make_double3(0,0,0),
                make_double3(1,0,0),make_double3(0,1,0)},direction(4,make_double3(0,0,0));
            cudatool::DeviceBuffer<double3> dx(x),dd(direction);
            cudatool::DeviceBuffer<int4> dp(std::vector<int4>{make_int4(-1,1,2,3)});
            auto result=ipc_bounded_ccd_compare(dx.data(),dp.data(),dd.data(),.8,1,.05);
            const bool rejected=!result["passed"].get<bool>() && !result["finite_inputs"].get<bool>()
                && result["fallback_pairs"].get<uint64_t>()==1 && result["raw_returns_bitwise_equal"].get<bool>();
            result["case"]="invalid_geometry_explicit_rejection";result["expected_rejection"]=true;
            result["fixture_case_passed"]=rejected;report["cases"].push_back(result);++total_pairs;
            if(!rejected)throw std::runtime_error("nonfinite geometry was not rejected with unchanged legacy fallback");
        }
        if(!total_cuts)throw std::runtime_error("bounded CCD fixture did not exercise a cutoff");
        report["pairs_compared"]=total_pairs;report["cut_pairs"]=total_cuts;
        report["passed"]=true;
    }
    catch(const std::exception& e){report["error"]=e.what();status=1;}
    try
    {
        if(!output || !*output)throw std::runtime_error("bounded CCD fixture requires an output JSON path");
        std::ofstream file(output,std::ios::binary);
        if(!file)throw std::runtime_error("cannot open bounded CCD fixture output");
        file<<report.dump(2)<<'\n';if(!file)throw std::runtime_error("cannot write bounded CCD fixture output");
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    return status;
}
} // namespace gipc
