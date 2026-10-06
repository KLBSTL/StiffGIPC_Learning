// Experimental Stiff-GIPC integration of Zheng et al., Algorithms 1--3.
// Full swept BVH/ACCD remains the safety filter. Earliest-per-vertex filtering
// only reduces the AL active set. Trial states never evaluate IPC barriers.
#include <core/GIPC.cuh>
#include <abd_system/abd_sim_data.h>
#include <abd_system/abd_system.h>
#include <collision/ACCD.cuh>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <climits>
#include <filesystem>
#include <fstream>
#include <functional>
#include <chrono>
#include <core/accel_features.h>
#include <solver/toi_volume_step.cuh> // Includes conservative handling of volume tangency.
#include <solver/toi_port_policy.cuh>
#include <solver/toi_options.h>
#include <solver/toi_penalty_scale.h>
#include <solver/toi_observer.h>
#include <linear_system/linear_system/global_linear_system.h>
#include <cstring>
extern double total_Cg_count;
extern int total_Frames;

namespace
{
struct TriangleEnergySnapshot
{
    std::vector<double> energy;
    double total=0;
    double max_tensile_axis_strain=0;
};

TriangleEnergySnapshot triangle_energy_snapshot(
    const std::vector<double3>& vertices,const std::vector<uint3>& triangles,
    const std::vector<__GEIGEN__::Matrix2x2d>& inverse_rest,
    const std::vector<double>& area,double dt2,double stretch,double shear,double strain_rate)
{
    TriangleEnergySnapshot result;
    result.energy.resize(triangles.size());
    for(size_t i=0;i<triangles.size();++i)
    {
        const auto t=triangles[i];
        const auto a=vertices[t.x],b=vertices[t.y],c=vertices[t.z];
        const double3 d0=make_double3(b.x-a.x,b.y-a.y,b.z-a.z);
        const double3 d1=make_double3(c.x-a.x,c.y-a.y,c.z-a.z);
        const auto& m=inverse_rest[i].m;
        const double3 u=make_double3(d0.x*m[0][0]+d1.x*m[1][0],
                                     d0.y*m[0][0]+d1.y*m[1][0],
                                     d0.z*m[0][0]+d1.z*m[1][0]);
        const double3 v=make_double3(d0.x*m[0][1]+d1.x*m[1][1],
                                     d0.y*m[0][1]+d1.y*m[1][1],
                                     d0.z*m[0][1]+d1.z*m[1][1]);
        const double eu=std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z)-1;
        const double ev=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z)-1;
        const double uv=u.x*v.x+u.y*v.y+u.z*v.z;
        const double value=dt2*area[i]*(stretch*(eu*eu+(eu>0?strain_rate*eu*eu*eu:0)
                                                +ev*ev+(ev>0?strain_rate*ev*ev*ev:0))
                                        +shear*uv*uv);
        if(!std::isfinite(value))throw std::runtime_error("TOI triangle audit found non-finite cloth energy");
        result.energy[i]=value;
        result.total+=value;
        result.max_tensile_axis_strain=std::max({result.max_tensile_axis_strain,eu,ev});
    }
    return result;
}

gipc::Json triangle_energy_delta_summary(const TriangleEnergySnapshot& before,
                                         const TriangleEnergySnapshot& full)
{
    std::vector<double> positive;
    double negative_total=0,net=0;
    int max_triangle=-1;
    double max_delta=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<before.energy.size();++i)
    {
        const double delta=full.energy[i]-before.energy[i];
        net+=delta;
        if(delta>0)positive.push_back(delta);
        else negative_total+=delta;
        if(delta>max_delta){max_delta=delta;max_triangle=int(i);}
    }
    std::sort(positive.begin(),positive.end(),std::greater<double>());
    double positive_total=0,top10=0;
    for(size_t i=0;i<positive.size();++i)
    {
        positive_total+=positive[i];
        if(i<10)top10+=positive[i];
    }
    double cumulative=0;
    int half_count=0,ninety_count=0;
    for(size_t i=0;i<positive.size();++i)
    {
        cumulative+=positive[i];
        if(!half_count && cumulative>=.5*positive_total)half_count=int(i)+1;
        if(!ninety_count && cumulative>=.9*positive_total)ninety_count=int(i)+1;
    }
    return {{"triangles",before.energy.size()},{"positive_triangles",positive.size()},
        {"net_delta",net},{"positive_total",positive_total},{"negative_total",negative_total},
        {"top1_positive_fraction",positive_total>0?positive.front()/positive_total:0},
        {"top10_positive_fraction",positive_total>0?top10/positive_total:0},
        {"half_positive_count",half_count},{"ninety_positive_count",ninety_count},
        {"max_delta_triangle",max_triangle},{"max_triangle_delta",max_delta},
        {"max_tensile_axis_strain_before",before.max_tensile_axis_strain},
        {"max_tensile_axis_strain_full",full.max_tensile_axis_strain}};
}

struct ToiMinimum {__host__ __device__ double operator()(double a,double b)const{return a<b?a:b;}};
__host__ __device__ double3 sub(double3 a,double3 b) { return make_double3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ double3 add(double3 a,double3 b) { return make_double3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ double3 mul(double3 a,double s) { return make_double3(a.x*s,a.y*s,a.z*s); }
__host__ __device__ double dot(double3 a,double3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ double3 cross(double3 a,double3 b)
{ return make_double3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
__host__ __device__ double clamp01(double t) { return fmin(1.,fmax(0.,t)); }

// Closest point on a triangle, including edges and vertices. Degenerate
// triangles are handled by the closest of their three segments.
__host__ __device__ double3 closest_triangle(double3 p,double3 a,double3 b,double3 c,double& u,double& v)
{
    double3 ab=sub(b,a), ac=sub(c,a), ap=sub(p,a);
    double d1=dot(ab,ap), d2=dot(ac,ap);
    if(dot(cross(ab,ac),cross(ab,ac)) <= 1e-28*fmax(1.,dot(ab,ab)*dot(ac,ac)))
    {
        double best=1e300; double3 result=a;
        double3 points[3]={a,b,c};
        for(int e=0;e<3;++e)
        {
            int j=(e+1)%3; double3 edge=sub(points[j],points[e]);
            double len=dot(edge,edge);
            double t=len>0?clamp01(dot(sub(p,points[e]),edge)/len):0;
            double3 q=add(points[e],mul(edge,t)); double ds=dot(sub(p,q),sub(p,q));
            if(ds<best)
            { best=ds; result=q; u=e==0?t:(e==1?1-t:0); v=e==1?t:(e==2?1-t:0); }
        }
        return result;
    }
    if(d1<=0 && d2<=0) {u=v=0;return a;}
    double3 bp=sub(p,b); double d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0 && d4<=d3) {u=1;v=0;return b;}
    double vc=d1*d4-d3*d2;
    if(vc<=0 && d1>=0 && d3<=0) {u=d1/(d1-d3);v=0;return add(a,mul(ab,u));}
    double3 cp=sub(p,c); double d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0 && d5<=d6) {u=0;v=1;return c;}
    double vb=d5*d2-d1*d6;
    if(vb<=0 && d2>=0 && d6<=0) {u=0;v=d2/(d2-d6);return add(a,mul(ac,v));}
    double va=d3*d6-d5*d4;
    if(va<=0 && d4-d3>=0 && d5-d6>=0)
    {v=(d4-d3)/((d4-d3)+(d5-d6));u=1-v;return add(b,mul(sub(c,b),v));}
    double inv=1/(va+vb+vc);u=vb*inv;v=vc*inv;
    return add(a,add(mul(ab,u),mul(ac,v)));
}
__host__ __device__ void closest_segments(double3 a,double3 b,double3 c,double3 d,double& s,double& t)
{
    double3 e=sub(b,a),f=sub(d,c),r=sub(a,c);
    double aa=dot(e,e),bb=dot(f,f),cc=dot(e,r),ff=dot(f,r);
    if(aa<=1e-28 && bb<=1e-28) {s=t=0;return;}
    if(aa<=1e-28) {s=0;t=clamp01(ff/bb);return;}
    if(bb<=1e-28) {t=0;s=clamp01(-cc/aa);return;}
    double ef=dot(e,f),den=aa*bb-ef*ef;
    s=den>1e-14*aa*bb?clamp01((ef*ff-cc*bb)/den):0;
    t=(ef*s+ff)/bb;
    if(t<0) {t=0;s=clamp01(-cc/aa);}
    else if(t>1) {t=1;s=clamp01((ef-cc)/aa);}
}
__host__ __device__ double constraint(const ToiContact& c,const double3* x)
{
    double value=c.offset;
    for(int j=0;j<(c.kind==2?1:4);++j) value+=dot(c.grad[j],sub(x[c.ids[j]],c.anchor[j]));
    return value;
}
__global__ void linearize(ToiContact* cs,int n,const double3* x,
                           const double3* gn,const double* go,double delta,int* invalid)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& c=cs[i]; double weights[4]={0,0,0,0}; double3 difference{};
    int count=c.kind==2?1:4;
    for(int j=0;j<count;++j)c.anchor[j]=x[c.ids[j]];
    if(c.kind==2)
    { c.grad[0]=gn[0];c.offset=dot(gn[0],c.anchor[0])-go[0]-delta;return; }
    if(c.kind==0)
    {
        double u,v; auto q=closest_triangle(c.anchor[0],c.anchor[1],c.anchor[2],c.anchor[3],u,v);
        difference=sub(c.anchor[0],q);weights[0]=1;weights[1]=u+v-1;weights[2]=-u;weights[3]=-v;
        c.coordinates=make_double2(u,v);
    }
    else
    {
        double s,t;closest_segments(c.anchor[0],c.anchor[1],c.anchor[2],c.anchor[3],s,t);
        difference=sub(add(c.anchor[0],mul(sub(c.anchor[1],c.anchor[0]),s)),
                       add(c.anchor[2],mul(sub(c.anchor[3],c.anchor[2]),t)));
        weights[0]=1-s;weights[1]=s;weights[2]=t-1;weights[3]=-t;c.coordinates=make_double2(s,t);
    }
    double distance=sqrt(dot(difference,difference));
    if(!isfinite(distance) || distance<=1e-15) {atomicMin(invalid,i+1);return;}
    auto normal=mul(difference,1/distance);
    for(int j=0;j<4;++j)c.grad[j]=mul(normal,weights[j]);
    c.offset=distance-delta;
}
__device__ double contact_mu(const ToiContact& c,double mu,double initial_mu)
{return c.penalty_mu>0?c.penalty_mu*(mu/initial_mu):mu;}
__global__ void update_slack(ToiContact* cs,int n,const double3* x,double mu,double initial_mu,bool multiplier,
                             bool independent_friction=false,bool age_release=false)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& c=cs[i];double value=constraint(c,x),local_mu=contact_mu(c,mu,initial_mu);
    c.slack=fmax(0.,value-c.lambda/local_mu);
    if(multiplier)
    {
        if(age_release)toi_port::update_release(c.lambda,c.gamma,c.release_age,value,local_mu,c.slack==0);
        else if(c.slack==0) {c.lambda-=local_mu*value;c.gamma=1;}
        else {c.lambda=0;c.gamma*=.9;}
        if(independent_friction)
        {
            // This proxy sees the same trial constraint, but has its own
            // frame-start history. It is an implementation ablation, not an
            // additional primal/dual variable in the AL objective.
            const double friction_slack=fmax(0.,value-c.friction_lambda/local_mu);
            if(friction_slack==0)
            {c.friction_lambda-=local_mu*value;c.friction_gamma=1;}
            else {c.friction_lambda=0;c.friction_gamma*=.9;}
        }
    }
}
__global__ void al_hessian(const ToiContact* cs,int begin,int n,const double3* x,
                          double3* g,Eigen::Matrix3d* values,int* rows,int* cols,
                          int offset,double mu,double initial_mu,bool reduced=false)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& c=cs[begin+i];double value=constraint(c,x),local_mu=contact_mu(c,mu,initial_mu);
    const double shifted=value-c.lambda/local_mu;
    double scale=reduced?c.gamma*local_mu*fmin(shifted,0.):c.gamma*(local_mu*(value-c.slack)-c.lambda);
    const double curvature=(reduced && shifted>0)?0.:local_mu*c.gamma;
    int count=c.kind==2?1:4,k=offset+i*count*count;
    for(int a=0;a<count;++a)
    {
        atomicAdd(&g[c.ids[a]].x,scale*c.grad[a].x);
        atomicAdd(&g[c.ids[a]].y,scale*c.grad[a].y);
        atomicAdd(&g[c.ids[a]].z,scale*c.grad[a].z);
        for(int b=0;b<count;++b)
        {
            int ia=c.ids[a],ib=c.ids[b];
            Eigen::Vector3d ga(c.grad[a].x,c.grad[a].y,c.grad[a].z);
            Eigen::Vector3d gb(c.grad[b].x,c.grad[b].y,c.grad[b].z);
            Eigen::Matrix3d h=curvature*ga*gb.transpose();
            rows[k]=ia;cols[k]=ib;values[k]=h;++k;
        }
    }
}
__global__ void al_energy(const ToiContact* cs,int n,const double3* x,double mu,double initial_mu,double* e,bool reduced=false)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& c=cs[i];double value=constraint(c,x)-c.slack;
    if(reduced)
    {
        // Eliminate the nonnegative slack at each line-search trial too.
        const double local_mu=contact_mu(c,mu,initial_mu);
        value=fmin(constraint(c,x)-c.lambda/local_mu,0.);
        e[i]=.5*c.gamma*(local_mu*value*value-c.lambda*c.lambda/local_mu);
        return;
    }
    e[i]=c.gamma*(.5*contact_mu(c,mu,initial_mu)*value*value-c.lambda*value);
}
__global__ void make_direction(const double3* safe,const double3* trial,double3* dir,int n)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dir[i]=sub(safe[i],trial[i]);}
__global__ void direction_squared_norm(const double3* dir,double* result,int n)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)result[i]=dot(dir[i],dir[i]);}
__global__ void direction_axis_max(const double3* dir,double* result,int n)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)result[i]=fmax(fabs(dir[i].x),fmax(fabs(dir[i].y),fabs(dir[i].z)));}
__global__ void blend_vertices(double3* x,const double3* safe,const double3* trial,double a,int n)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=a==1?trial[i]:add(safe[i],mul(sub(trial[i],safe[i]),a));}
__global__ void blend_q(double* q,const double* safe,const double* trial,double a,int n)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]=a==1?trial[i]:safe[i]+a*(trial[i]-safe[i]);}
__global__ void pair_ccd(const double3* x,const double3* dir,const int4* pairs,int n,double* times,double ratio)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int4 p=pairs[i];int ids[4]={p.x<0?-p.x-1:p.x,p.y,p.z,p.w};
    double3 d[4];for(int j=0;j<4;++j)d[j]=mul(dir[ids[j]],-1);
    if(p.x<0) times[i]=point_triangle_ccd(x[ids[0]],x[ids[1]],x[ids[2]],x[ids[3]],d[0],d[1],d[2],d[3],ratio,0);
    else times[i]=edge_edge_ccd(x[ids[0]],x[ids[1]],x[ids[2]],x[ids[3]],d[0],d[1],d[2],d[3],ratio,0);
}
__global__ void ground_ccd(const double3* x,const double3* dir,const uint32_t* surface,
                            int n,const double3* normal,const double* offset,double* times)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto v=surface[i];double distance=dot(x[v],normal[0])-offset[0];
    double speed=dot(dir[v],normal[0]);
    // Physical crossing time classifies candidates; safe acceptance uses 0.8t.
    times[i]=speed>0?fmin(1.,fmax(0.,distance/speed)):1.;
}
__global__ void validate_times(const double* times,int n,int* invalid)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && (!isfinite(times[i])||times[i]<0||times[i]>1))atomicExch(invalid,1);}
__global__ void diagonal_max(const Eigen::Matrix3d* h,const int* rows,const int* cols,int n,double* out)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(rows[i]==cols[i])
        for(int j=0;j<3;++j)atomicAdd(out+3*rows[i]+j,h[i](j,j));
}
__global__ void movable_diagonal(double* diagonal,int dofs,int abd_dofs,
                                 const BodyBoundaryType* body_type,const int* fem_type)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dofs)return;
    const bool fixed=i<abd_dofs?body_type[i/12]==BodyBoundaryType::Fixed:fem_type[(i-abd_dofs)/3]!=0;
    diagonal[i]=fixed?0:fabs(diagonal[i]);
}
__global__ void fem_only_diagonal(double* diagonal,int dofs,int abd_dofs)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<dofs && i<abd_dofs)diagonal[i]=0;}
__global__ void world_abd_diagonal(const gipc::ABDJacobi* jacobi,const int* body_ids,
                                  const BodyBoundaryType* boundary,const gipc::Matrix12x12* inverse,
                                  int bodies,int points,double* diagonal,int* invalid)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=points)return;
    diagonal[i]=0;
    const int body=body_ids[i];
    if(body<0||body>=bodies){atomicExch(invalid,1);return;}
    if(boundary[body]==BodyBoundaryType::Fixed)return;
    const auto j=jacobi[i].to_mat();
    const Eigen::Matrix3d c=j*inverse[body]*j.transpose();
    // Reject an invalid compliance rather than silently switching estimators.
    if(!(c(0,0)>0) || !(c(0,0)*c(1,1)-c(0,1)*c(1,0)>0) || !(c.determinant()>0))
    {atomicExch(invalid,1);return;}
    const Eigen::Matrix3d k=c.inverse();
    double value=0;
    for(int axis=0;axis<3;++axis)
    {
        if(!isfinite(k(axis,axis))||!(k(axis,axis)>0)){atomicExch(invalid,1);return;}
        value=fmax(value,k(axis,axis));
    }
    diagonal[i]=value;
}
__global__ void volume_step_bounds(const double3* x,const double3* move,
                                   const double3* end,const double3* rest,const uint4* tets,
                                   int n,double retain,double* bounds,int* invalid)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint4 t=tets[i];int ids[4]={int(t.x),int(t.y),int(t.z),int(t.w)};
    double3 p[4],dx[4],r[4];
    for(int j=0;j<4;++j)
    {p[j]=x[ids[j]];r[j]=rest[ids[j]];dx[j]=end?sub(end[ids[j]],p[j]):mul(move[ids[j]],-1);}
    int error=0;bounds[i]=toi_volume::bound(p,dx,r,retain,error);
    if(error)atomicExch(invalid,error);
}
__global__ void volume_fixture(int kind,double scale,double retain,double* result,int* invalid)
{
    double3 rest[4]={make_double3(2,2,2),make_double3(2+scale,2,2),
                    make_double3(2,2+scale,2),make_double3(2,2,2+scale)};
    double3 dx[4]{};
    if(kind==0)for(auto& d:dx)d=make_double3(.2,-.3,.4);
    if(kind==1)dx[3].z=-2*scale;
    if(kind==2){dx[1].x=-2*scale;dx[2].y=-2*scale;}
    if(kind==3||kind==4){dx[1].x=-3*scale;dx[2].y=-1.5*scale;}
    if(kind==4)dx[3].z=scale;
    int error=0;result[0]=toi_volume::bound(rest,dx,rest,retain,error);invalid[0]=error;
}
__global__ void friction_basis(const ToiContact* cs,int n,double* lambda,double2* coord,
                                __GEIGEN__::Matrix3x2d* basis,bool independent_history)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& c=cs[i];double3 normal=c.grad[0];
    if(dot(normal,normal)<1e-28)normal=c.grad[1];
    normal=mul(normal,rsqrt(dot(normal,normal)));
    double3 axis=fabs(normal.x)<.8?make_double3(1,0,0):make_double3(0,1,0);
    double3 t=cross(normal,axis);t=mul(t,rsqrt(dot(t,t)));double3 u=cross(normal,t);
    basis[i].m[0][0]=t.x;basis[i].m[1][0]=t.y;basis[i].m[2][0]=t.z;
    basis[i].m[0][1]=u.x;basis[i].m[1][1]=u.y;basis[i].m[2][1]=u.z;
    coord[i]=c.coordinates;
    lambda[i]=fmax(0.,independent_history?c.friction_lambda*c.friction_gamma:c.lambda*c.gamma);
}
__global__ void ground_lambda(const ToiContact* cs,int start,int n,double* lambda,bool independent_history)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){const auto& c=cs[start+i];lambda[i]=fmax(0.,independent_history?c.friction_lambda*c.friction_gamma:c.lambda*c.gamma);}}

using Key=std::array<int,5>;
Key key(const ToiContact& c) {return {c.kind,c.ids[0],c.ids[1],c.ids[2],c.ids[3]};}
gipc::Json outer_contact_probe(const std::vector<ToiContact>& contacts,
                              const std::vector<double3>& x,double mu,double initial_mu,double delta,
                              const std::vector<ToiContact>* previous=nullptr)
{
    double min_c=0,violation=0,split=0,complementarity=0,lambda_change=0,max_lambda=0;
    int positive_slack_zero_lambda=0;
    for(size_t i=0;i<contacts.size();++i)
    {
        const auto& c=contacts[i];const double value=constraint(c,x.data());
        const double local_mu=c.penalty_mu>0?c.penalty_mu*mu/initial_mu:mu;
        min_c=std::min(min_c,value);violation=std::max(violation,std::max(-value,0.));
        split=std::max(split,std::abs(value-c.slack));
        complementarity=std::max(complementarity,std::abs(c.gamma*c.lambda*value));
        max_lambda=std::max(max_lambda,c.gamma*c.lambda);
        positive_slack_zero_lambda+=c.slack>0&&c.lambda==0;
        if(previous)
        {
            if(previous->size()!=contacts.size()||key(previous->at(i))!=key(c))
                throw std::runtime_error("Outer dual probe contact identity mismatch");
            lambda_change=std::max(lambda_change,std::abs(c.lambda-previous->at(i).lambda)/local_mu/delta);
        }
    }
    return {{"contacts",contacts.size()},{"min_affine_gap_m",min_c},
        {"max_negative_affine_gap_over_delta",violation/delta},
        {"max_split_residual_over_delta",split/delta},
        {"max_abs_effective_lambda_times_gap",complementarity},
        {"max_abs_effective_lambda_times_gap_over_mu_delta2",complementarity/(mu*delta*delta)},
        {"max_effective_lambda",max_lambda},{"positive_slack_zero_lambda",positive_slack_zero_lambda},
        {"max_lambda_change_over_local_mu_delta",lambda_change},
        {"global_stationarity_certified",false}};
}
gipc::Json outer_shape_probe(const std::vector<double3>& x,const std::vector<double3>& target,
                            const std::vector<double3>& rest,const std::vector<uint4>& tets,int abd_points)
{
    double minimum=std::numeric_limits<double>::infinity(),negative_volume=0,max_distance=0;
    int nonpositive=0,fem_tets=0;
    for(const auto& t:tets)
    {
        if(t.x<unsigned(abd_points)||t.y<unsigned(abd_points)||t.z<unsigned(abd_points)||t.w<unsigned(abd_points))continue;
        const double v0=dot(sub(rest[t.y],rest[t.x]),cross(sub(rest[t.z],rest[t.x]),sub(rest[t.w],rest[t.x])));
        const double v=dot(sub(x[t.y],x[t.x]),cross(sub(x[t.z],x[t.x]),sub(x[t.w],x[t.x])));
        if(v0==0||!std::isfinite(v/v0))throw std::runtime_error("Outer probe invalid FEM volume");
        const double j=v/v0;minimum=std::min(minimum,j);nonpositive+=j<=0;++fem_tets;
        negative_volume+=std::max(-j,0.)*std::abs(v0)/6.;
    }
    for(size_t i=0;i<x.size();++i){const auto d=sub(x[i],target[i]);max_distance=std::max(max_distance,std::sqrt(dot(d,d)));}
    return {{"fem_tets",fem_tets},{"fem_min_J",fem_tets?gipc::Json(minimum):gipc::Json(nullptr)},
        {"fem_nonpositive",nonpositive},{"negative_fem_volume",negative_volume},
        {"max_distance_to_target_m",max_distance}};
}
gipc::Json outer_plane_probe(const std::vector<ToiContact>& old,const std::vector<ToiContact>& current,
                            const std::vector<double3>& warm,double delta)
{
    std::map<Key,const ToiContact*> prior;for(const auto& c:old)prior[key(c)]=&c;
    int retained=0;double shift=0,normal_change=0;
    for(const auto& c:current)
    {
        const auto it=prior.find(key(c));if(it==prior.end())continue;++retained;
        shift=std::max(shift,std::abs(constraint(c,warm.data())-constraint(*it->second,warm.data()))/delta);
        for(int slot=0;slot<(c.kind==2?1:4);++slot)
        {const auto d=sub(c.grad[slot],it->second->grad[slot]);normal_change=std::max(normal_change,std::sqrt(dot(d,d)));}
    }
    return {{"previous_contacts",old.size()},{"current_contacts",current.size()},{"retained",retained},
        {"added",current.size()-retained},{"dropped",old.size()-retained},
        {"max_affine_gap_change_at_same_warm_trial_over_delta",shift},{"max_gradient_vector_change",normal_change}};
}
bool history_flag(const char* name,bool fallback)
{
    return gipc::toi_history_flag(name,fallback);
}
void prepare_warm_contacts(std::vector<ToiContact>& contacts,bool keep_lambda,
                           bool keep_gamma,bool independent_friction,bool keep_friction)
{
    for(auto& c:contacts)
    {
        if(!independent_friction)
        {c.friction_lambda=c.lambda;c.friction_gamma=c.gamma;}
        else if(!keep_friction)
        {c.friction_lambda=0;c.friction_gamma=1;}
        if(!keep_lambda)c.lambda=0;
        if(!keep_gamma)c.gamma=1;
    }
}
double mass_contact_mu(const ToiContact& c,const std::vector<double>& per_vertex)
{
    if(per_vertex.empty())return 0;
    double result=std::numeric_limits<double>::infinity();
    for(int j=0;j<(c.kind==2?1:4);++j)result=std::min(result,per_vertex.at(c.ids[j]));
    return result;
}
ToiContact canonical(int4 p)
{
    ToiContact c;c.kind=p.x<0?0:1;
    c.ids[0]=p.x<0?-p.x-1:p.x;c.ids[1]=p.y;c.ids[2]=p.z;c.ids[3]=p.w;
    if(c.kind==0)std::sort(c.ids+1,c.ids+4);
    else
    {
        if(c.ids[0]>c.ids[1])std::swap(c.ids[0],c.ids[1]);
        if(c.ids[2]>c.ids[3])std::swap(c.ids[2],c.ids[3]);
        if(std::pair<int,int>(c.ids[0],c.ids[1])>std::pair<int,int>(c.ids[2],c.ids[3]))
        {std::swap(c.ids[0],c.ids[2]);std::swap(c.ids[1],c.ids[3]);}
    }
    return c;
}
int4 encoded(const ToiContact& c)
{return make_int4(c.kind==0?-c.ids[0]-1:c.ids[0],c.ids[1],c.ids[2],c.ids[3]);}
template<class T> void copy_d(T* dst,const T* src,size_t n)
{if(n)CUDA_SAFE_CALL(cudaMemcpy(dst,src,n*sizeof(T),cudaMemcpyDeviceToDevice));}
template<class T> void upload(cudatool::DeviceBuffer<T>& dst,const std::vector<T>& src)
{dst.resize(src.size());if(!src.empty())CUDA_SAFE_CALL(cudaMemcpy(dst.data(),src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> std::vector<T> download(const T* src,size_t n)
{std::vector<T> out(n);if(n)CUDA_SAFE_CALL(cudaMemcpy(out.data(),src,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}

// Opt-in diagnostics: no device work or I/O on the default path.
bool stage_enabled()
{
    return gipc::ToiObserver::stage_enabled(total_Frames+1);
}
void stage_event(const char* stage,gipc::Json details=gipc::Json::object())
{
    gipc::ToiObserver::stage_event(stage,std::move(details),total_Frames+1,
        [](){CUDA_SAFE_CALL(cudaDeviceSynchronize());});
}
void save_ccd_geometry(GIPC& g,const double3* target,bool update_set,const gipc::Json& record)
{
    const char* dir=std::getenv("GIPC_CCD_GEOMETRY_DIR");
    if(!stage_enabled() || !dir || !dir[0])return;
    const std::filesystem::path folder(dir);
    std::filesystem::create_directories(folder);
    // A metadata file is the commit marker. Interrupted writes are invalid.
    std::filesystem::remove(folder/"current.json");
    auto start=download(g._vertexes,g.vertexNum),end=download(target,g.vertexNum),direction=download(g._moveDir,g.vertexNum);
    double max_motion=0,max_position=0;size_t nonfinite=0;
    auto save=[&](const char* name,const std::vector<double3>& values)
    {
        std::ofstream out(folder/name,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double3));
        out.close();if(!out)throw std::runtime_error("Failed to save CCD geometry");
    };
    for(size_t i=0;i<start.size();++i)
    {
        const auto a=start[i],b=end[i],d=direction[i];
        const double dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
        max_motion=std::max(max_motion,std::sqrt(dx*dx+dy*dy+dz*dz));
        max_position=std::max(max_position,std::sqrt(b.x*b.x+b.y*b.y+b.z*b.z));
        nonfinite+=!(std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z)&&std::isfinite(b.x)&&std::isfinite(b.y)&&std::isfinite(b.z)&&std::isfinite(d.x)&&std::isfinite(d.y)&&std::isfinite(d.z));
    }
    save("start.bin",start);save("target.bin",end);save("direction.bin",direction);
    gipc::Json meta={{"frame",total_Frames+1},{"outer",record.value("outer",-1)},
        {"update_set",update_set},{"vertices",g.vertexNum},{"dHat",g.dHat},{"alpha",1},
        {"max_motion_m",max_motion},{"max_target_norm_m",max_position},{"nonfinite_vertices",nonfinite},
        {"complete_restart_checkpoint",false},{"contents","Rolling CCD geometry only; topology and parameters require the frozen scene"}};
    gipc::attach_solve_context(meta);
    std::ofstream out(folder/"current.json");out<<meta.dump(2);out.close();
    if(!out)throw std::runtime_error("Failed to commit CCD geometry metadata");
    stage_event("ccd_geometry_saved",meta);
}

void trace_safe(GIPC& g,int step)
{
    if(const char* dir=std::getenv("GIPC_TRACE_SUBSTEPS"))
    {
        std::filesystem::create_directories(dir);
        char name[64];std::snprintf(name,sizeof(name),"/safe_%04d_%04d.bin",total_Frames,step);
        auto state=download(g._vertexes,g.vertexNum);
        std::ofstream out(std::string(dir)+name,std::ios::binary);
        out.write(reinterpret_cast<const char*>(state.data()),state.size()*sizeof(double3));
        if(!out)throw std::runtime_error("Failed to export TOI safe substep");
    }
}

void publish_contacts(GIPC& g)
{
    auto& s=g.toi;
    std::sort(s.host_contacts.begin(),s.host_contacts.end(),[](const auto& a,const auto& b){return key(a)<key(b);});
    s.self_count=0;std::vector<int4> pairs;std::vector<uint32_t> ground;
    for(const auto& c:s.host_contacts)
        if(c.kind==2)ground.push_back(c.ids[0]);else {pairs.push_back(encoded(c));++s.self_count;}
    s.ground_count=ground.size();upload(s.contacts,s.host_contacts);
    upload(g._collisonPairs,pairs);upload(g._environment_collisionPair,ground);
    std::fill(g.h_cpNum,g.h_cpNum+5,0);
    g.h_cpNum[0]=g.h_cpNum[4]=s.self_count;g.h_gpNum=s.ground_count;
}
void linearize_contacts(GIPC& g,int outer)
{
    auto& s=g.toi;int n=s.contacts.size();if(!n)return;
    g._scalar_scratch.resize(1);const int sentinel=INT_MAX;
    CUDA_SAFE_CALL(cudaMemcpy(g._scalar_scratch.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
    linearize<<<(n+255)/256,256>>>(s.contacts.data(),n,s.safe.data(),g._groundNormal.data(),g._groundOffset.data(),s.delta,g._scalar_scratch.data());
    int invalid=sentinel;CUDA_SAFE_CALL(cudaMemcpy(&invalid,g._scalar_scratch.data(),sizeof(int),cudaMemcpyDeviceToHost));
    if(invalid!=sentinel)
    {
        const int index=invalid-1;
        const auto c=download(s.contacts.data()+index,1)[0];
        auto point=[](double3 p){return gipc::Json::array({p.x,p.y,p.z});};
        gipc::Json entry={{"frame",total_Frames},{"outer",outer},{"contact_index",index},
            {"active_count",n},{"kind",c.kind},{"ids",gipc::Json::array({c.ids[0],c.ids[1],c.ids[2],c.ids[3]})},
            {"closest_coordinates",gipc::Json::array({c.coordinates.x,c.coordinates.y})},
            {"lambda",c.lambda},{"gamma",c.gamma},{"slack",c.slack},
            {"penalty_mu_initial",c.penalty_mu>0?c.penalty_mu:s.initial_mu},
            {"penalty_mu_current",c.penalty_mu>0?c.penalty_mu*s.mu/s.initial_mu:s.mu},
            {"delta",s.delta}};
        auto& anchors=entry["safe_anchors"]=gipc::Json::array();
        for(int j=0;j<4;++j)anchors.push_back(point(c.anchor[j]));
        if(c.kind==0)
        {
            const auto q=add(c.anchor[1],add(mul(sub(c.anchor[2],c.anchor[1]),c.coordinates.x),
                mul(sub(c.anchor[3],c.anchor[1]),c.coordinates.y)));
            const auto d=sub(c.anchor[0],q);
            entry["distance_m"]=std::sqrt(dot(d,d));
            const auto e0=sub(c.anchor[2],c.anchor[1]),e1=sub(c.anchor[3],c.anchor[1]);
            entry["triangle_double_area_m2"]=std::sqrt(dot(cross(e0,e1),cross(e0,e1)));
        }
        else if(c.kind==1)
        {
            const auto a=add(c.anchor[0],mul(sub(c.anchor[1],c.anchor[0]),c.coordinates.x));
            const auto b=add(c.anchor[2],mul(sub(c.anchor[3],c.anchor[2]),c.coordinates.y));
            const auto d=sub(a,b),e0=sub(c.anchor[1],c.anchor[0]),e1=sub(c.anchor[3],c.anchor[2]);
            entry["distance_m"]=std::sqrt(dot(d,d));
            entry["edge_lengths_m"]=gipc::Json::array({std::sqrt(dot(e0,e0)),std::sqrt(dot(e1,e1))});
        }
        throw std::runtime_error("TOI distance linearization is singular at a safe contact: "+entry.dump());
    }
}
// Host-only probes use copies and never publish constraints or alter device state.
double blocker_distance(const ToiContact& c,const std::vector<double3>& x,
                        double3 ground_normal,double ground_offset)
{
    if(c.kind==2)return dot(ground_normal,x[c.ids[0]])-ground_offset;
    double3 difference;
    if(c.kind==0)
    {
        double u,v;
        difference=sub(x[c.ids[0]],closest_triangle(x[c.ids[0]],x[c.ids[1]],
            x[c.ids[2]],x[c.ids[3]],u,v));
    }
    else
    {
        double u,v;
        closest_segments(x[c.ids[0]],x[c.ids[1]],x[c.ids[2]],x[c.ids[3]],u,v);
        difference=sub(add(x[c.ids[0]],mul(sub(x[c.ids[1]],x[c.ids[0]]),u)),
            add(x[c.ids[2]],mul(sub(x[c.ids[3]],x[c.ids[2]]),v)));
    }
    return std::sqrt(dot(difference,difference));
}

bool blocker_linear_probe(ToiContact& c,const std::vector<double3>& safe,
                          double3 ground_normal,double ground_offset,double delta)
{
    const int count=c.kind==2?1:4;
    for(int j=0;j<count;++j)c.anchor[j]=safe[c.ids[j]];
    if(c.kind==2)
    {
        c.grad[0]=ground_normal;
        c.offset=dot(ground_normal,c.anchor[0])-ground_offset-delta;
        return std::isfinite(c.offset);
    }
    double weights[4]={0,0,0,0};double3 difference;
    if(c.kind==0)
    {
        double u,v;
        difference=sub(c.anchor[0],closest_triangle(c.anchor[0],c.anchor[1],
            c.anchor[2],c.anchor[3],u,v));
        weights[0]=1;weights[1]=u+v-1;weights[2]=-u;weights[3]=-v;
    }
    else
    {
        double u,v;
        closest_segments(c.anchor[0],c.anchor[1],c.anchor[2],c.anchor[3],u,v);
        difference=sub(add(c.anchor[0],mul(sub(c.anchor[1],c.anchor[0]),u)),
            add(c.anchor[2],mul(sub(c.anchor[3],c.anchor[2]),v)));
        weights[0]=1-u;weights[1]=u;weights[2]=v-1;weights[3]=-v;
    }
    const double distance=std::sqrt(dot(difference,difference));
    if(!std::isfinite(distance)||distance<=1e-15)return false;
    const auto normal=mul(difference,1/distance);
    for(int j=0;j<4;++j)c.grad[j]=mul(normal,weights[j]);
    c.offset=distance-delta;
    return true;
}

void audit_safe_blocker(GIPC& g,const double3* target,double alpha,
                       const std::vector<ToiContact>& trial_contacts,
                       gipc::Json& record)
{
    auto& s=g.toi;
    gipc::Json audit={{"ccd_alpha",alpha},{"limited",alpha<1},
        {"trial_contact_count",trial_contacts.size()},
        {"scope","Read-only CCD argmin and copied trial-system contacts before multiplier update; probe planes for missing contacts were not used in the solve."}};
    if(alpha==1){record["blocker_audit"]=std::move(audit);return;}
    const auto times=download(s.pair_times.data(),s.pair_times.size());
    const auto ground_times=download(s.ground_times.data(),s.ground_times.size());
    struct Candidate {double alpha;size_t index;bool ground;};
    std::vector<Candidate> candidates;
    for(size_t i=0;i<times.size();++i)
        if(times[i]<1)candidates.push_back({times[i],i,false});
    for(size_t i=0;i<ground_times.size();++i)
        if(ground_times[i]<1)candidates.push_back({.8*ground_times[i],i,true});
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b)
        {return a.alpha<b.alpha;});
    if(candidates.empty()||std::abs(candidates.front().alpha-alpha)>1e-12)
        throw std::runtime_error("TOI blocker audit disagrees with the GPU CCD minimum");
    audit["minimum_ties_within_1e_minus12"]=std::count_if(candidates.begin(),candidates.end(),
        [&](const auto& c){return std::abs(c.alpha-alpha)<=1e-12;});
    audit["candidate_limit"]=4;
    const auto safe=download(s.safe.data(),g.vertexNum);
    const auto trial=download(target,g.vertexNum);
    const auto normal=download(g._groundNormal.data(),1)[0];
    const auto offset=download(g._groundOffset.data(),1)[0];
    std::map<Key,ToiContact> solved,updated;
    for(const auto& c:trial_contacts)solved.emplace(key(c),c);
    for(const auto& c:s.host_contacts)updated.emplace(key(c),c);
    std::set<Key> seen;
    auto entries=gipc::Json::array();
    for(const auto& candidate:candidates)
    {
        ToiContact c;
        if(candidate.ground)
        {c.kind=2;c.ids[0]=download(g._surfVerts.data()+candidate.index,1)[0];}
        else c=canonical(download(g._ccd_collisonPairs.data()+candidate.index,1)[0]);
        const auto contact_key=key(c);
        if(!seen.insert(contact_key).second)continue;
        const auto before=solved.find(contact_key),after=updated.find(contact_key);
        const bool in_trial=before!=solved.end(),after_update=after!=updated.end();
        bool plane_valid=true;
        if(in_trial)c=before->second;
        else plane_valid=blocker_linear_probe(c,safe,normal,offset,s.delta);
        const double safe_c=plane_valid?constraint(c,safe.data()):0;
        const double trial_c=plane_valid?constraint(c,trial.data()):0;
        const double mu=c.penalty_mu>0?c.penalty_mu*s.mu/s.initial_mu:s.mu;
        gipc::Json entry={{"kind",c.kind},
            {"ids",gipc::Json::array({c.ids[0],c.ids[1],c.ids[2],c.ids[3]})},
            {"pair_safe_alpha",candidate.alpha},
            {"is_global_minimum",std::abs(candidate.alpha-alpha)<=1e-12},
            {"in_trial_system",in_trial},{"in_active_set_after_update",after_update},
            {"plane_valid",plane_valid},
            {"plane_scope",in_trial?"plane used by this trial solve":"diagnostic safe-state probe only"},
            {"safe_unsigned_distance_m",blocker_distance(c,safe,normal,offset)},
            {"trial_unsigned_distance_m",blocker_distance(c,trial,normal,offset)},
            {"delta_m",s.delta},{"effective_mu",mu}};
        entry["constraint_safe_m"]=plane_valid?gipc::Json(safe_c):gipc::Json(nullptr);
        entry["constraint_trial_m"]=plane_valid?gipc::Json(trial_c):gipc::Json(nullptr);
        entry["lambda_used"]=in_trial?gipc::Json(c.lambda):gipc::Json(nullptr);
        entry["gamma_used"]=in_trial?gipc::Json(c.gamma):gipc::Json(nullptr);
        entry["slack_last_assembly_m"]=in_trial?gipc::Json(c.slack):gipc::Json(nullptr);
        entry["lambda_after_update"]=after_update?gipc::Json(after->second.lambda):gipc::Json(nullptr);
        entry["gamma_after_update"]=after_update?gipc::Json(after->second.gamma):gipc::Json(nullptr);
        entry["slack_after_update_m"]=after_update?gipc::Json(after->second.slack):gipc::Json(nullptr);
        if(in_trial)
        {
            auto probe=c;
            if(blocker_linear_probe(probe,safe,normal,offset,s.delta))
            {
                double error=std::abs(probe.offset-c.offset);
                for(int j=0;j<(c.kind==2?1:4);++j)
                    error=std::max({error,std::abs(probe.grad[j].x-c.grad[j].x),
                        std::abs(probe.grad[j].y-c.grad[j].y),std::abs(probe.grad[j].z-c.grad[j].z)});
                entry["host_probe_vs_solver_plane_max_abs_error"]=error;
            }
            if(after_update)
            {
                const double expected_slack=std::max(0.,trial_c-c.lambda/mu);
                const double expected_lambda=expected_slack==0?c.lambda-mu*trial_c:0;
                const double expected_gamma=expected_slack==0?1:c.gamma*.9;
                entry["multiplier_update_relative_error"]=std::abs(after->second.lambda-expected_lambda)
                    /std::max({1.,std::abs(after->second.lambda),std::abs(expected_lambda)});
                entry["slack_update_abs_error_m"]=std::abs(after->second.slack-expected_slack);
                entry["gamma_update_abs_error"]=std::abs(after->second.gamma-expected_gamma);
            }
        }
        entry["classification"]=!in_trial?(after_update?"added_after_trial":"missing_after_active_update")
            :!plane_valid?"invalid_probe":trial_c<-1e-10?"active_linear_violation":"active_linear_satisfied_geometric_block";
        entries.push_back(std::move(entry));
        if(entries.size()==4)break;
    }
    audit["candidates"]=std::move(entries);
    record["blocker_audit"]=std::move(audit);
}

double normalized_volume_step(GIPC& g,device_TetraData& mesh,const double3* start,
                              const double3* end,double retain)
{
    if(!g.tetrahedraNum)return 1;
    auto& s=g.toi;s.energy.resize(g.tetrahedraNum);s.scalar.resize(1);
    g._scalar_scratch.resize(1);CUDA_SAFE_CALL(cudaMemset(g._scalar_scratch.data(),0,sizeof(int)));
    volume_step_bounds<<<(g.tetrahedraNum+255)/256,256>>>(start,g._moveDir,end,g._rest_vertexes,
        mesh.tetrahedras,g.tetrahedraNum,retain,s.energy.data(),g._scalar_scratch.data());
    int invalid=download(g._scalar_scratch.data(),1)[0];
    if(invalid)throw std::runtime_error("TOI volume bound: invalid or already inverted safe tetrahedron");
    cudatool::DeviceReduce().Reduce(s.energy.data(),s.scalar.data(),g.tetrahedraNum,ToiMinimum{},1.);
    return download(s.scalar.data(),1)[0];
}
// Safe state is already bound to vertexes. Returns a conservative unfiltered
// full-scene step and, optionally, updates C using the previous trial state.
double swept_query(GIPC& g,const double3* target,bool update_set,gipc::Json& record,
                   const std::vector<ToiContact>* trial_contacts=nullptr)
{
    auto& s=g.toi;
    make_direction<<<(g.vertexNum+255)/256,256>>>(s.safe.data(),target,g._moveDir,g.vertexNum);
    save_ccd_geometry(g,target,update_set,record);
    const gipc::Json context={{"outer",record.value("outer",-1)},{"update_set",update_set}};
    stage_event("ccd_bvh_begin",context);
    g.buildBVH_FULLCCD(1);
    stage_event("ccd_broad_begin",context);
    g.buildFullCP(1);
    stage_event("ccd_narrow_begin",{{"outer",record.value("outer",-1)},{"pairs",g.h_ccd_cpNum},{"update_set",update_set}});
    s.pair_times.resize(g.h_ccd_cpNum);s.ground_times.resize(g.surf_vertexNum);
    double ratio=.2;
    if(update_set && std::getenv("GIPC_TOI_FILTER_RATIO"))ratio=std::stod(std::getenv("GIPC_TOI_FILTER_RATIO"));
    if(!std::isfinite(ratio)||ratio<=0||ratio>=1)throw std::runtime_error("TOI filter ratio must lie in (0,1)");
    if(s.robust_port&&ratio!=.2)
        throw std::runtime_error("Shared Robust CCD requires the full safety query ratio 0.2");
    record[update_set?"active_filter_accd_ratio":"safe_accd_ratio"]=ratio;
    if(g.h_ccd_cpNum)pair_ccd<<<(g.h_ccd_cpNum+255)/256,256>>>(g._vertexes,g._moveDir,g._ccd_collisonPairs.data(),g.h_ccd_cpNum,s.pair_times.data(),ratio);
    if(g.surf_vertexNum)ground_ccd<<<(g.surf_vertexNum+255)/256,256>>>(g._vertexes,g._moveDir,g._surfVerts.data(),g.surf_vertexNum,g._groundNormal.data(),g._groundOffset.data(),s.ground_times.data());
    stage_event("ccd_narrow_end",context);
    if(!update_set)
    {
        g._scalar_scratch.resize(1);CUDA_SAFE_CALL(cudaMemset(g._scalar_scratch.data(),0,sizeof(int)));
        if(s.pair_times.size())validate_times<<<(s.pair_times.size()+255)/256,256>>>(s.pair_times.data(),s.pair_times.size(),g._scalar_scratch.data());
        if(s.ground_times.size())validate_times<<<(s.ground_times.size()+255)/256,256>>>(s.ground_times.data(),s.ground_times.size(),g._scalar_scratch.data());
        if(download(g._scalar_scratch.data(),1)[0])throw std::runtime_error("Invalid pair or ground TOI");
        s.scalar.resize(1);double alpha=1;
        if(s.pair_times.size())
        {cudatool::DeviceReduce().Reduce(s.pair_times.data(),s.scalar.data(),s.pair_times.size(),ToiMinimum{},1.);alpha=download(s.scalar.data(),1)[0];}
        if(s.ground_times.size())
        {cudatool::DeviceReduce().Reduce(s.ground_times.data(),s.scalar.data(),s.ground_times.size(),ToiMinimum{},1.);double t=download(s.scalar.data(),1)[0];if(t<1)alpha=std::min(alpha,.8*t);}
        if(!std::isfinite(alpha)||alpha<0||alpha>1)throw std::runtime_error("Invalid full CCD minimum TOI");
        record["safe_ccd_broad_pairs"]=g.h_ccd_cpNum;
        if(trial_contacts)audit_safe_blocker(g,target,alpha,*trial_contacts,record);
        stage_event("ccd_query_end",{{"alpha",alpha},{"outer",record.value("outer",-1)},{"update_set",update_set}});
        return alpha;
    }
    auto times=download(s.pair_times.data(),s.pair_times.size());
    auto ground_times=download(s.ground_times.data(),s.ground_times.size());
    stage_event("ccd_active_cpu_begin",context);
    double alpha=1;
    for(double t:times) {if(!std::isfinite(t)||t<0||t>1)throw std::runtime_error("Invalid pair TOI");alpha=std::min(alpha,t);}
    for(double t:ground_times) {if(!std::isfinite(t)||t<0||t>1)throw std::runtime_error("Invalid ground TOI");if(t<1)alpha=std::min(alpha,.8*t);}
    record[update_set?"active_update_broad_pairs":"safe_ccd_broad_pairs"]=g.h_ccd_cpNum;
    if(!update_set)return alpha;
    auto raw=download(g._ccd_collisonPairs.data(),g.h_ccd_cpNum);
    auto surface=download(g._surfVerts.data(),g.surf_vertexNum);
    std::set<Key> existing;
    for(const auto& c:s.host_contacts)
        if(!s.robust_port||toi_port::retain_contact(c.release_age))existing.insert(key(c));
    std::map<Key,std::pair<ToiContact,double>> candidates;
    for(size_t i=0;i<raw.size();++i)if(times[i]<1)
    {auto c=canonical(raw[i]);if(!existing.count(key(c)))candidates.emplace(key(c),std::make_pair(c,times[i]));}
    for(size_t i=0;i<surface.size();++i)if(ground_times[i]<1)
    {ToiContact c;c.kind=2;c.ids[0]=surface[i];if(!existing.count(key(c)))candidates.emplace(key(c),std::make_pair(c,ground_times[i]));}
    std::vector<double> minimum(g.vertexNum,1.);
    if(s.robust_port)
    {
        // Existing constraints participate in earliest-vertex filtering too.
        for(size_t i=0;i<raw.size();++i)if(times[i]<1)
        {const auto c=canonical(raw[i]);for(int j=0;j<4;++j)minimum[c.ids[j]]=std::min(minimum[c.ids[j]],times[i]);}
        for(size_t i=0;i<surface.size();++i)if(ground_times[i]<1)
            minimum[surface[i]]=std::min(minimum[surface[i]],ground_times[i]);
    }
    else for(const auto& entry:candidates)
    {const auto& c=entry.second.first;for(int j=0;j<(c.kind==2?1:4);++j)minimum[c.ids[j]]=std::min(minimum[c.ids[j]],entry.second.second);}
    int added=0;
    for(const auto& entry:candidates)
    {
        const auto& c=entry.second.first;bool keep=false;
        for(int j=0;j<(c.kind==2?1:4);++j)keep=keep || (s.robust_port
            ?toi_port::selected_toi(entry.second.second,minimum[c.ids[j]])
            :entry.second.second==minimum[c.ids[j]]);
        if(keep){auto selected=c;selected.penalty_mu=mass_contact_mu(selected,s.host_vertex_mu);s.host_contacts.push_back(selected);++added;}
    }
    auto old_size=s.host_contacts.size();
    s.host_contacts.erase(std::remove_if(s.host_contacts.begin(),s.host_contacts.end(),[&](const auto& c)
        {return s.robust_port?!toi_port::retain_contact(c.release_age):c.gamma<.01;}),s.host_contacts.end());
    record["active_candidates_new"]=candidates.size();record["active_added"]=added;
    record["active_removed"]=old_size-s.host_contacts.size();
    stage_event("ccd_query_end",{{"alpha",alpha},{"outer",record.value("outer",-1)},{"update_set",update_set},{"active_contacts",s.host_contacts.size()}});
    return alpha;
}

std::vector<ToiContact> frame_friction_copy(const std::vector<ToiContact>& previous)
{
    auto snapshot=previous;
    std::sort(snapshot.begin(),snapshot.end(),[](const auto& a,const auto& b){return key(a)<key(b);});
    for(auto& c:snapshot)c.gamma=1;
    return snapshot;
}

void install_frame_friction(GIPC& g,const std::vector<ToiContact>& previous,const double3* frame_begin)
{
    auto& s=g.toi;
    const bool fresh=s.frame_friction_frame_id!=total_Frames;
    auto snapshot=fresh?frame_friction_copy(previous):download(s.frame_friction_contacts.data(),s.frame_friction_contacts.size());
    std::vector<int4> pairs;std::vector<uint32_t> ground;
    double force_sum=0;
    for(auto& c:snapshot)
    {
        // Public frame lagging snapshots lambda, not the current AL weight.
        force_sum+=std::max(0.,c.lambda);
        if(c.kind==2)ground.push_back(c.ids[0]);else pairs.push_back(encoded(c));
    }
    if(fresh)upload(s.frame_friction_contacts,snapshot);
    if(fresh&&!snapshot.empty())
    {
        g._scalar_scratch.resize(1);const int sentinel=INT_MAX;
        CUDA_SAFE_CALL(cudaMemcpy(g._scalar_scratch.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<(snapshot.size()+255)/256,256>>>(s.frame_friction_contacts.data(),snapshot.size(),frame_begin,
            g._groundNormal.data(),g._groundOffset.data(),s.delta,g._scalar_scratch.data());
        int invalid;CUDA_SAFE_CALL(cudaMemcpy(&invalid,g._scalar_scratch.data(),sizeof(int),cudaMemcpyDeviceToHost));
        if(invalid!=sentinel)throw std::runtime_error("Frame friction snapshot has singular previous geometry");
    }
    s.frame_friction_frame_id=total_Frames;
    std::fill(g.h_cpNum_last,g.h_cpNum_last+5,0);
    const int self=pairs.size(),ng=ground.size();g.h_cpNum_last[0]=g.h_cpNum_last[4]=self;g.h_gpNum_last=ng;
    upload(g._collisonPairs_lastH,pairs);upload(g._collisonPairs_lastH_gd,ground);
    g.lambda_lastH_scalar.resize(self);g.distCoord.resize(self);g.tanBasis.resize(self);g._MatIndex_last.resize(self);
    if(self)friction_basis<<<(self+255)/256,256>>>(s.frame_friction_contacts.data(),self,g.lambda_lastH_scalar.data(),
        g.distCoord.data(),g.tanBasis.data(),false);
    g.lambda_lastH_scalar_gd.resize(ng);
    if(ng)ground_lambda<<<(ng+255)/256,256>>>(s.frame_friction_contacts.data(),self,ng,g.lambda_lastH_scalar_gd.data(),false);
    auto& frame=gipc::Statistics::instance().at_current_frame();
    frame["toi_frame_friction_snapshot"]={{"self",self},{"ground",ng},{"lambda_sum",force_sum},
        {"refresh_policy","frame_start_previous_geometry"},{"force_policy","previous_lambda"},
        {"physical_frame",total_Frames},{"captured_this_solve",fresh}};
}

std::string frame_friction_signature(GIPC& g)
{
    // Read the buffers actually consumed by friction assembly. Matrix offsets
    // are scratch and deliberately excluded; contact geometry and forces are not.
    uint64_t hash=14695981039346656037ull;
    auto bytes=[&](const void* data,size_t count)
    {
        const auto* p=static_cast<const unsigned char*>(data);
        for(size_t i=0;i<count;++i){hash^=p[i];hash*=1099511628211ull;}
    };
    auto buffer=[&](const auto& value,size_t count)
    {
        bytes(&count,sizeof(count));
        std::vector<unsigned char> host(count*sizeof(*value.data()));
        if(!host.empty())CUDA_SAFE_CALL(cudaMemcpy(host.data(),value.data(),host.size(),cudaMemcpyDeviceToHost));
        bytes(host.data(),host.size());
    };
    const size_t self=g.h_cpNum_last[0],ground=g.h_gpNum_last;
    buffer(g._collisonPairs_lastH,self);buffer(g._collisonPairs_lastH_gd,ground);
    buffer(g.lambda_lastH_scalar,self);buffer(g.lambda_lastH_scalar_gd,ground);
    buffer(g.distCoord,self);buffer(g.tanBasis,self);
    return std::to_string(hash);
}
}

namespace {
bool reduced_slack_enabled()
{
    static const bool enabled=std::getenv("GIPC_TOI_REDUCED_SLACK")
        && std::string(std::getenv("GIPC_TOI_REDUCED_SLACK"))=="1";
    return enabled;
}
}
void GIPC::toiSelfGradientHessian(double3* g)
{
    if(!toi.self_count)return;
    al_hessian<<<(toi.self_count+255)/256,256>>>(toi.contacts.data(),0,toi.self_count,_vertexes,g,
        gipc_global_triplet.block_values(),gipc_global_triplet.block_row_indices(),gipc_global_triplet.block_col_indices(),0,toi.mu,toi.initial_mu,reduced_slack_enabled());
}
void GIPC::toiGroundGradientHessian(double3* g)
{
    if(!toi.ground_count)return;
    al_hessian<<<(toi.ground_count+255)/256,256>>>(toi.contacts.data(),toi.self_count,toi.ground_count,_vertexes,g,
        gipc_global_triplet.block_values(),gipc_global_triplet.block_row_indices(),gipc_global_triplet.block_col_indices(),gipc_global_triplet.global_triplet_offset,toi.mu,toi.initial_mu,reduced_slack_enabled());
}
double GIPC::toiContactEnergy()
{
    int n=toi.contacts.size();if(!n)return 0;
    toi.energy.resize(n);toi.scalar.resize(1);
    al_energy<<<(n+255)/256,256>>>(toi.contacts.data(),n,_vertexes,toi.mu,toi.initial_mu,toi.energy.data(),reduced_slack_enabled());
    cudatool::DeviceReduce().Sum(toi.energy.data(),toi.scalar.data(),n);
    double value;CUDA_SAFE_CALL(cudaMemcpy(&value,toi.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));return value;
}
void GIPC::toiFrictionSets()
{
    std::fill(h_cpNum_last,h_cpNum_last+5,0);h_gpNum_last=0;
#ifdef USE_FRICTION
    int n=toi.self_count;
    _collisonPairs_lastH.resize(n);lambda_lastH_scalar.resize(n);distCoord.resize(n);tanBasis.resize(n);_MatIndex_last.resize(n);
    copy_d(_collisonPairs_lastH.data(),_collisonPairs.data(),n);
    h_cpNum_last[0]=h_cpNum_last[4]=n;
    if(n)friction_basis<<<(n+255)/256,256>>>(toi.contacts.data(),n,lambda_lastH_scalar.data(),distCoord.data(),tanBasis.data(),toi.independent_friction_history);
    n=toi.ground_count;h_gpNum_last=n;
    lambda_lastH_scalar_gd.resize(n);_collisonPairs_lastH_gd.resize(n);
    copy_d(_collisonPairs_lastH_gd.data(),_environment_collisionPair.data(),n);
    if(n)ground_lambda<<<(n+255)/256,256>>>(toi.contacts.data(),toi.self_count,n,lambda_lastH_scalar_gd.data(),toi.independent_friction_history);
#endif
}

int GIPC::solve_subTOI(device_TetraData& mesh)
{
    auto& s=toi;auto& frame=gipc::Statistics::instance().at_current_frame();
#ifdef USE_SNK1
    const bool material_requires_noninversion=false;
#else
    const bool material_requires_noninversion=true;
#endif
    const auto options=gipc::ToiOptions::from_environment(Newton_solver_threshold,material_requires_noninversion);
    gipc::set_solve_frame(total_Frames+1);
    s.robust_port=options.robust_port;frame["toi_policy"]=options.policy;
    frame["newton"]=gipc::Json::array();frame["toi"]=gipc::Json::array();frame["contact_backend"]="toi_al";
    // Json is ordered_json: inserting parent keys may deep-copy child arrays
    // and invalidate references into them. Create audit keys before retaining
    // the outer record across assembly / CCD / energy calls.
    frame["energy_batch_audit"]=gipc::Json::object();
    frame["bvh_refit_audit"]=gipc::Json::object();
    frame["toi_exit"]="in_progress";
    const bool clamp_trial=options.clamp_trial;
    frame["trial_injectivity_clamp"]=clamp_trial;
    // Match the base material policy: Stable NH1 is defined through J=0,
    // and the official base does not enable the injectivity step bound.
    // Requiring positive volume is an explicit additional-constraint ablation.
#ifdef USE_SNK1
    frame["fem_model"]="stable_nh1";
    frame["material_requires_noninversion"]=false;
#else
    frame["fem_model"]="other";
    frame["material_requires_noninversion"]=true;
#endif
    const bool clamp_safe=options.clamp_safe;
    frame["safe_injectivity_clamp"]=clamp_safe;
    const bool persist_contacts=options.persist_contacts;
    const bool keep_lambda=options.keep_lambda;
    const bool keep_gamma=options.keep_gamma;
    s.independent_friction_history=options.independent_friction_history;
    const bool keep_friction=options.keep_friction;
    frame["toi_cross_frame_contacts"]=persist_contacts;
    frame["toi_warm_history"]={{"contacts",persist_contacts},{"lambda",keep_lambda},{"gamma",keep_gamma},
        {"friction_mode",s.robust_port?"frame_snapshot_previous_lambda":s.independent_friction_history?"independent_proxy":"coupled_al"},
        {"friction",s.independent_friction_history?gipc::Json(keep_friction):gipc::Json(nullptr)}};
    const bool legacy_volume=options.legacy_volume;
    frame["safe_volume_bound_mode"]=legacy_volume?"legacy":"normalized";
    const bool reuse_initial=options.reuse_initial;
    // Optional public Robust displacement safeguard. Keep the paper's r=1
    // stopping rule as the default, and label this implementation ablation.
    const double robust_velocity_tol=options.trial_velocity_tol;
    const bool robust_velocity_stop=options.velocity_stop;
    const std::string& inner_exit=options.inner_exit;
    frame["robust_trial_velocity_tol"]=robust_velocity_tol;
    frame["toi_inner_stop_policy"]=robust_velocity_stop?"robust_velocity":"full_step";
    frame["diagnostic_inner_exit_override"]=inner_exit;
    frame["toi_reduced_slack"]=reduced_slack_enabled();
    frame["toi_remaining_fraction_tol"]=options.remaining_fraction_tol;
    const int blocker_from=options.blocker_from,blocker_to=options.blocker_to;
    const bool blocker_audit=blocker_from>0 && total_Frames+1>=blocker_from && total_Frames+1<=blocker_to;
    frame["toi_blocker_audit_enabled"]=blocker_audit;
    s.safe.resize(vertexNum);s.trial.resize(vertexNum);s.previous_trial.resize(vertexNum);
    auto& q=m_abd_sim_data->device.body_id_to_q;
    s.safe_q.resize(q.size());s.trial_q.resize(q.size());
    copy_d(s.safe.data(),_vertexes,vertexNum);copy_d(s.trial.data(),_vertexes,vertexNum);
    trace_safe(*this,0);
    copy_d(s.safe_q.data(),q.data(),q.size());copy_d(s.trial_q.data(),q.data(),q.size());
    auto previous_contacts=persist_contacts?std::move(s.host_contacts):std::vector<ToiContact>{};
    const auto friction_previous=s.robust_port?previous_contacts:std::vector<ToiContact>{};
    s.host_contacts.clear();s.contacts.clear();s.mu=1;
    frame["toi_previous_frame_active_count"]=previous_contacts.size();
    prepare_warm_contacts(previous_contacts,keep_lambda,keep_gamma,s.independent_friction_history,keep_friction);
    double warm_lambda_sum=0,warm_friction_sum=0,warm_proxy_difference=0;int warm_lambda_positive=0,warm_gamma_decayed=0;
    for(const auto& c:previous_contacts)
    {
        warm_lambda_sum+=c.lambda;warm_lambda_positive+=c.lambda>0;warm_gamma_decayed+=c.gamma<1;
        warm_friction_sum+=std::max(0.,s.independent_friction_history?c.friction_lambda*c.friction_gamma:c.lambda*c.gamma);
        if(s.independent_friction_history)warm_proxy_difference=std::max({warm_proxy_difference,
            std::abs(c.friction_lambda-c.lambda),std::abs(c.friction_gamma-c.gamma)});
    }
    frame["toi_warm_start_state"]={{"contacts",previous_contacts.size()},{"lambda_sum",warm_lambda_sum},
        {"positive_lambda_contacts",warm_lambda_positive},{"gamma_below_one_contacts",warm_gamma_decayed},
        {"friction_force_proxy_sum",warm_friction_sum},{"independent_proxy_vs_al_max_abs_difference",warm_proxy_difference}};
    s.delta=1e-4*std::sqrt(bboxDiagSize2);
    if(const char* v=std::getenv("GIPC_TOI_DELTA"))s.delta=std::stod(v);
    if(!std::isfinite(s.delta)||s.delta<=0)throw std::runtime_error("TOI delta must be finite and positive");
    if(std::getenv("GIPC_RESOLVED_CONFIG"))
    {
        auto resolved=gipc::resolved_execution_options("toi_al",IPC_dt,Newton_solver_threshold,pcg_threshold);
        resolved["toi"]=options.to_json();
        resolved["toi"]["contact_delta_m"]=s.delta;
        resolved["toi"]["delta_source"]=std::getenv("GIPC_TOI_DELTA")?"GIPC_TOI_DELTA":"1e-4_times_initial_bbox_diagonal";
        resolved["toi_remaining_fraction_tol"]=options.remaining_fraction_tol;
        resolved["toi_trial_velocity_tol_m_s"]=options.trial_velocity_tol;
        resolved["material_requires_noninversion"]=material_requires_noninversion;
        const char* preconditioner=std::getenv("GIPC_PCG_PRECONDITIONER");
        const std::string effective_preconditioner=preconditioner?preconditioner:(pcg_data.P_type==1?"mas":"diag");
        resolved["effective_preconditioner"]=effective_preconditioner;
        resolved["mas"]["active"]=effective_preconditioner=="mas";
        gipc::write_resolved_config(resolved);
    }
    publish_contacts(*this);toiFrictionSets();
    // Estimate mu from the contact-free projected energy Hessian (paper Eq20).
    computeGradientAndHessian(mesh);
    int nh=gipc_global_triplet.global_triplet_offset;
    int diagonal_dofs=3*(abd_fem_count_info.abd_body_num*4+abd_fem_count_info.fem_point_num);
    s.energy.resize(diagonal_dofs);s.scalar.resize(1);
    const std::string& mu_scope=options.mu_scope;
    frame["toi_penalty_scope"]=mu_scope;
    CUDA_SAFE_CALL(cudaMemset(s.energy.data(),0,diagonal_dofs*sizeof(double)));
    if(nh)
    {
        diagonal_max<<<(nh+255)/256,256>>>(gipc_global_triplet.block_values(),gipc_global_triplet.block_row_indices(),gipc_global_triplet.block_col_indices(),nh,s.energy.data());
        if(mu_scope=="movable")movable_diagonal<<<(diagonal_dofs+255)/256,256>>>(s.energy.data(),diagonal_dofs,
            12*abd_fem_count_info.abd_body_num,mesh.body_id_to_boundary_type+abd_fem_count_info.abd_body_offset,
            mesh.BoundaryType.data()+abd_fem_count_info.fem_point_offset);
        cudatool::DeviceReduce().Max(s.energy.data(),s.scalar.data(),diagonal_dofs);
        double diagonal;CUDA_SAFE_CALL(cudaMemcpy(&diagonal,s.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));s.mu=.1*diagonal;
        if(std::getenv("GIPC_TOI_DIAGNOSTIC_LOG"))
        {
            auto diagonal_entries=download(s.energy.data(),diagonal_dofs);
            const int abd_dofs=12*abd_fem_count_info.abd_body_num;
            double abd_max=0,fem_max=0;
            for(int i=0;i<diagonal_dofs;++i)
                (i<abd_dofs?abd_max:fem_max)=std::max(i<abd_dofs?abd_max:fem_max,diagonal_entries[i]);
            frame["toi_initial_hessian_diagonal"]={{"abd_max",abd_max},{"fem_max",fem_max},{"global_max",diagonal},
                {"scope",mu_scope=="movable"?"absolute assembled generalized diagonal, fixed ABD/FEM excluded":
                    "assembled generalized coordinates, including boundary contributions"}};
        }
    }
    if(!std::isfinite(s.mu)||s.mu<=0)throw std::runtime_error("TOI initial stiffness estimate failed");
    frame["toi_penalty_coordinates"]=options.mu_coordinates;
    if(options.mu_coordinates=="world_block")
    {
        const double generalized_mu=s.mu;
        const int bodies=abd_fem_count_info.abd_body_num,points=abd_fem_count_info.abd_point_num;
        fem_only_diagonal<<<(diagonal_dofs+255)/256,256>>>(s.energy.data(),diagonal_dofs,12*bodies);
        cudatool::DeviceReduce().Max(s.energy.data(),s.scalar.data(),diagonal_dofs);
        double fem_max=0,abd_max=0;
        CUDA_SAFE_CALL(cudaMemcpy(&fem_max,s.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));
        if(points)
        {
            auto h=download(m_abd_system->abd_body_hessian.data(),bodies);
            const auto boundary=download(mesh.body_id_to_boundary_type+abd_fem_count_info.abd_body_offset,bodies);
            for(int body=0;body<bodies;++body)
                h[body]=boundary[body]==BodyBoundaryType::Fixed?gipc::Matrix12x12::Zero():gipc::toi_body_compliance(h[body]);
            cudatool::DeviceBuffer<gipc::Matrix12x12> inverse;
            cudatool::DeviceBuffer<double> world_diagonal(points);
            cudatool::DeviceBuffer<int> invalid_vertex(1);
            upload(inverse,h);
            CUDA_SAFE_CALL(cudaMemset(invalid_vertex.data(),0,sizeof(int)));
            world_abd_diagonal<<<(points+255)/256,256>>>(m_abd_sim_data->device.unique_point_id_to_J.data(),
                mesh.point_id_to_body_id.data()+abd_fem_count_info.abd_point_offset,
                mesh.body_id_to_boundary_type+abd_fem_count_info.abd_body_offset,inverse.data(),bodies,points,
                world_diagonal.data(),invalid_vertex.data());
            cudatool::DeviceReduce().Max(world_diagonal.data(),s.scalar.data(),points);
            CUDA_SAFE_CALL(cudaMemcpy(&abd_max,s.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));
            int invalid=0;CUDA_SAFE_CALL(cudaMemcpy(&invalid,invalid_vertex.data(),sizeof(int),cudaMemcpyDeviceToHost));
            if(invalid)throw std::runtime_error("TOI world penalty: invalid vertex compliance");
        }
        s.mu=.1*std::max(fem_max,abd_max);
        if(!std::isfinite(s.mu)||s.mu<=0)throw std::runtime_error("TOI world stiffness estimate failed");
        frame["toi_world_hessian_diagonal"]={{"fem_max",fem_max},{"abd_max",abd_max},
            {"generalized_mu",generalized_mu},{"world_mu",s.mu},
            {"method","full contact-free ABD block compliance pulled into world vertex coordinates; FEM diagonal unchanged"}};
    }
    const double mu_scale=options.mu_scale;
    s.mu*=mu_scale;
    if(!std::isfinite(s.mu)||s.mu<=0)throw std::runtime_error("TOI stiffness scale must be positive");
    s.initial_mu=s.mu;
    const std::string& mu_mode=options.mu_mode;
    frame["toi_penalty_mode"]=mu_mode;
    s.host_vertex_mu.clear();
    if(mu_mode=="mass")
    {
        // Public Robust's optional mass estimator: FEM vertex mass and ABD
        // whole-body mass are first converted to a per-vertex stiffness.
        const auto masses=download(mesh.masses.data(),vertexNum);
        const auto body_ids=download(mesh.point_id_to_body_id.data(),vertexNum);
        const auto body_masses=download(m_abd_system->body_mass.data(),abd_fem_count_info.abd_body_num);
        s.host_vertex_mu.resize(vertexNum);
        const double dt2=IPC_dt*IPC_dt;
        double minimum=std::numeric_limits<double>::infinity(),maximum=0;
        for(int i=0;i<vertexNum;++i)
        {
            const int body=body_ids[i];
            if(body>=int(body_masses.size()))throw std::runtime_error("TOI mass estimator: invalid ABD body ID");
            const double mass=body>=0?body_masses[body]:masses[i];
            const double stiffness=mass*(body>=0?1e5:5e7)*dt2*mu_scale;
            if(!std::isfinite(stiffness)||stiffness<=0)throw std::runtime_error("TOI mass estimator: nonpositive contact stiffness");
            s.host_vertex_mu[i]=stiffness;
            minimum=std::min(minimum,stiffness);maximum=std::max(maximum,stiffness);
        }
        frame["toi_mass_penalty_range"]={{"minimum",minimum},{"maximum",maximum},
            {"fem_factor",5e7},{"abd_factor",1e5},{"dt",IPC_dt},{"global_scale",mu_scale}};
    }
    // The stiffness estimate above is explicitly contact-free. Only after it
    // has been recorded can last frame's multipliers be relinearized at the
    // new safe state and used to snapshot friction for this frame.
    if(persist_contacts)s.host_contacts=std::move(previous_contacts);
    if(s.robust_port)install_frame_friction(*this,friction_previous,mesh.o_vertexes);
    const bool friction_audit=s.robust_port&&history_flag("GIPC_TOI_FRAME_FRICTION_AUDIT",false);
    const std::string friction_signature=friction_audit?frame_friction_signature(*this):"";
    if(s.robust_port)
        frame["toi_frame_friction_snapshot"]["buffer_audit_enabled"]=friction_audit;
    for(auto& contact:s.host_contacts)contact.penalty_mu=mass_contact_mu(contact,s.host_vertex_mu);
    const int stall_window=options.stall_window;
    const bool hold_delta=options.hold_delta;
    frame["toi_stall_policy"]={{"window",stall_window},{"hold_delta",hold_delta}};
    std::ofstream diagnostic;
    if(const char* file=std::getenv("GIPC_TOI_DIAGNOSTIC_LOG"))diagnostic.open(file,std::ios::app);
    const bool triangle_audit=std::getenv("GIPC_TOI_TRIANGLE_AUDIT")
        && std::string(std::getenv("GIPC_TOI_TRIANGLE_AUDIT"))=="1";
    if(triangle_audit && (!diagnostic.is_open() || !std::getenv("GIPC_TOI_CURVATURE_FROM_FRAME")))
        throw std::runtime_error("TOI triangle audit requires curvature diagnostic logging");
    int state_audit_from=0;
    if(const char* from=std::getenv("GIPC_TOI_STATE_AUDIT_FROM_FRAME"))
    {
        state_audit_from=std::stoi(from);
        if(state_audit_from<1)throw std::runtime_error("TOI state audit requires a positive frame");
    }
    auto state_snapshot=[&]()
    {
        std::map<std::string,std::vector<unsigned char>> result;
        auto device=[&](const std::string& name,const void* pointer,size_t bytes)
        {
            auto& value=result[name];value.resize(bytes);
            if(bytes)CUDA_SAFE_CALL(cudaMemcpy(value.data(),pointer,bytes,cudaMemcpyDeviceToHost));
        };
        auto buffer=[&](const std::string& name,const auto& value)
        {device(name,value.data(),value.size()*sizeof(*value.data()));};
        device("vertices",_vertexes,vertexNum*sizeof(double3));
        device("direction",_moveDir,vertexNum*sizeof(double3));
        buffer("q",q);buffer("q_temp",m_abd_sim_data->device.body_id_to_q_temp);
        buffer("dq",m_abd_sim_data->device.body_id_to_dq);buffer("temp_vertices",mesh.temp_double3Mem);
        buffer("safe",s.safe);buffer("trial",s.trial);buffer("previous_trial",s.previous_trial);
        buffer("safe_q",s.safe_q);buffer("trial_q",s.trial_q);buffer("contacts",s.contacts);
        device("contact_ids",_collisonPairs.data(),s.self_count*sizeof(int4));
        device("ground_ids",_environment_collisionPair.data(),s.ground_count*sizeof(uint32_t));
        buffer("friction_ids",_collisonPairs_lastH);buffer("ground_friction_ids",_collisonPairs_lastH_gd);
        buffer("friction_lambda",lambda_lastH_scalar);buffer("ground_friction_lambda",lambda_lastH_scalar_gd);
        buffer("friction_coordinates",distCoord);buffer("friction_basis",tanBasis);buffer("friction_matrix_ids",_MatIndex_last);
        auto& host=result["host_contacts"];host.resize(s.host_contacts.size()*sizeof(ToiContact));
        if(!host.empty())std::memcpy(host.data(),s.host_contacts.data(),host.size());
        const std::array<double,5> scalars={s.mu,s.initial_mu,s.delta,frictionRate,gd_frictionRate};
        auto& scalar_bytes=result["scalars"];scalar_bytes.resize(sizeof(scalars));
        std::memcpy(scalar_bytes.data(),scalars.data(),sizeof(scalars));
        const std::array<uint32_t,12> counts={h_cpNum[0],h_cpNum[1],h_cpNum[2],h_cpNum[3],h_cpNum[4],h_gpNum,
            h_cpNum_last[0],h_cpNum_last[1],h_cpNum_last[2],h_cpNum_last[3],h_cpNum_last[4],h_gpNum_last};
        auto& count_bytes=result["counts"];count_bytes.resize(sizeof(counts));
        std::memcpy(count_bytes.data(),counts.data(),sizeof(counts));
        return result;
    };
    std::vector<uint3> audit_triangles;
    std::vector<__GEIGEN__::Matrix2x2d> audit_inverse_rest;
    std::vector<double> audit_area;
    Key watched_key{};bool watch_contact=false;
    if(const char* value=std::getenv("GIPC_TOI_WATCH_CONTACT"))
    {
        int ids[4];
        if(std::sscanf(value,"%d,%d,%d,%d",&ids[0],&ids[1],&ids[2],&ids[3])!=4)
            throw std::runtime_error("TOI watched PT contact must contain four comma-separated vertex IDs");
        std::sort(ids+1,ids+4);
        watched_key={0,ids[0],ids[1],ids[2],ids[3]};watch_contact=true;
    }
    auto watched_state=[&]()->gipc::Json
    {
        if(!watch_contact)return gipc::Json(nullptr);
        auto it=std::find_if(s.host_contacts.begin(),s.host_contacts.end(),[&](const auto& c){return key(c)==watched_key;});
        if(it==s.host_contacts.end())return {{"active",false}};
        return {{"active",true},{"lambda",it->lambda},{"gamma",it->gamma},
                {"slack",it->slack},{"offset",it->offset}};
    };
    double beta=1;int stalls=0;bool previous_full_step=true;
    const bool movement_exit=options.movement_exit;
    const bool profile=std::getenv("GIPC_PROFILE") && std::string(std::getenv("GIPC_PROFILE"))=="1";
    auto stamp=[&](){if(profile)CUDA_SAFE_CALL(cudaDeviceSynchronize());return std::chrono::steady_clock::now();};
    auto elapsed=[](auto start,auto end){return std::chrono::duration<double,std::milli>(end-start).count();};
    const char* event_dir=std::getenv("GIPC_DIRECTION_EVENT_DIR");
    const int event_from=std::getenv("GIPC_DIRECTION_EVENT_FROM")?std::stoi(std::getenv("GIPC_DIRECTION_EVENT_FROM")):30;
    const int event_first=std::getenv("GIPC_DIRECTION_EVENT_FIRST")?std::stoi(std::getenv("GIPC_DIRECTION_EVENT_FIRST")):8;
    const int event_second=std::getenv("GIPC_DIRECTION_EVENT_SECOND")?std::stoi(std::getenv("GIPC_DIRECTION_EVENT_SECOND")):32;
    const double event_velocity=std::getenv("GIPC_DIRECTION_EVENT_VELOCITY")?std::stod(std::getenv("GIPC_DIRECTION_EVENT_VELOCITY")):100*robust_velocity_tol;
    static int event_count=0; // Process-wide bounded diagnostics, never simulation state.
    auto event_components=[&]()
    {
        return gipc::Json{{"kinetic",Energy_Add_Reduction_Algorithm(0,mesh)+m_abd_system->cal_abd_kinetic_energy(*m_abd_sim_data)},
            {"abd_shape",m_abd_system->cal_abd_shape_energy(*m_abd_sim_data)},
            {"fem",IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(1,mesh)},
            {"cloth",IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(8,mesh)},
            {"bending",IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(10,mesh)},
            {"constraint",Energy_Add_Reduction_Algorithm(9,mesh)},{"al_contact",toiContactEnergy()},
            {"friction",frictionRate*Energy_Add_Reduction_Algorithm(5,mesh)+gd_frictionRate*Energy_Add_Reduction_Algorithm(6,mesh)}};
    };
    const bool outer_probe=gipc::ToiObserver::outer_enabled(total_Frames+1);
    std::vector<double3> probe_rest,probe_safe_begin,probe_trial;
    std::vector<uint4> probe_tets;
    std::vector<ToiContact> probe_previous_planes,probe_before_dual,probe_after_dual;
    if(outer_probe)
    {probe_rest=download(_rest_vertexes,vertexNum);probe_tets=download(mesh.tetrahedras.data(),tetrahedraNum);}
    auto event_progress=[&](gipc::Json entry)
    {
        if(!event_dir || !event_count)return;
        gipc::ToiObserver::event_progress(event_dir,std::move(entry));
    };
    bool first_contact_audited=false;
    auto event_save=[&](const std::string& prefix,const std::string& name,const void* ptr,size_t bytes)
    {
        std::vector<unsigned char> data(bytes);
        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(data.data(),ptr,bytes,cudaMemcpyDeviceToHost));
        std::ofstream file(prefix+"_"+name+".bin",std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()),bytes);file.close();
        if(!file)throw std::runtime_error("Failed to save v49 event state");
    };
    std::string pending_update_prefix;
    for(int k=0;k<10000;++k)
    {
        gipc::set_solve_outer(k);
        frame["toi"].push_back(gipc::Json::object());auto& rec=frame["toi"].back();
        gipc::attach_solve_context(rec);
        rec["outer"]=k;rec["mu"]=s.mu;rec["delta"]=s.delta;
        stage_event("outer_begin",{{"outer",k}});
        auto begin_outer=stamp();
        copy_d(s.previous_trial.data(),s.trial.data(),vertexNum);
        if(!pending_update_prefix.empty())
        {
            std::ofstream file(pending_update_prefix+"_next_host.bin",std::ios::binary);
            file.write(reinterpret_cast<const char*>(s.host_contacts.data()),s.host_contacts.size()*sizeof(ToiContact));
            file.close();if(!file)throw std::runtime_error("Failed to save next contact set");
            event_save(pending_update_prefix,"next_safe",s.safe.data(),vertexNum*sizeof(double3));
        }
        publish_contacts(*this);linearize_contacts(*this,k);if(!s.robust_port)toiFrictionSets();
        if(outer_probe)
        {
            const auto planes=download(s.contacts.data(),s.contacts.size());
            probe_safe_begin=download(s.safe.data(),vertexNum);
            const auto warm=download(s.trial.data(),vertexNum);
            gipc::ToiObserver::outer_event({{"stage","model_begin"},{"outer",k},
                {"plane_change",outer_plane_probe(probe_previous_planes,planes,warm,s.delta)},
                {"safe_shape",outer_shape_probe(probe_safe_begin,warm,probe_rest,probe_tets,abd_fem_count_info.abd_point_num)},
                {"warm_shape",outer_shape_probe(warm,probe_safe_begin,probe_rest,probe_tets,abd_fem_count_info.abd_point_num)}});
            probe_previous_planes=planes;
        }
        if(!pending_update_prefix.empty())
        {
            event_save(pending_update_prefix,"next_linearized",s.contacts.data(),s.contacts.size()*sizeof(ToiContact));
            std::ofstream file(pending_update_prefix+"_next_linearized.json");
            file<<gipc::Json{{"frame",total_Frames+1},{"outer",k},{"mu",s.mu},{"initial_mu",s.initial_mu},
                {"delta",s.delta},{"contacts",s.contacts.size()},{"contact_bytes",sizeof(ToiContact)}}.dump(2);
            file.close();if(!file)throw std::runtime_error("Failed to commit next linearization");
            pending_update_prefix.clear();
        }
        rec["friction_snapshot_refresh"] = !s.robust_port;
        copy_d(_vertexes,s.trial.data(),vertexNum);copy_d(q.data(),s.trial_q.data(),q.size());
        const bool choose_start=options.choose_start;
        const bool restart_guard=options.restart_guard;
        bool restarted_from_safe=false;
        if(choose_start && s.robust_port && !s.host_contacts.empty())
        {
            // A new set of affine contact planes changes the subproblem.
            // Compare two initial guesses using the SAME complete objective,
            // each with its exact nonnegative slack minimizer. Subsequent
            // Newton iterations retain the original fixed-slack model.
            const int nc=s.contacts.size();
            update_slack<<<(nc+255)/256,256>>>(s.contacts.data(),nc,_vertexes,s.mu,s.initial_mu,false);
            const double warm_energy=computeEnergy(mesh);
            copy_d(_vertexes,s.safe.data(),vertexNum);copy_d(q.data(),s.safe_q.data(),q.size());
            update_slack<<<(nc+255)/256,256>>>(s.contacts.data(),nc,_vertexes,s.mu,s.initial_mu,false);
            const double safe_energy=computeEnergy(mesh);
            const bool use_safe=std::isfinite(safe_energy)
                && (!std::isfinite(warm_energy) || safe_energy<warm_energy);
            restarted_from_safe=use_safe;
            if(!use_safe)
            {copy_d(_vertexes,s.trial.data(),vertexNum);copy_d(q.data(),s.trial_q.data(),q.size());}
            // The first inner iteration always refreshes slack for the winner.
            rec["initial_guess_selection"]={{"warm_energy",warm_energy},{"safe_energy",safe_energy},
                {"selected",use_safe?"safe":"warm"},{"full_objective",true}};
        }
        rec["restart_minimum_inner_iterations"]=(restart_guard&&restarted_from_safe)?6:0;
        bool solved=false;int inner=0;double direction_norm=std::numeric_limits<double>::infinity();
        const bool exit_probe=options.exit_probe_frame==total_Frames+1&&options.exit_probe_outer==k;
        rec["full_step_exit_probe_active"]=exit_probe;
        int large_direction_streak=0;
        std::string outer_update_prefix;
        for(;inner<(exit_probe?8:1024);++inner)
        {
            gipc::set_solve_inner(inner);
            frame["newton"].push_back(gipc::Json::object());
            gipc::attach_solve_context(frame["newton"].back());
            int nc=s.contacts.size();
            if(nc)update_slack<<<(nc+255)/256,256>>>(s.contacts.data(),nc,_vertexes,s.mu,s.initial_mu,false);
            auto assembly_start=stamp();
            stage_event("assembly_begin",{{"outer",k},{"inner",inner}});
            const char* state_dir=std::getenv("GIPC_STATE_WINDOW_DIR");
            const bool state_window=state_dir && total_Frames+1>=21 && total_Frames+1<=25;
            std::string state_prefix;
            if(state_window)
            {
                state_prefix=std::string(state_dir)+"/f"+std::to_string(total_Frames+1)
                    +"_n"+std::to_string(frame["newton"].size());
                gipc::Json meta={{"frame",total_Frames+1},{"outer",k},{"inner",inner},
                    {"stage","before_derivative_assembly"},{"complete_restart_checkpoint",false}};
                gipc::attach_solve_context(meta);
                for(const auto& entry:state_snapshot())
                {
                    std::ofstream file(state_prefix+"_state_"+entry.first+".bin",std::ios::binary);
                    file.write(reinterpret_cast<const char*>(entry.second.data()),entry.second.size());
                    if(!file)throw std::runtime_error("Failed to write state window");
                    meta["buffers"][entry.first]=entry.second.size();
                }
                std::ofstream(state_prefix+"_state.json")<<meta.dump(2);
            }
            // The stiffness estimate assembled this exact state with empty C.
            // Publishing an empty C and copying the initial safe/trial/q do
            // not change the energy, its gradient, or its projected Hessian.
            const bool initial_reuse=reuse_initial && k==0 && inner==0 && s.host_contacts.empty()
                && h_cpNum_last[0]==0 && h_gpNum_last==0;
            if(!initial_reuse)computeGradientAndHessian(mesh);
            auto pcg_start=stamp();
            stage_event("pcg_begin",{{"outer",k},{"inner",inner}});
            int cg=calculateMovingDirection(mesh,h_cpNum[0],pcg_data.P_type);total_Cg_count+=cg;
            gipc::attach_solve_context(frame["newton"].back());
            if(stage_enabled())stage_event("pcg_end",{{"outer",k},{"inner",inner},{"pcg",frame["newton"].back()["pcg"]}});
            if(state_window)m_global_linear_system->snapshot_system(state_prefix);
            if(frame["newton"].back()["pcg"].value("iteration_limit",false))
                throw std::runtime_error("TOI PCG reached iteration limit; refusing an unconverged trial update");
            auto line_search_start=stamp();
            stage_event("line_search_begin",{{"outer",k},{"inner",inner}});
            if(movement_exit && s.host_contacts.empty())
            {
                s.energy.resize(vertexNum);s.scalar.resize(1);
                direction_squared_norm<<<(vertexNum+255)/256,256>>>(_moveDir,s.energy.data(),vertexNum);
                cudatool::DeviceReduce().Max(s.energy.data(),s.scalar.data(),vertexNum);
                double squared_norm;CUDA_SAFE_CALL(cudaMemcpy(&squared_norm,s.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));
                direction_norm=std::sqrt(squared_norm);
            }
            double trial_velocity=0;
            if(robust_velocity_tol>0)
            {
                s.energy.resize(vertexNum);s.scalar.resize(1);
                direction_axis_max<<<(vertexNum+255)/256,256>>>(_moveDir,s.energy.data(),vertexNum);
                cudatool::DeviceReduce().Max(s.energy.data(),s.scalar.data(),vertexNum);
                CUDA_SAFE_CALL(cudaMemcpy(&trial_velocity,s.scalar.data(),sizeof(double),cudaMemcpyDeviceToHost));
                trial_velocity/=IPC_dt;
                if(!std::isfinite(trial_velocity))throw std::runtime_error("Non-finite TOI trial velocity");
            }
            double before=computeEnergy(mesh);
            if(!std::isfinite(before))throw std::runtime_error("Non-finite TOI trial energy before line search");
            std::string event_prefix;gipc::Json direction_event;
            if(event_dir && total_Frames+1>=event_from)
            {
                large_direction_streak=trial_velocity>event_velocity?large_direction_streak+1:0;
                if(event_count<4 && (large_direction_streak==event_first || large_direction_streak==event_second))
                {
                    event_prefix=std::string(event_dir)+"/e"+std::to_string(++event_count);
                    if(outer_update_prefix.empty())outer_update_prefix=event_prefix;
                    if(std::filesystem::exists(event_prefix+"_event.json"))throw std::runtime_error("Refusing event overwrite");
                    const auto protected_state=state_snapshot();
                    m_global_linear_system->snapshot_system(event_prefix);
                    m_global_linear_system->snapshot_solution(event_prefix);
                    auto save=[&](const std::string& name,const void* ptr,size_t bytes)
                    {
                        std::vector<unsigned char> data(bytes);
                        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(data.data(),ptr,bytes,cudaMemcpyDeviceToHost));
                        std::ofstream f(event_prefix+"_"+name+".bin",std::ios::binary);
                        f.write(reinterpret_cast<const char*>(data.data()),bytes);f.close();
                        if(!f)throw std::runtime_error("Failed to persist event state");
                    };
                    save("vertices",_vertexes,vertexNum*sizeof(double3));save("direction",_moveDir,vertexNum*sizeof(double3));
                    save("safe",s.safe.data(),vertexNum*sizeof(double3));save("contacts",s.contacts.data(),s.contacts.size()*sizeof(ToiContact));
                    save("abd_q",q.data(),q.size()*sizeof(*q.data()));
                    save("abd_dq",m_abd_sim_data->device.body_id_to_dq.data(),q.size()*sizeof(*q.data()));
                    const auto event_contacts=download(s.contacts.data(),s.contacts.size());
                    std::array<size_t,3> contact_kinds{};double max_lambda=0,max_gamma=0,max_penalty=0;
                    for(const auto& c:event_contacts)
                    {
                        if(c.kind>=0 && c.kind<3)++contact_kinds[c.kind];
                        max_lambda=std::max(max_lambda,std::abs(c.lambda));max_gamma=std::max(max_gamma,std::abs(c.gamma));
                        max_penalty=std::max(max_penalty,std::abs(c.penalty_mu));
                    }
                    direction_event={{"schema","gipc-v49-full-energy-event-1"},{"frame",total_Frames+1},{"outer",k},{"inner",inner},
                        {"direction_index",frame["newton"].size()},{"streak",large_direction_streak},{"trigger_velocity",event_velocity},
                        {"trial_velocity",trial_velocity},{"dt",IPC_dt},{"mu",s.mu},{"delta",s.delta},{"beta",beta},
                        {"contacts",s.contacts.size()},{"contact_bytes",sizeof(ToiContact)},{"vertices",vertexNum},
                        {"energy_before",before},{"energy_components_before",event_components()},
                        {"pcg",frame["newton"].back()["pcg"]},{"complete_restart_checkpoint",false},{"line_search_completed",false}};
                    gipc::attach_solve_context(direction_event);
                    direction_event["contact_summary"]={{"pt_ee_ground",contact_kinds},{"max_abs_lambda",max_lambda},
                        {"max_abs_gamma",max_gamma},{"max_abs_penalty_mu",max_penalty}};
                    direction_event["contact_update_prefix"]=outer_update_prefix;
                    direction_event["capture_state_unchanged"]=protected_state==state_snapshot();
                    if(!direction_event["capture_state_unchanged"].get<bool>())throw std::runtime_error("Direction event changed protected state");
                    std::ofstream f(event_prefix+"_event.json");f<<direction_event.dump(2);f.close();
                    if(!f)throw std::runtime_error("Failed to commit direction event");
                }
            }
            copy_d(mesh.temp_double3Mem.data(),_vertexes,vertexNum);
            m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);
            auto restore_trial_snapshot=[&]()
            {
                // alpha=0 remaps ABD vertices through J*q. Restore the actual
                // saved vertex and generalized-coordinate buffers instead.
                copy_d(_vertexes,mesh.temp_double3Mem.data(),vertexNum);
                copy_d(q.data(),m_abd_sim_data->device.body_id_to_q_temp.data(),q.size());
            };
            const bool state_sample=state_audit_from>0 && total_Frames+1>=state_audit_from && inner==0
                && (k<=1 || (!first_contact_audited && s.contacts.size()>0));
            std::map<std::string,std::vector<unsigned char>> protected_before;
            gipc::Json state_audit;
            if(state_sample)
            {
                if(s.contacts.size()>0)first_contact_audited=true;
                protected_before=state_snapshot();
                state_audit={{"frame",total_Frames+1},{"outer",k},{"active_contacts",s.contacts.size()},
                    {"assembly",m_global_linear_system->audit_reassembly([&](){computeGradientAndHessian(mesh);})}};
                const auto saved_vertices=download(_vertexes,vertexNum);
                step_forward(mesh,0,false);
                const auto remapped=download(_vertexes,vertexNum);
                double maximum=0;size_t unequal=0;
                for(size_t i=0;i<saved_vertices.size();++i)
                {
                    const auto a=saved_vertices[i],b=remapped[i];
                    maximum=std::max({maximum,std::abs(a.x-b.x),std::abs(a.y-b.y),std::abs(a.z-b.z)});
                    unequal+=std::memcmp(&saved_vertices[i],&remapped[i],sizeof(double3))!=0;
                }
                state_audit["alpha_zero_vertex_max_abs_difference"]=maximum;
                state_audit["alpha_zero_bitwise_different_vertices"]=unequal;
                restore_trial_snapshot();
                for(double alpha:{1e-6,-1e-6,1.})
                {
                    step_forward(mesh,alpha,false);
                    if(!std::isfinite(computeEnergy(mesh)))throw std::runtime_error("Nonfinite snapshot-audit probe energy");
                }
                restore_trial_snapshot();
            }
            if(diagnostic.is_open() && total_Frames>=22)
            {
                constexpr double epsilon=1e-6;
                step_forward(mesh,epsilon,false);double plus=computeEnergy(mesh);
                step_forward(mesh,-epsilon,false);double minus=computeEnergy(mesh);
                restore_trial_snapshot();
                auto& direction_audit=frame["newton"].back();
                double predicted=direction_audit["pcg"]["gradient_dot_newton_step"].get<double>();
                double fd=(plus-minus)/(2*epsilon);
                direction_audit["total_energy_directional_fd"]=fd;
                direction_audit["total_energy_directional_scaled_error"]=std::abs(fd-predicted)/std::max({1.,std::abs(fd),std::abs(predicted)});
            }
            if(const char* from=std::getenv("GIPC_TOI_CURVATURE_FROM_FRAME"); diagnostic.is_open() && from
               && total_Frames+1>=std::stoi(from) && (inner==0 || inner%64==0))
            {
                const double saved_self=frictionRate,saved_ground=gd_frictionRate;
                auto energy_at=[&](double alpha){step_forward(mesh,alpha,false);return computeEnergy(mesh);};
                frictionRate=0;gd_frictionRate=0;
                restore_trial_snapshot();
                const double nonfriction_before=computeEnergy(mesh);
                auto audit=gipc::Json::array();
                for(double h:{1e-3,1e-2})
                {
                    frictionRate=saved_self;gd_frictionRate=saved_ground;
                    const double plus=energy_at(h),minus=energy_at(-h);
                    frictionRate=0;gd_frictionRate=0;
                    const double nonfriction_plus=energy_at(h),nonfriction_minus=energy_at(-h);
                    const double total=(plus+minus-2*before)/(h*h);
                    const double nonfriction=(nonfriction_plus+nonfriction_minus-2*nonfriction_before)/(h*h);
                    audit.push_back({{"epsilon",h},{"total",total},{"nonfriction",nonfriction},
                                     {"friction_difference",total-nonfriction}});
                }
                auto finite_steps=gipc::Json::array();
                for(double alpha:{.25,.5,1.})
                {
                    frictionRate=saved_self;gd_frictionRate=saved_ground;
                    const double total_delta=energy_at(alpha)-before;
                    frictionRate=0;gd_frictionRate=0;
                    const double nonfriction_delta=energy_at(alpha)-nonfriction_before;
                    finite_steps.push_back({{"alpha",alpha},{"total_delta",total_delta},
                                            {"nonfriction_delta",nonfriction_delta},
                                            {"friction_delta",total_delta-nonfriction_delta}});
                }
                frictionRate=saved_self;gd_frictionRate=saved_ground;
                auto components_at=[&](double alpha)
                {
                    if(alpha==0)restore_trial_snapshot();else step_forward(mesh,alpha,false);
                    std::array<double,8> c{};
                    c[0]=Energy_Add_Reduction_Algorithm(0,mesh)
                        +m_abd_system->cal_abd_kinetic_energy(*m_abd_sim_data);
                    c[1]=m_abd_system->cal_abd_shape_energy(*m_abd_sim_data);
                    c[2]=IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(1,mesh);
                    c[3]=IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(8,mesh);
                    c[4]=IPC_dt*IPC_dt*Energy_Add_Reduction_Algorithm(10,mesh);
                    c[5]=Energy_Add_Reduction_Algorithm(9,mesh);
                    c[6]=toiContactEnergy();
#ifdef USE_FRICTION
                    c[7]=frictionRate*Energy_Add_Reduction_Algorithm(5,mesh)
                        +gd_frictionRate*Energy_Add_Reduction_Algorithm(6,mesh);
#endif
                    return c;
                };
                const auto component_before=components_at(0),component_full=components_at(1);
                gipc::Json component_delta={{"kinetic",component_full[0]-component_before[0]},
                    {"abd_shape",component_full[1]-component_before[1]},
                    {"fem",component_full[2]-component_before[2]},
                    {"cloth",component_full[3]-component_before[3]},
                    {"bending",component_full[4]-component_before[4]},
                    {"constraint",component_full[5]-component_before[5]},
                    {"al_contact",component_full[6]-component_before[6]},
                    {"friction",component_full[7]-component_before[7]}};
                double summed_delta=0;
                for(size_t i=0;i<component_full.size();++i)
                    summed_delta+=component_full[i]-component_before[i];
                const double full_delta=finite_steps.back()["total_delta"].get<double>();
                if(!std::isfinite(summed_delta)
                   || std::abs(summed_delta-full_delta)>1e-8*std::max({1.,std::abs(summed_delta),std::abs(full_delta)}))
                    throw std::runtime_error("TOI diagnostic energy component sum mismatch");
                if(triangle_audit)
                {
                    if(audit_triangles.empty())
                    {
                        audit_triangles=download(mesh.triangles.data(),triangleNum);
                        audit_inverse_rest=download(mesh.triDmInverses.data(),triangleNum);
                        audit_area=download(mesh.area.data(),triangleNum);
                    }
                    restore_trial_snapshot();
                    const auto x_before=download(mesh.vertexes.data(),vertexNum);
                    step_forward(mesh,1,false);
                    const auto x_full=download(mesh.vertexes.data(),vertexNum);
                    const double dt2=IPC_dt*IPC_dt;
                    const auto triangles_before=triangle_energy_snapshot(x_before,audit_triangles,
                        audit_inverse_rest,audit_area,dt2,stretchStiff,shearStiff,strainRate);
                    const auto triangles_full=triangle_energy_snapshot(x_full,audit_triangles,
                        audit_inverse_rest,audit_area,dt2,stretchStiff,shearStiff,strainRate);
                    const double before_error=std::abs(triangles_before.total-component_before[3]);
                    const double full_error=std::abs(triangles_full.total-component_full[3]);
                    if(before_error>1e-8*std::max({1.,std::abs(triangles_before.total),std::abs(component_before[3])})
                       || full_error>1e-8*std::max({1.,std::abs(triangles_full.total),std::abs(component_full[3])}))
                        throw std::runtime_error("TOI per-triangle cloth energy does not match GPU sum");
                    frame["newton"].back()["triangle_energy_audit"]=
                        triangle_energy_delta_summary(triangles_before,triangles_full);
                }
                restore_trial_snapshot();
                frame["newton"].back()["direction_curvature_audit"]=std::move(audit);
                frame["newton"].back()["finite_step_energy_audit"]=std::move(finite_steps);
                frame["newton"].back()["full_step_component_delta"]=std::move(component_delta);
            }
            if(state_sample)
            {
                const auto after_snapshot=state_snapshot();
                auto changed=gipc::Json::array();
                for(const auto& entry:protected_before)
                    if(after_snapshot.at(entry.first)!=entry.second)changed.push_back(entry.first);
                state_audit["changed_buffers"]=changed;
                state_audit["protected_state_restored_bitwise"]=changed.empty();
                frame["newton"].back()["state_snapshot_audit"]=std::move(state_audit);
                if(!changed.empty())throw std::runtime_error("TOI diagnostic altered protected physics state: "+changed.dump());
            }
            double r=1;
            // AL solves the unrestricted trial state. Stable NH / ARAP are
            // defined for inverted trial tetrahedra; imposing injectivity
            // here can lock the subproblem at det=0 before UpdateC runs.
    // Optional positive-volume policy bounds only accepted safe paths below.
            if(clamp_trial && tetrahedraNum>0)r=std::min(r,legacy_volume?InjectiveStepSize(.8,1e-6,mesh.tetrahedras)
                :normalized_volume_step(*this,mesh,_vertexes,nullptr,.8));
            const double initial_r=r;double after=before;int backtracks=0;
            bool accepted=false;
            for(int backtrack=0;backtrack<64;++backtrack)
            {
                backtracks=backtrack;step_forward(mesh,r,false);after=computeEnergy(mesh);
                if(std::isfinite(after) && after<=before+1e-12*std::max(1.,std::abs(before))) {accepted=true;break;}
                r*=.5;
            }
            if(!accepted)throw std::runtime_error("TOI inner line search exhausted 64 backtracks");
            auto& nr=frame["newton"].back();nr["toi_outer"]=k;nr["toi_inner"]=inner;nr["line_search_r"]=r;
            nr["line_search_initial_r"]=initial_r;nr["line_search_backtracks"]=backtracks;
            nr["trial_energy_before"]=before;nr["trial_energy_after"]=after;
            if(!event_prefix.empty())
            {
                const auto protected_state=state_snapshot();
                direction_event["energy_components_after"]=event_components();
                direction_event["post_probe_state_unchanged"]=protected_state==state_snapshot();
                if(!direction_event["post_probe_state_unchanged"].get<bool>())throw std::runtime_error("Direction energy probe changed protected state");
                direction_event["energy_after"]=after;direction_event["line_search_r"]=r;
                const auto accepted_vertices=download(_vertexes,vertexNum);
                std::ofstream vertices_file(event_prefix+"_after_vertices.bin",std::ios::binary);
                vertices_file.write(reinterpret_cast<const char*>(accepted_vertices.data()),accepted_vertices.size()*sizeof(double3));
                vertices_file.close();if(!vertices_file)throw std::runtime_error("Failed to export event trial endpoint");
                // Probe the complete frozen objective, then restore the accepted
                // endpoint byte-for-byte. This never changes the selected r.
                const auto accepted_q=download(q.data(),q.size());
                auto probes=gipc::Json::array();
                auto probe=[&](double alpha)
                {
                    step_forward(mesh,alpha,false);
                    const double energy=computeEnergy(mesh);
                    if(!std::isfinite(energy))throw std::runtime_error("Nonfinite full-energy event probe");
                    auto value=gipc::Json{{"alpha",alpha},{"energy",energy},{"components",event_components()}};
                    return value;
                };
                const double probe_h=std::min(r/16.,1e-6/std::max(1.,trial_velocity*IPC_dt));
                for(double center:{r,r*.5})
                {
                    auto sample=probe(center);auto derivatives=gipc::Json::array();
                    for(double h:{probe_h,probe_h*.25})
                    {
                        auto plus=probe(center+h),minus=probe(center-h);
                        const double derivative=(plus["energy"].get<double>()-minus["energy"].get<double>())/(2*h);
                        derivatives.push_back({{"h",h},{"plus",plus},{"minus",minus},{"derivative",derivative}});
                    }
                    sample["derivatives"]=std::move(derivatives);probes.push_back(std::move(sample));
                }
                CUDA_SAFE_CALL(cudaMemcpy(_vertexes,accepted_vertices.data(),vertexNum*sizeof(double3),cudaMemcpyHostToDevice));
                if(!accepted_q.empty())CUDA_SAFE_CALL(cudaMemcpy(q.data(),accepted_q.data(),accepted_q.size()*sizeof(*q.data()),cudaMemcpyHostToDevice));
                direction_event["full_energy_probes"]=std::move(probes);
                direction_event["full_probe_state_restored"]=protected_state==state_snapshot();
                if(!direction_event["full_probe_state_restored"].get<bool>())throw std::runtime_error("Full energy probes failed state restoration");
                direction_event["line_search_completed"]=true;
                std::ofstream f(event_prefix+"_line_search.json");f<<direction_event.dump(2);f.close();
                if(!f)throw std::runtime_error("Failed to complete direction event");
            }
            if(event_dir && event_count)event_progress({{"stage","inner_end"},{"outer",k},{"inner",inner},{"streak",large_direction_streak},
                {"velocity",trial_velocity},{"r",r},{"energy_before",before},{"energy_after",after}});
            nr["initial_assembly_reused"]=initial_reuse;
            if(robust_velocity_tol>0)nr["trial_newton_axis_velocity_m_s"]=trial_velocity;
            if(profile){auto end=stamp();nr["profile_ms"]={{"assembly",elapsed(assembly_start,pcg_start)},{"pcg",elapsed(pcg_start,line_search_start)},{"line_search",elapsed(line_search_start,end)}};}
            // Optional Robust-style small-direction convergence. The accepted
            // trial has already passed the energy check; the full swept CCD
            // and safe-state validation still run after this inner exit.
            const bool velocity_converged=robust_velocity_stop && trial_velocity<=robust_velocity_tol;
            const bool full_step=r==1 && (robust_velocity_tol==0 || trial_velocity<=100*robust_velocity_tol);
            if(stage_enabled())stage_event("inner_end",{{"outer",k},{"inner",inner},{"newton",nr}});
            const bool allow_velocity=inner_exit!="full_step_only";
            const bool allow_full=inner_exit!="velocity_only"&&!exit_probe;
            nr["full_step_exit_probe_active"]=exit_probe;
            // Replacing the warm trial invalidates its accumulated iteration
            // history for the heuristic full-step exit. Keep the native six
            // iterations, but count them within this restarted subproblem.
            // A genuinely small raw direction can still converge earlier.
            const bool restart_ready=!restart_guard||!restarted_from_safe||inner+1>=6;
            const bool full_history_ready=!s.robust_port||(frame["newton"].size()>=6&&restart_ready);
            nr["restart_full_step_guard_active"]=restart_guard&&restarted_from_safe;
            nr["restart_full_step_ready"]=restart_ready;
            nr["restart_full_step_blocked"]=s.robust_port&&allow_full&&full_step
                && frame["newton"].size()>=6&&!restart_ready;
            if((allow_velocity&&velocity_converged) || (allow_full&&full_step&&full_history_ready))
            {nr["inner_exit_reason"]=(allow_velocity&&velocity_converged)?"velocity_converged":"full_step";solved=true;break;}
        }
        if(!solved)throw std::runtime_error(exit_probe?"Selected exit probe reached its eight-iteration budget":"TOI subproblem reached inner iteration limit");
        rec["inner_iterations"]=inner+1;
        int nc=s.contacts.size();
        if(outer_probe)
        {
            probe_trial=download(_vertexes,vertexNum);probe_before_dual=download(s.contacts.data(),nc);
            const auto protected_state=state_snapshot();
            const double complete=computeEnergy(mesh);const auto parts=event_components();
            if(protected_state!=state_snapshot())throw std::runtime_error("Outer energy probe altered protected physics state");
            double sum=0;for(const auto& part:parts.items())sum+=part.value().get<double>();
            gipc::ToiObserver::outer_event({{"stage","trial_before_dual"},{"outer",k},
                {"mu",s.mu},{"delta",s.delta},{"inner_iterations",inner+1},
                {"complete_energy",complete},{"energy_parts",parts},{"parts_sum_minus_complete",sum-complete},
                {"physics_state_unchanged",true},{"last_newton",frame["newton"].back()},
                {"trial_shape",outer_shape_probe(probe_trial,probe_safe_begin,probe_rest,probe_tets,abd_fem_count_info.abd_point_num)},
                {"contact",outer_contact_probe(probe_before_dual,probe_trial,s.mu,s.initial_mu,s.delta)}});
        }
        std::vector<ToiContact> trial_contacts;
        if(blocker_audit)trial_contacts=download(s.contacts.data(),nc);
        if(!outer_update_prefix.empty())
        {
            event_save(outer_update_prefix,"update_before",s.contacts.data(),nc*sizeof(ToiContact));
            event_save(outer_update_prefix,"update_vertices",_vertexes,vertexNum*sizeof(double3));
        }
        if(nc)update_slack<<<(nc+255)/256,256>>>(s.contacts.data(),nc,_vertexes,s.mu,s.initial_mu,true,s.independent_friction_history,s.robust_port);
        if(outer_probe)
        {
            probe_after_dual=download(s.contacts.data(),nc);
            gipc::ToiObserver::outer_event({{"stage","dual_update"},{"outer",k},
                {"contact",outer_contact_probe(probe_after_dual,probe_trial,s.mu,s.initial_mu,s.delta,&probe_before_dual)}});
        }
        if(!outer_update_prefix.empty())
        {
            event_save(outer_update_prefix,"update_after",s.contacts.data(),nc*sizeof(ToiContact));
            std::ofstream file(outer_update_prefix+"_update.json");
            gipc::Json update={{"frame",total_Frames+1},{"outer",k},{"contacts",nc},{"contact_bytes",sizeof(ToiContact)},
                {"mu",s.mu},{"initial_mu",s.initial_mu},{"robust_port",s.robust_port},
                {"independent_friction",s.independent_friction_history}};
            gipc::attach_solve_context(update);file<<update.dump(2);
            file.close();if(!file)throw std::runtime_error("Failed to commit contact update event");
            pending_update_prefix=outer_update_prefix;
        }
        s.host_contacts=download(s.contacts.data(),s.contacts.size());
        if(s.independent_friction_history)
        {
            double proxy_difference=0;
            for(const auto& c:s.host_contacts)proxy_difference=std::max({proxy_difference,
                std::abs(c.friction_lambda-c.lambda),std::abs(c.friction_gamma-c.gamma)});
            rec["independent_proxy_vs_al_max_abs_difference"]=proxy_difference;
        }
        copy_d(s.trial.data(),_vertexes,vertexNum);copy_d(s.trial_q.data(),q.data(),q.size());
        if(diagnostic.is_open())
        {
            auto trial=download(s.trial.data(),vertexNum),safe=download(s.safe.data(),vertexNum);
            double max_motion=0,max_position=0;
            for(int i=0;i<vertexNum;++i)
            {double dx=trial[i].x-safe[i].x,dy=trial[i].y-safe[i].y,dz=trial[i].z-safe[i].z;
             max_motion=std::max(max_motion,std::sqrt(dx*dx+dy*dy+dz*dz));
             max_position=std::max(max_position,std::sqrt(trial[i].x*trial[i].x+trial[i].y*trial[i].y+trial[i].z*trial[i].z));}
            gipc::Json entry={{"frame",total_Frames},{"outer",k},{"inner_iterations",inner+1},{"mu",s.mu},{"delta",s.delta},
                {"active_before_update",s.contacts.size()},{"trial_max_motion_m",max_motion},{"trial_max_position_norm_m",max_position},
                {"last_inner",frame["newton"].back()}};
            entry["legacy_frame_index"]=total_Frames;gipc::attach_solve_context(entry);
            if(watch_contact)entry["watched_contact_before_update"]=watched_state();
            diagnostic<<entry.dump()<<std::endl;
        }
        copy_d(_vertexes,s.safe.data(),vertexNum);copy_d(q.data(),s.safe_q.data(),q.size());
        // Paper Alg1 updates C with the previous trial, then advances safe
        // using a full CCD query toward the newly solved trial.
        auto active_start=stamp();
        double current_trial_alpha=1;
        if(s.robust_port)
            current_trial_alpha=swept_query(*this,s.trial.data(),true,rec);
        else if(previous_full_step && s.host_contacts.empty())
        {
            // Previous trial is bitwise identical to safe after alpha=1.
            // The stationary path cannot introduce a new crossing. The new
            // trial still goes through the complete unfiltered CCD below.
            rec["active_update_broad_pairs"]=0;rec["active_candidates_new"]=0;
            rec["active_added"]=0;rec["active_removed"]=0;rec["stationary_active_update_skipped"]=true;
        }
        else swept_query(*this,s.previous_trial.data(),true,rec);
        auto ccd_start=stamp();
        double alpha=s.robust_port?current_trial_alpha:
            swept_query(*this,s.trial.data(),false,rec,blocker_audit?&trial_contacts:nullptr);
        if(s.robust_port&&blocker_audit)
            audit_safe_blocker(*this,s.trial.data(),alpha,trial_contacts,rec);
        if(s.robust_port)
        {rec["safe_ccd_broad_pairs"]=rec["active_update_broad_pairs"];rec["candidate_query_policy"]="current_trial_shared_full_ccd";}
        rec["full_ccd_alpha"]=alpha;
        if(clamp_safe && tetrahedraNum>0)
        {
            double injective_alpha=legacy_volume?InjectiveStepSize(.8,1e-6,mesh.tetrahedras)
                :normalized_volume_step(*this,mesh,_vertexes,nullptr,.8);
            rec["tet_injective_alpha"]=injective_alpha;alpha=std::min(alpha,injective_alpha);
        }
        auto safe_start=stamp();
        stage_event("safe_update_begin",{{"outer",k},{"alpha",alpha}});
        double safe_blend_ms=0,safe_bvh_ms=0,safe_candidates_ms=0,safe_intersection_ms=0;
        int safe_attempts=0;
        for(int backtrack=0;;++backtrack)
        {
            auto stage=safe_start;if(profile)stage=stamp();
            ++safe_attempts;
            blend_vertices<<<(vertexNum+255)/256,256>>>(_vertexes,s.safe.data(),s.trial.data(),alpha,vertexNum);
            if(q.size())blend_q<<<(q.size()*12+255)/256,256>>>(reinterpret_cast<double*>(q.data()),reinterpret_cast<const double*>(s.safe_q.data()),reinterpret_cast<const double*>(s.trial_q.data()),alpha,q.size()*12);
            // Check the actual rounded endpoints, including interior cubic
            // extrema, before publishing an accepted path.
            if(clamp_safe && !legacy_volume && normalized_volume_step(*this,mesh,s.safe.data(),_vertexes,0)<1)
            {
                if(backtrack>=63)throw std::runtime_error("TOI safe volume validation failed after 64 backtracks");
                alpha*=.5;continue;
            }
            if(profile){auto now=stamp();safe_blend_ms+=elapsed(stage,now);stage=now;}
            if(gipc_accel_feature("GIPC_CCD_BVH_REFIT"))buildBVH_FULLCCD(0);
            else buildBVH();
            if(profile){auto now=stamp();safe_bvh_ms+=elapsed(stage,now);stage=now;}
            buildCP();
            if(profile){auto now=stamp();safe_candidates_ms+=elapsed(stage,now);stage=now;}
            const bool intersected=isIntersected(mesh);
            if(profile){auto now=stamp();safe_intersection_ms+=elapsed(stage,now);}
            if(!intersected)break;
            if(backtrack>=63)throw std::runtime_error("TOI safe state CCD validation failed after 64 backtracks");
            alpha*=.5;
        }
        auto publish_start=safe_start;if(profile)publish_start=stamp();
        if(profile)rec["safe_detail_ms"]={{"blend_and_volume",safe_blend_ms},{"bvh",safe_bvh_ms},
            {"candidates",safe_candidates_ms},{"intersection",safe_intersection_ms},{"attempts",safe_attempts}};
        copy_d(s.safe.data(),_vertexes,vertexNum);copy_d(s.safe_q.data(),q.data(),q.size());
        previous_full_step=alpha==1;
        trace_safe(*this,k+1);
        if(s.robust_port||k+1>=6)beta*=1-alpha;
        rec["alpha"]=alpha;rec["beta"]=beta;rec["active_self"]=s.self_count;rec["active_ground"]=s.ground_count;
        if(outer_probe)
        {
            const auto endpoint=download(_vertexes,vertexNum);
            gipc::ToiObserver::outer_event({{"stage","safe_accepted"},{"outer",k},{"alpha",alpha},{"beta",beta},
                {"active_added",rec["active_added"]},{"active_removed",rec["active_removed"]},
                {"next_host_contacts",s.host_contacts.size()},{"contact_model_scope","solved model after dual update; next host set not yet relinearized"},
                {"safe_shape",outer_shape_probe(endpoint,probe_trial,probe_rest,probe_tets,abd_fem_count_info.abd_point_num)},
                {"solved_contact_model_at_safe",outer_contact_probe(probe_after_dual,endpoint,s.mu,s.initial_mu,s.delta)}});
        }
        if(event_dir && event_count)event_progress({{"stage","outer_end"},{"outer",k},{"alpha",alpha},{"beta",beta}});
        if(watch_contact)rec["watched_contact_after_update"]=watched_state();
        rec["physical_contact_force_proxy"]=std::any_of(s.host_contacts.begin(),s.host_contacts.end(),[](const auto& c){return c.lambda>0;});
        rec["safe_state_verified"]=true;
        if(friction_audit)
        {
            const auto actual=frame_friction_signature(*this);
            rec["friction_snapshot_signature"]=actual;
            rec["friction_snapshot_unchanged"]=actual==friction_signature;
            if(actual!=friction_signature)throw std::runtime_error("Frame friction buffers changed during solving");
        }
        rec["safe_volume_path_verified"]=clamp_safe&&!legacy_volume;
        gipc::attach_solve_context(rec);
        stage_event("outer_end",{{"outer",k},{"alpha",alpha},{"beta",beta}});
        rec["last_newton_direction_norm"]=std::isfinite(direction_norm)?gipc::Json(direction_norm):gipc::Json(nullptr);
        if(diagnostic.is_open())
        {
            gipc::Json entry={{"legacy_frame_index",total_Frames},{"accepted",rec}};
            gipc::attach_solve_context(entry);diagnostic<<entry.dump()<<std::endl;
        }
        if(profile){auto end=stamp();rec["safe_detail_ms"]["publication_and_bookkeeping"]=elapsed(publish_start,end);rec["profile_ms"]={{"outer_total",elapsed(begin_outer,end)},{"active_update",elapsed(active_start,ccd_start)},{"safe_ccd",elapsed(ccd_start,safe_start)},{"safe_update",elapsed(safe_start,end)}};}
        // Preserve the new base's movement exit only after an unfiltered full
        // CCD accepts the complete trial, with no AL constraints or new pairs.
        // Contact iterations retain Algorithms 1--3's cumulative TOI exit.
        if(movement_exit && k>0 && alpha==1 && s.host_contacts.empty()
           && rec["active_added"].get<int>()==0
           && direction_norm<Newton_solver_threshold*std::sqrt(bboxDiagSize2)*IPC_dt)
        {frame["toi_exit"]="movement_contact_free";return k+1;}
        if(beta<=options.remaining_fraction_tol){frame["toi_exit"]="cumulative_toi";return k+1;}
        stalls=alpha<1e-4?stalls+1:0;
        if(stalls>=stall_window)
        {
            s.mu*=2;
            if(!hold_delta)s.delta*=.5;
            rec["penalty_escalation"]={{"next_mu",s.mu},{"next_delta",s.delta}};
            stalls=0;
        }
    }
    frame["toi_exit"]="iteration_limit";
    throw std::runtime_error("TOI reached outer iteration limit");
}

int validate_toi_components(const char* report_path)
{
    gipc::Json report;report["tests"]=gipc::Json::array();bool passed=true;
    cudatool::DeviceBuffer<double3> x,gradient,normal;
    cudatool::DeviceBuffer<double> offset,energy;
    cudatool::DeviceBuffer<int> invalid,rows,cols;
    cudatool::DeviceBuffer<Eigen::Matrix3d> hessian;
    cudatool::DeviceBuffer<ToiContact> contacts;
    upload(normal,std::vector<double3>{make_double3(0,1,0)});
    upload(offset,std::vector<double>{-1});
    invalid.resize(1);gradient.resize(4);hessian.resize(16);rows.resize(16);cols.resize(16);energy.resize(1);
    for(int test=0;test<6;++test)
    {
        ToiContact c;c.kind=(test==2||test==4)?1:(test==3?2:0);
        for(int j=0;j<4;++j)c.ids[j]=j;
        std::vector<double3> positions;
        double expected_distance;
        if(test==0||test==5){positions={make_double3(.2,.3,.7),make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0)};expected_distance=.7;}
        if(test==1){positions={make_double3(.4,.6,0),make_double3(0,0,0),make_double3(1,0,0),make_double3(2,0,0)};expected_distance=.6;}
        if(test==2){positions={make_double3(-1,0,.6),make_double3(1,0,.6),make_double3(0,-1,0),make_double3(0,1,0)};expected_distance=.6;}
        if(test==3){positions={make_double3(.1,.6,0),make_double3(0,0,0),make_double3(0,0,0),make_double3(0,0,0)};expected_distance=1.6;}
        if(test==4){positions={make_double3(.4,.6,0),make_double3(.4,.6,0),make_double3(0,0,0),make_double3(1,0,0)};expected_distance=.6;}
        upload(x,positions);upload(contacts,std::vector<ToiContact>{c});
        const int sentinel=INT_MAX;
        CUDA_SAFE_CALL(cudaMemcpy(invalid.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<1,32>>>(contacts.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        auto linear=download(contacts.data(),1)[0];
        double distance_error=std::abs(linear.offset+.2-expected_distance);
        linear.lambda=test==2?3.3:1.3;linear.gamma=.8;linear.slack=.05;
        if(test==5)linear.penalty_mu=2.5;
        upload(contacts,std::vector<ToiContact>{linear});
        constexpr double mu=4.2,epsilon=1e-6;
        const double current_mu=test==5?2*mu:mu;
        const double effective_mu=test==5?5.:mu;
        CUDA_SAFE_CALL(cudaMemset(gradient.data(),0,4*sizeof(double3)));
        al_hessian<<<1,32>>>(contacts.data(),0,1,x.data(),gradient.data(),hessian.data(),rows.data(),cols.data(),0,current_mu,mu);
        auto g=download(gradient.data(),4);auto h=download(hessian.data(),test==3?1:16);
        auto evaluate=[&](const std::vector<double3>& state){upload(x,state);al_energy<<<1,32>>>(contacts.data(),1,x.data(),current_mu,mu,energy.data());return download(energy.data(),1)[0];};
        double gradient_error=0,hessian_error=0;
        int count=test==3?1:4;
        for(int a=0;a<count;++a)for(int d=0;d<3;++d)
        {
            auto plus=positions,minus=positions;
            reinterpret_cast<double*>(&plus[a])[d]+=epsilon;reinterpret_cast<double*>(&minus[a])[d]-=epsilon;
            double fd=(evaluate(plus)-evaluate(minus))/(2*epsilon);
            gradient_error=std::max(gradient_error,std::abs(fd-reinterpret_cast<double*>(&g[a])[d]));
            for(int b=0;b<count;++b)for(int e=0;e<3;++e)
            {
                double expected=effective_mu*linear.gamma*reinterpret_cast<double*>(&linear.grad[a])[d]*reinterpret_cast<double*>(&linear.grad[b])[e];
                hessian_error=std::max(hessian_error,std::abs(h[a*count+b](d,e)-expected));
            }
        }
        upload(x,positions);
        update_slack<<<1,32>>>(contacts.data(),1,x.data(),current_mu,mu,true);
        auto updated=download(contacts.data(),1)[0];
        double expected_slack=std::max(0.,linear.offset-linear.lambda/effective_mu);
        double expected_lambda=expected_slack==0?linear.lambda-effective_mu*linear.offset:0;
        double expected_gamma=expected_slack==0?1:linear.gamma*.9;
        double update_error=std::max({std::abs(updated.slack-expected_slack),std::abs(updated.lambda-expected_lambda),std::abs(updated.gamma-expected_gamma)});
        bool ok=distance_error<1e-12 && gradient_error<1e-8 && hessian_error<1e-12 && update_error<1e-12 && download(invalid.data(),1)[0]==sentinel;
        passed=passed&&ok;
        report["tests"].push_back({{"case",test==0?"PT interior":test==1?"PT degenerate":test==2?"EE interior":test==3?"ground":test==4?"EE degenerate":"PT per-contact mu adaptation"},{"distance_error",distance_error},{"energy_gradient_fd_max_error",gradient_error},{"full_hessian_max_error",hessian_error},{"slack_multiplier_gamma_error",update_error},{"passed",ok}});
    }
    // Check elimination against explicit slack minimization, including a
    // trial crossing the active-set boundary and a per-contact penalty.
    report["reduced_slack_tests"]=gipc::Json::array();
    for(int test=0;test<6;++test)
    {
        ToiContact c;c.kind=2;c.ids[0]=0;c.grad[0]=make_double3(0,1,0);
        c.anchor[0]=make_double3(0,0,0);c.offset=0;c.lambda=.7;c.gamma=.8;
        c.slack=17.; // Deliberately stale: the reduced objective must ignore it.
        const double mu=test>=3?5.:4.2;
        if(test>=3)c.penalty_mu=2.5;
        const double current_mu=test>=3?8.4:4.2;
        const double y=c.lambda/mu+(test%3-1)*.2;
        std::vector<double3> pos(4,make_double3(0,0,0));pos[0].y=y;
        upload(contacts,std::vector<ToiContact>{c});upload(x,pos);
        CUDA_SAFE_CALL(cudaMemset(gradient.data(),0,4*sizeof(double3)));
        al_hessian<<<1,32>>>(contacts.data(),0,1,x.data(),gradient.data(),hessian.data(),rows.data(),cols.data(),0,current_mu,4.2,true);
        const auto g=download(gradient.data(),4);const auto h=download(hessian.data(),1);
        auto eval=[&](double yy){pos[0].y=yy;upload(x,pos);
            al_energy<<<1,32>>>(contacts.data(),1,x.data(),current_mu,4.2,energy.data(),true);
            return download(energy.data(),1)[0];};
        const double slack=std::max(0.,y-c.lambda/mu),v=y-slack;
        const double expected=c.gamma*(.5*mu*v*v-c.lambda*v);
        const double energy_error=std::abs(eval(y)-expected),eps=1e-6;
        const double gradient_error=std::abs((eval(y+eps)-eval(y-eps))/(2*eps)-g[0].y);
        const double curvature_error=test%3==1?0.:std::abs((eval(y+eps)-2*eval(y)+eval(y-eps))/(eps*eps)-h[0](1,1));
        const bool ok=energy_error<1e-12 && gradient_error<2e-6 && curvature_error<1e-4;
        passed=passed&&ok;report["reduced_slack_tests"].push_back({{"case",test},{"energy_error",energy_error},
            {"gradient_error",gradient_error},{"curvature_error",curvature_error},{"passed",ok}});
    }
    // Fixed PT geometry: independently warm-start normal and friction
    // histories, then test their actual GPU force and multiplier update.
    report["warm_history_tests"]=gipc::Json::array();
    struct HistoryCase {const char* name;bool lambda,gamma,independent,friction;};
    const HistoryCase history_cases[]={{"C only",false,false,true,false},
        {"C + lambda",true,false,true,false},{"C + gamma",false,true,true,false},
        {"C + lambda + gamma",true,true,true,false},{"C + friction",false,false,true,true},
        {"all independent",true,true,true,true},{"legacy coupled",true,true,false,false}};
    cudatool::DeviceBuffer<double> force;force.resize(1);
    cudatool::DeviceBuffer<double2> coordinates;coordinates.resize(1);
    cudatool::DeviceBuffer<__GEIGEN__::Matrix3x2d> basis;basis.resize(1);
    const std::vector<double3> rest={make_double3(.2,.3,.7),make_double3(0,0,0),
        make_double3(1,0,0),make_double3(0,1,0)};
    std::vector<double3> trial=rest;trial[0].z=.1;
    for(const auto& policy:history_cases)
    {
        ToiContact old;for(int j=0;j<4;++j)old.ids[j]=j;
        old.lambda=1.3;old.gamma=.8;old.friction_lambda=.9;old.friction_gamma=.6;
        std::vector<ToiContact> host{old};
        prepare_warm_contacts(host,policy.lambda,policy.gamma,policy.independent,policy.friction);
        upload(contacts,host);upload(x,rest);const int sentinel=INT_MAX;
        CUDA_SAFE_CALL(cudaMemcpy(invalid.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<1,32>>>(contacts.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        al_energy<<<1,32>>>(contacts.data(),1,x.data(),4.2,4.2,energy.data());
        const double actual_energy=download(energy.data(),1)[0];
        const double expected_energy=(policy.gamma?.8:1)*(.5*4.2*.5*.5-(policy.lambda?1.3:0)*.5);
        friction_basis<<<1,32>>>(contacts.data(),1,force.data(),coordinates.data(),basis.data(),policy.independent);
        const double actual_force=download(force.data(),1)[0];
        const double expected_force=policy.independent?(policy.friction?.54:0):1.04;
        upload(x,trial);
        update_slack<<<1,32>>>(contacts.data(),1,x.data(),4.2,4.2,true,policy.independent);
        const auto next=download(contacts.data(),1)[0];
        const double lambda_error=std::abs(next.lambda-(policy.lambda?1.72:.42));
        double proxy_error=0;
        if(policy.independent)proxy_error=std::max(std::abs(next.friction_lambda-(policy.friction?1.32:.42)),std::abs(next.friction_gamma-1));
        const bool ok=std::abs(actual_energy-expected_energy)<1e-12
            &&std::abs(actual_force-expected_force)<1e-12&&lambda_error<1e-12&&proxy_error<1e-12
            &&next.gamma==1&&download(invalid.data(),1)[0]==sentinel;
        passed=passed&&ok;report["warm_history_tests"].push_back({{"case",policy.name},
            {"normal_energy",actual_energy},{"friction_force_proxy",actual_force},
            {"normal_lambda_update_error",lambda_error},{"friction_proxy_update_error",proxy_error},{"passed",ok}});
    }
    report["port_tests"]=gipc::Json::array();
    {
        ToiContact c;c.kind=0;for(int i=0;i<4;++i)c.ids[i]=i;
        std::vector<double3> p={make_double3(.2,.3,.7),make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0)};
        upload(x,p);upload(contacts,std::vector<ToiContact>{c});
        const int sentinel=INT_MAX;CUDA_SAFE_CALL(cudaMemcpy(invalid.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<1,32>>>(contacts.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        for(int i=0;i<26;++i)update_slack<<<1,32>>>(contacts.data(),1,x.data(),4.2,4.2,true,false,true);
        auto released=download(contacts.data(),1)[0];
        bool ok=released.release_age==26&&!toi_port::retain_contact(released.release_age)
            &&std::abs(released.gamma-std::pow(.9,26))<1e-14&&released.lambda==0;
        passed=passed&&ok;report["port_tests"].push_back({{"case","release_age_gpu"},{"age",released.release_age},{"passed",ok}});
        p[0].z=-.3;upload(x,p);
        update_slack<<<1,32>>>(contacts.data(),1,x.data(),4.2,4.2,true,false,true);
        auto active=download(contacts.data(),1)[0];
        ok=active.release_age==0&&active.gamma==1&&std::abs(active.lambda-2.1)<1e-12;
        passed=passed&&ok;report["port_tests"].push_back({{"case","reactivation_gpu"},{"lambda",active.lambda},{"passed",ok}});
    }
    for(int kind:{0,1})
    {
        ToiContact c;c.kind=kind;c.lambda=3;c.gamma=.4;for(int i=0;i<4;++i)c.ids[i]=i;
        std::vector<double3> p=kind==0?std::vector<double3>{make_double3(.2,.3,.7),make_double3(0,0,0),make_double3(1,0,0),make_double3(0,1,0)}
            :std::vector<double3>{make_double3(0,0,0),make_double3(1,0,0),make_double3(.25,-.6,.01),make_double3(.25,.4,.01)};
        cudatool::DeviceBuffer<ToiContact> snapshot;upload(snapshot,frame_friction_copy({c}));upload(x,p);
        const int sentinel=INT_MAX;CUDA_SAFE_CALL(cudaMemcpy(invalid.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<1,32>>>(snapshot.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        // Mutating the live normal contact must not affect a frozen frame snapshot.
        c.lambda=99;c.gamma=.7;upload(contacts,std::vector<ToiContact>{c});
        cudatool::DeviceBuffer<double> forces;forces.resize(1);
        cudatool::DeviceBuffer<double2> coordinates;coordinates.resize(1);
        cudatool::DeviceBuffer<__GEIGEN__::Matrix3x2d> basis;basis.resize(1);
        friction_basis<<<1,32>>>(snapshot.data(),1,forces.data(),coordinates.data(),basis.data(),false);
        const auto uv=download(coordinates.data(),1)[0];const double force=download(forces.data(),1)[0];
        const double u=kind==0?.2:.25,v=kind==0?.3:.6;
        auto basis_error=[&](double3 normal)
        {
            const auto b=download(basis.data(),1)[0];
            const double3 t=make_double3(b.m[0][0],b.m[1][0],b.m[2][0]);
            const double3 q=make_double3(b.m[0][1],b.m[1][1],b.m[2][1]);
            return std::max({std::abs(dot(t,t)-1),std::abs(dot(q,q)-1),
                std::abs(dot(t,q)),std::abs(dot(normal,t)),std::abs(dot(normal,q))});
        };
        const double old_basis_error=basis_error(make_double3(0,0,1));
        bool ok=force==3&&std::abs(uv.x-u)<1e-12&&std::abs(uv.y-v)<1e-12
            &&old_basis_error<1e-12&&download(invalid.data(),1)[0]==sentinel;
        // Next physical frame: rotate normal by 90 degrees and shift closest
        // coordinates. Rebuild with the same production snapshot copy helper.
        if(kind==0){p[0].x=.35;p[0].y=.1;}
        else {p[2].x=p[3].x=.6;p[2].y=-.25;p[3].y=.75;}
        for(auto& point:p)point=make_double3(point.z,point.y,-point.x);
        c.lambda=3;c.gamma=.4;upload(snapshot,frame_friction_copy({c}));upload(x,p);
        linearize<<<1,32>>>(snapshot.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        friction_basis<<<1,32>>>(snapshot.data(),1,forces.data(),coordinates.data(),basis.data(),false);
        const auto next_uv=download(coordinates.data(),1)[0];const double next_force=download(forces.data(),1)[0];
        const double next_basis_error=basis_error(make_double3(1,0,0));
        ok=ok&&next_force==3&&std::abs(next_uv.x-(kind==0?.35:.6))<1e-12
            &&std::abs(next_uv.y-(kind==0?.1:.25))<1e-12&&next_basis_error<1e-12
            &&download(invalid.data(),1)[0]==sentinel;
        passed=passed&&ok;report["port_tests"].push_back({{"case",kind==0?"pt_frame_snapshot":"ee_frame_snapshot"},
            {"force",force},{"coordinates",gipc::Json::array({uv.x,uv.y})},
            {"next_frame_coordinates",gipc::Json::array({next_uv.x,next_uv.y})},
            {"maximum_basis_error",std::max(old_basis_error,next_basis_error)},{"passed",ok}});
    }
    {
        ToiContact c;c.kind=2;c.ids[0]=0;c.lambda=3;c.gamma=.4;
        cudatool::DeviceBuffer<ToiContact> snapshot;upload(snapshot,frame_friction_copy({c}));
        c.lambda=99;c.gamma=.7;upload(contacts,std::vector<ToiContact>{c});
        ground_lambda<<<1,32>>>(snapshot.data(),0,1,force.data(),false);
        const double actual=download(force.data(),1)[0];const bool ok=actual==3;
        passed=passed&&ok;report["port_tests"].push_back({{"case","ground_frame_snapshot"},{"force",actual},{"passed",ok}});
    }
    {
        ToiContact c;for(int i=0;i<4;++i)c.ids[i]=i;
        upload(contacts,std::vector<ToiContact>{c});upload(x,std::vector<double3>(4,make_double3(0,0,0)));
        const int sentinel=INT_MAX;CUDA_SAFE_CALL(cudaMemcpy(invalid.data(),&sentinel,sizeof(int),cudaMemcpyHostToDevice));
        linearize<<<1,32>>>(contacts.data(),1,x.data(),normal.data(),offset.data(),.2,invalid.data());
        const bool ok=download(invalid.data(),1)[0]==1;
        passed=passed&&ok;report["port_tests"].push_back({{"case","singular_geometry_guard"},{"passed",ok}});
    }
    report["volume_step_tests"]=gipc::Json::array();
    report["world_penalty_tests"]=gipc::Json::array();
    {
        gipc::Matrix12x12 a=gipc::Matrix12x12::Identity();
        for(int i=0;i<12;++i)for(int j=0;j<i;++j)a(i,j)=.02*(i+j+1);
        const gipc::Matrix12x12 h=a*a.transpose()+.3*gipc::Matrix12x12::Identity();
        const gipc::ABDJacobi jac(gipc::Vector3(.2,-.3,.5));
        const gipc::Matrix3x12 j=jac.to_mat();
        const auto inverse=gipc::toi_body_compliance(h);
        const Eigen::Matrix3d compliance=j*inverse*j.transpose();
        const Eigen::Matrix3d stiffness=compliance.inverse();
        // Change affine units without changing the physical state/operator.
        gipc::Matrix12x12 t=gipc::Matrix12x12::Identity();
        for(int i=3;i<12;++i)t(i,i)=i%2?.01:20.;
        const gipc::Matrix12x12 changed=t.transpose()*h*t;
        const gipc::Matrix3x12 changed_j=j*t;
        const Eigen::Matrix3d changed_c=changed_j*gipc::toi_body_compliance(changed)*changed_j.transpose();
        const double coordinate_error=(changed_c-compliance).norm()/compliance.norm();
        cudatool::DeviceBuffer<gipc::ABDJacobi> js;upload(js,std::vector<gipc::ABDJacobi>{jac,jac,jac});
        cudatool::DeviceBuffer<gipc::Matrix12x12> inv;upload(inv,std::vector<gipc::Matrix12x12>{inverse,inverse});
        cudatool::DeviceBuffer<int> ids;upload(ids,std::vector<int>{0,1,0});
        cudatool::DeviceBuffer<BodyBoundaryType> types;
        upload(types,std::vector<BodyBoundaryType>{BodyBoundaryType::Free,BodyBoundaryType::Fixed});
        cudatool::DeviceBuffer<double> diagonal;diagonal.resize(3);
        CUDA_SAFE_CALL(cudaMemset(invalid.data(),0,sizeof(int)));
        world_abd_diagonal<<<1,32>>>(js.data(),ids.data(),types.data(),inv.data(),2,3,diagonal.data(),invalid.data());
        const auto actual=download(diagonal.data(),3);
        const double expected=stiffness.diagonal().maxCoeff();
        const double gpu_error=std::max(std::abs(actual[0]-expected),std::abs(actual[2]-expected));
        bool ok=coordinate_error<1e-10&&gpu_error<1e-10&&actual[1]==0&&download(invalid.data(),1)[0]==0;
        passed&=ok;report["world_penalty_tests"].push_back({{"case","full_ABD_compliance_and_fixed_vertex"},
            {"coordinate_relative_error",coordinate_error},{"gpu_vs_cpu_error",gpu_error},{"passed",ok}});
        bool rejected=false;
        try{gipc::Matrix12x12 bad=h;bad(0,0)=-1;gipc::toi_body_compliance(bad);}
        catch(const std::runtime_error&){rejected=true;}
        passed&=rejected;report["world_penalty_tests"].push_back({{"case","nonpositive_ABD_pivot_rejected"},{"passed",rejected}});
        upload(ids,std::vector<int>{2,1,0});CUDA_SAFE_CALL(cudaMemset(invalid.data(),0,sizeof(int)));
        world_abd_diagonal<<<1,32>>>(js.data(),ids.data(),types.data(),inv.data(),2,3,diagonal.data(),invalid.data());
        ok=download(invalid.data(),1)[0]==1;passed&=ok;
        report["world_penalty_tests"].push_back({{"case","invalid_ABD_mapping_rejected"},{"passed",ok}});
    }
    {
        cudatool::DeviceBuffer<double> diagonal;std::vector<double> values(18,300.);
        values[12]=-2;values[13]=3;values[14]=-4;values[15]=values[16]=values[17]=500;
        upload(diagonal,values);
        cudatool::DeviceBuffer<BodyBoundaryType> bodies;upload(bodies,std::vector<BodyBoundaryType>{BodyBoundaryType::Fixed});
        cudatool::DeviceBuffer<int> fem;upload(fem,std::vector<int>{0,1});
        movable_diagonal<<<1,32>>>(diagonal.data(),18,12,bodies.data(),fem.data());
        const auto actual=download(diagonal.data(),18);bool ok=true;
        for(int i=0;i<18;++i)ok=ok&&actual[i]==(i==12?2:i==13?3:i==14?4:0);
        passed=passed&&ok;report["port_tests"].push_back({{"case","movable_mu_diagonal"},
            {"expected_free_max",4},{"actual_free_max",*std::max_element(actual.begin(),actual.end())},{"passed",ok}});
    }
    cudatool::DeviceBuffer<double> volume_result;volume_result.resize(1);
    for(double scale:{1.,1e-3,1e-6})for(int kind=0;kind<5;++kind)
    {
        double expected=1;
        if(kind==1)expected=.1;
        if(kind==2)expected=(1-std::sqrt(.8))/2;
        if(kind==3)expected=(4.5-std::sqrt(16.65))/9;
        if(kind==4)
        {
            double low=0,high=.1;
            for(int i=0;i<80;++i){double mid=(low+high)*.5;if(.2-3.5*mid+4.5*mid*mid*mid>0)low=mid;else high=mid;}
            expected=low;
        }
        volume_fixture<<<1,1>>>(kind,scale,.8,volume_result.data(),invalid.data());
        double actual=download(volume_result.data(),1)[0];int error=download(invalid.data(),1)[0];
        bool ok=error==0&&std::isfinite(actual)&&std::abs(actual-expected)<1e-7;
        passed=passed&&ok;report["volume_step_tests"].push_back({{"kind",kind},{"scale",scale},
            {"retain_volume_fraction",.8},{"expected_alpha",expected},{"actual_alpha",actual},{"passed",ok}});
    }
    for(int kind:{2,3})
    {
        volume_fixture<<<1,1>>>(kind,1e-3,0,volume_result.data(),invalid.data());
        double actual=download(volume_result.data(),1)[0],expected=kind==2?.5:1./3;
        bool ok=download(invalid.data(),1)[0]==0&&std::abs(actual-expected)<1e-7;
        passed=passed&&ok;report["volume_step_tests"].push_back({{"kind",kind},{"scale",1e-3},
            {"retain_volume_fraction",0},{"expected_alpha",expected},{"actual_alpha",actual},{"passed",ok}});
    }
    report["passed"]=passed;report["scope"]="GPU AL derivatives and updates; independent warm normal/friction histories on fixed PT input; normalized determinant bound across scales and interior extrema";
    std::ofstream out(report_path);out<<report.dump(2);if(!out)throw std::runtime_error("Cannot write component validation report");
    return passed?0:1;
}
