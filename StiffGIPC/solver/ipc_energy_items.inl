// Failure-only shadow evaluation. Each item calls the production expression;
// no production kernel, gradient, candidate list, or acceptance rule changes.
namespace gipc_x1 {
__global__ void inertia(double* out,const double3* x,const double3* target,const double* mass,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i]=__GEIGEN__::__squaredNorm(__GEIGEN__::__minus(x[i],target[i]))*mass[i]*.5;
}
__global__ void stretch(double* out,const double3* x,const uint3* triangles,
    const __GEIGEN__::Matrix2x2d* inverse,const double* area,int n,double a,double b,double c) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i]=__cal_BaraffWitkinStretch_energy(x,triangles[i],inverse[i],area[i],a,b,c);
}
#ifdef USE_QUADRATIC_BENDING
__global__ void bending(double* out,const double3* x,const double3* rest,const uint2* edges,
    const uint2* adj,const Eigen::Matrix4d* q,int n,double stiffness) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i]=__cal_quad_bending_energy(x,rest,edges[i],adj[i],q[i],stiffness);
}
#endif
__global__ void barrier(double* out,const double3* x,const double3* rest,const int4* pairs,
    int n,double kappa,double dhat) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i]=__cal_Barrier_energy(x,rest,pairs[i],kappa,dhat);
}
inline void save(const std::filesystem::path& file,const void* ptr,size_t bytes) {
    std::ofstream out(file,std::ios::binary);out.write(static_cast<const char*>(ptr),bytes);
    if(!out)throw std::runtime_error("X1 item export failed");
}
}
