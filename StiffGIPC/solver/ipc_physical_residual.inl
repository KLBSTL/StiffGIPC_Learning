// Full equation residual in velocity units: max |J M_free^-1 g| / dt.
// It is a forcing monitor, not a physical trajectory-error certificate.
#include <abd_system/abd_jacobi_matrix.h>
#include <core/body_boundary_type.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

namespace {
struct IpcPhysicalResidualMaximum {
    __host__ __device__ double operator()(double a,double b)const{return a>b?a:b;}
};
__global__ void ipc_mass_residual_abd(int count,const double* gradient,
    const gipc::Matrix12x12* inverse_mass,const BodyBoundaryType* boundary,
    gipc::Vector12* result) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(boundary[i]==BodyBoundaryType::Fixed){result[i].setZero();return;}
    gipc::Vector12 g;for(int j=0;j<12;++j)g[j]=gradient[12*i+j];
    result[i]=inverse_mass[i]*g;
}
__global__ void ipc_mass_residual_max(int abd_vertices,int bodies,const gipc::ABDJacobi* jacobian,
    const int* body_ids,const gipc::Vector12* abd_residual,int fem_offset,int fem_count,
    const double3* barrier,const double3* shape,const double* mass,const int* boundary,
    double inverse_dt,double* partials) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;double maximum=0;
    if(i<abd_vertices) {
        const int body=body_ids[i];
        if(body<0||body>=bodies)maximum=CUDART_INF;
        else {
            const gipc::Vector3 value=jacobian[i]*abd_residual[body];
            const double norm=value.norm()*inverse_dt;
            maximum=isfinite(norm)?norm:CUDART_INF;
        }
    }
    if(i<fem_count&&boundary[fem_offset+i]==0) {
        const int vertex=fem_offset+i;
        const double m=mass[vertex];
        const double x=barrier[vertex].x+shape[vertex].x;
        const double y=barrier[vertex].y+shape[vertex].y;
        const double z=barrier[vertex].z+shape[vertex].z;
        const double norm=sqrt(x*x+y*y+z*z)*inverse_dt/m;
        if(!isfinite(m)||m<=0||!isfinite(norm))maximum=CUDART_INF;
        else maximum=fmax(maximum,norm);
    }
    using Reduce=cub::BlockReduce<double,256>;
    __shared__ typename Reduce::TempStorage storage;
    const double result=Reduce(storage).Reduce(maximum,IpcPhysicalResidualMaximum{});
    if(threadIdx.x==0)partials[blockIdx.x]=result;
}
double ipc_physical_residual(const double* abd_gradient,const gipc::Matrix12x12* inverse_mass,
    const BodyBoundaryType* abd_boundary,int bodies,const gipc::ABDJacobi* jacobian,
    const int* body_ids,int abd_vertices,const double3* barrier,const double3* shape,
    const double* mass,const int* boundary,int fem_offset,int fem_count,double dt) {
    if(!std::isfinite(dt)||dt<=0)throw std::runtime_error("Invalid dt for physical residual");
    static thread_local cudatool::DeviceBuffer<gipc::Vector12> abd_result;
    static thread_local cudatool::DeviceBuffer<double> partials,result;
    static thread_local cudatool::DeviceBuffer<unsigned char> scratch;
    const int blocks=(std::max(abd_vertices,fem_count)+255)/256;
    if(!blocks)return 0;
    abd_result.resize(bodies);partials.resize(blocks);result.resize(1);
    if(bodies)ipc_mass_residual_abd<<<(bodies+255)/256,256>>>(bodies,abd_gradient,inverse_mass,abd_boundary,abd_result.data());
    ipc_mass_residual_max<<<blocks,256>>>(abd_vertices,bodies,jacobian,body_ids,abd_result.data(),
        fem_offset,fem_count,barrier,shape,mass,boundary,1/dt,partials.data());
    CUDA_SAFE_CALL(cudaGetLastError());
    size_t bytes=0;
    CUDA_SAFE_CALL(cub::DeviceReduce::Max(nullptr,bytes,partials.data(),result.data(),blocks,cudaStreamPerThread));
    scratch.resize(bytes);
    CUDA_SAFE_CALL(cub::DeviceReduce::Max(scratch.data(),bytes,partials.data(),result.data(),blocks,cudaStreamPerThread));
    double host;CUDA_SAFE_CALL(cudaMemcpy(&host,result.data(),sizeof(host),cudaMemcpyDeviceToHost));
    return host;
}
}
