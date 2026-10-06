#include <math_constants.h>
namespace {
struct IpcResidualMax {
    __host__ __device__ double2 operator()(double2 a,double2 b)const {
        return make_double2(a.x>b.x?a.x:b.x,a.y>b.y?a.y:b.y);
    }
};
__global__ void ipc_residual_partials(const double* abd,int abd_scalars,const double3* fb,
    const double3* shape,const int* boundary,int fem_offset,int fem_count,double2* partials) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    double2 v=make_double2(0,0);
    if(i<abd_scalars){double a=abd[i];v.x=isfinite(a)?fabs(a):CUDART_INF;}
    if(i<fem_count&&boundary[fem_offset+i]==0){
        const double3 a=fb[fem_offset+i],b=shape[fem_offset+i];
        const double x=a.x+b.x,y=a.y+b.y,z=a.z+b.z;
        v.y=(!isfinite(x)||!isfinite(y)||!isfinite(z))?CUDART_INF:fmax(fabs(x),fmax(fabs(y),fabs(z)));
    }
    using Reduce=cub::BlockReduce<double2,256>;
    __shared__ typename Reduce::TempStorage storage;
    const double2 result=Reduce(storage).Reduce(v,IpcResidualMax{});
    if(threadIdx.x==0)partials[blockIdx.x]=result;
}
double2 ipc_measure_residual(const double* abd,int abd_scalars,const double3* fb,
    const double3* shape,const int* boundary,int fem_offset,int fem_count) {
    static thread_local cudatool::DeviceBuffer<double2> partials,result;
    static thread_local cudatool::DeviceBuffer<unsigned char> scratch;
    const int blocks=(std::max(abd_scalars,fem_count)+255)/256;
    if(!blocks)return make_double2(0,0);
    gipc::CostScope cost("ipc.residual_reduce_and_readback");
    partials.resize(blocks);result.resize(1);
    ipc_residual_partials<<<blocks,256>>>(abd,abd_scalars,fb,shape,boundary,fem_offset,fem_count,partials.data());
    CUDA_SAFE_CALL(cudaGetLastError());
    size_t bytes=0;
    CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(nullptr,bytes,partials.data(),result.data(),blocks,IpcResidualMax{},make_double2(0,0),cudaStreamPerThread));
    scratch.resize(bytes);
    CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(scratch.data(),bytes,partials.data(),result.data(),blocks,IpcResidualMax{},make_double2(0,0),cudaStreamPerThread));
    double2 host;CUDA_SAFE_CALL(cudaMemcpy(&host,result.data(),sizeof(host),cudaMemcpyDeviceToHost));
    if(const char* flag=std::getenv("GIPC_IPC_RESIDUAL_CPU_AUDIT");flag&&std::string(flag)=="1"){
        std::vector<double> ag(abd_scalars);std::vector<double3> fg(fem_count),sg(fem_count);
        std::vector<int> mask(fem_count);
        if(abd_scalars)CUDA_SAFE_CALL(cudaMemcpy(ag.data(),abd,abd_scalars*sizeof(double),cudaMemcpyDeviceToHost));
        if(fem_count){
            CUDA_SAFE_CALL(cudaMemcpy(fg.data(),fb+fem_offset,fem_count*sizeof(double3),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(sg.data(),shape+fem_offset,fem_count*sizeof(double3),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(mask.data(),boundary+fem_offset,fem_count*sizeof(int),cudaMemcpyDeviceToHost));
        }
        double2 expected=make_double2(0,0);
        for(double a:ag)expected.x=std::max(expected.x,std::isfinite(a)?std::abs(a):std::numeric_limits<double>::infinity());
        for(int i=0;i<fem_count;++i)if(mask[i]==0){
            for(double a:{fg[i].x+sg[i].x,fg[i].y+sg[i].y,fg[i].z+sg[i].z})
                expected.y=std::max(expected.y,std::isfinite(a)?std::abs(a):std::numeric_limits<double>::infinity());
        }
        auto matches=[](double a,double b){return a==b||(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}));};
        if(!matches(host.x,expected.x)||!matches(host.y,expected.y))throw std::runtime_error("Nonlinear GPU residual differs from CPU reference");
    }
    return host;
}
}
