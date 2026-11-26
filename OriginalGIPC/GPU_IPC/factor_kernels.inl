// Three fixed representations of the archived GIPC 96x96 local MAS operator.
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace factorbench {
using gpuipc::DeviceBuffer;
using Dense = __GEIGEN__::Matrix96x96T;
using Packed = __GEIGEN__::MasMatrixSymf;
constexpr int size96=96;
int arm=0;
bool capture=false;
static DeviceBuffer<Dense> snapshot;
void save(Dense* matrix,int count) {
    snapshot.resize_discard(count);
    GPUIPC_CHECK_CUDA(cudaMemcpy(snapshot.data(),matrix,count*sizeof(Dense),cudaMemcpyDeviceToDevice));
}
void restore(Dense* matrix,int count) {
    GPUIPC_CHECK_CUDA(cudaMemcpy(matrix,snapshot.data(),count*sizeof(Dense),cudaMemcpyDeviceToDevice));
}
struct State {
    cusolverDnHandle_t solver=nullptr;
    cublasHandle_t blas=nullptr;
    DeviceBuffer<double*> a,x;
    DeviceBuffer<Dense> inverse;
    DeviceBuffer<int> info,bad;
    DeviceBuffer<float> factor;
};
static State workspace[2]; // Each method pays for its own persistent workspace and handles.
template<class Status> void check(Status status,const char* operation) {
    if(int(status)!=0) throw std::runtime_error(std::string(operation)+" status "+std::to_string(int(status)));
}
__global__ void initializeBatch(Dense* blocks,double** ptrs,Dense* inverse,double** invPtrs) {
    int b=blockIdx.x;
    double* data=&blocks[b].m[0][0];
    if(!threadIdx.x) {
        ptrs[b]=data;
        if(inverse) invPtrs[b]=&inverse[b].m[0][0];
    }
    // Exactly the same zero-diagonal padding as the public inverse kernel.
    if(threadIdx.x<size96 && data[threadIdx.x*97]==0) data[threadIdx.x*97]=1.0;
    if(inverse) for(int i=threadIdx.x;i<size96*size96;i+=blockDim.x)
        (&inverse[b].m[0][0])[i]=(i%97==0)?1.0:0.0;
}
__global__ void checkPivots(const int* info,int n,int* bad) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && info[i]) atomicMax(bad,i+1);
}
__global__ void packInverse(const Dense* inverse,Packed* result) {
    int b=blockIdx.x;
    for(int i=threadIdx.x;i<96*96;i+=blockDim.x) {
        int row=i/96,col=i%96,vr=row/3,vc=col/3;
        if(vc>=vr) result[b].M[32*vr-vr*(vr+1)/2+vc].m[row%3][col%3]=
            inverse[b].m[col][row];  // cuBLAS column-major output
    }
}
__global__ void packFactor(const Dense* a,float* out) {
    int b=blockIdx.x;
    for(int i=threadIdx.x;i<96*96;i+=blockDim.x){
        int row=i%96,col=i/96;
        // Mirroring L provides coalesced column access for both substitutions.
        double value=a[b].m[min(row,col)][max(row,col)];
        out[size_t(b)*96*96+i]=row==col?float(1.0/value):float(value);
    }
}
void build(Dense* matrix,Packed* packed,int count) {
    auto& s=workspace[arm-1];
    if(!s.solver) {
        check(cusolverDnCreate(&s.solver),"cusolverDnCreate");
        check(cusolverDnSetStream(s.solver,cudaStreamPerThread),"cusolverDnSetStream");
    }
    if(arm==1 && !s.blas) {
        check(cublasCreate(&s.blas),"cublasCreate");
        check(cublasSetStream(s.blas,cudaStreamPerThread),"cublasSetStream");
    }
    s.a.resize_discard(count); s.info.resize_discard(count); s.bad.resize(1); s.bad.reset_zero();
    if(arm==1){s.inverse.resize_discard(count);s.x.resize_discard(count);}
    else s.factor.resize_discard(size_t(count)*96*96);
    initializeBatch<<<count,256>>>(matrix,s.a.data(),s.inverse.data(),s.x.data());
    check(cusolverDnDpotrfBatched(s.solver,CUBLAS_FILL_MODE_LOWER,96,s.a.data(),96,s.info.data(),count),"DpotrfBatched");
    checkPivots<<<(count+127)/128,128>>>(s.info.data(),count,s.bad.data());
    int bad=0;
    GPUIPC_CHECK_CUDA(cudaMemcpy(&bad,s.bad.data(),sizeof(int),cudaMemcpyDeviceToHost));
    if(bad) throw std::runtime_error("nonpositive local block "+std::to_string(bad-1));
    if(arm==1) {
        const double one=1.0;
        for(auto op:{CUBLAS_OP_N,CUBLAS_OP_T})
            check(cublasDtrsmBatched(s.blas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,op,CUBLAS_DIAG_NON_UNIT,
                  96,96,&one,s.a.data(),96,s.x.data(),96,count),"DtrsmBatched");
        packInverse<<<count,256>>>(s.inverse.data(),packed);
    } else packFactor<<<count,256>>>(matrix,s.factor.data());
    GPUIPC_CHECK_CUDA(cudaGetLastError());
}
// One warp per local block. Three live RHS entries per lane, one warp broadcast per pivot.
__global__ void triangular(const float* matrices,const float* rhs,float* result,int count) {
    int lane=threadIdx.x%32,b=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(b>=count)return;
    const float* a=matrices+size_t(b)*96*96;
    float x[3]={rhs[96*b+lane],rhs[96*b+32+lane],rhs[96*b+64+lane]};
    for(int pass=0;pass<2;pass++) {
        for(int p=0;p<3;p++) {
            int slice=pass==0?p:2-p;
            for(int j=0;j<32;j++) {
                int owner=pass==0?j:31-j,k=slice*32+owner;
                if(lane==owner) x[slice]*=a[k*96+k];
                float pivot=__shfl_sync(0xffffffff,x[slice],owner);
                #pragma unroll
                for(int part=0;part<3;part++) {
                    int row=part*32+lane;
                    if(pass==0?row>k:row<k) x[part]-=a[k*96+row]*pivot;
                }
            }
        }
    }
    #pragma unroll
    for(int part=0;part<3;part++) result[96*b+32*part+lane]=x[part];
}
void apply(const Precision_T3* rhs,Precision_T3* result,int count) {
    static_assert(sizeof(Precision_T3)==3*sizeof(float));
    triangular<<<(count+3)/4,128>>>(workspace[1].factor.data(),reinterpret_cast<const float*>(rhs),
                                   reinterpret_cast<float*>(result),count);
}
}
