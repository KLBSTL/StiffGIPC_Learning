#pragma once
#include <array>
#include <cstdlib>
#include <functional>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "native_balanced_math.h"

namespace gipc {
inline void balanced_cuda(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("Native balanced CUDA: ")+cudaGetErrorString(status));
}
inline void balanced_blas(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Native balanced BLAS: "+std::to_string(int(status)));
}
template<class T>struct BalancedBuffer {
    T* value=nullptr;size_t size=0;
    explicit BalancedBuffer(size_t count):size(count) {balanced_cuda(cudaMalloc(reinterpret_cast<void**>(&value),count*sizeof(T)));}
    ~BalancedBuffer(){if(value)cudaFree(value);}
    BalancedBuffer(const BalancedBuffer&)=delete;BalancedBuffer& operator=(const BalancedBuffer&)=delete;
};
struct BalancedBlas {
    cublasHandle_t handle=nullptr;
    BalancedBlas() {
        balanced_blas(cublasCreate(&handle));
        try {
            balanced_blas(cublasSetStream(handle,cudaStreamPerThread));
            balanced_blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
            balanced_blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
            balanced_blas(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
        } catch(...){cublasDestroy(handle);handle=nullptr;throw;}
    }
    ~BalancedBlas(){if(handle)cublasDestroy(handle);}
};
static __global__ void balanced_source_Z(int n,int width,const double* bank,const int* slots,
    const double* final_x,const double* coefficients,double* Z) {
    const int item=blockIdx.x*blockDim.x+threadIdx.x;
    if(item>=n*8)return;const int i=item%n,j=item/n;double sum=0;
    for(int k=0;k<width;++k)sum+=(final_x[i]-bank[size_t(slots[k])*n+i])*coefficients[k+size_t(j)*width];
    Z[item]=sum;
}
static __global__ void balanced_tiny_solve(const double* L,double* a) {
    if(threadIdx.x||blockIdx.x)return;
    for(int i=0;i<8;++i){double t=a[i];for(int j=0;j<i;++j)t-=L[i+8*j]*a[j];a[i]=t/L[i+8*i];}
    for(int i=7;i>=0;--i){double t=a[i];for(int j=i+1;j<8;++j)t-=L[j+8*i]*a[j];a[i]=t/L[i+8*i];}
}
static __global__ void balanced_difference(const double* a,double* b) {
    if(threadIdx.x<8)b[threadIdx.x]=a[threadIdx.x]-b[threadIdx.x];
}
// Column-major n x 8 storage, identical to the reference GEMV inputs.
// One thread owns one output; coefficients and matrices remain read-only.
static __global__ void balanced_input_projection(int n,const double* r,const double* W,
                                                const double* a,double* t) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double sum=0;
    #pragma unroll
    for(int j=0;j<8;++j)sum+=W[i+size_t(j)*n]*a[j];
    t[i]=r[i]-sum;
}
static __global__ void balanced_output_projection(int n,const double* Z,const double* a,
                                                 const double* c,double* z) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double sum=0;
    #pragma unroll
    for(int j=0;j<8;++j)sum+=Z[i+size_t(j)*n]*(a[j]-c[j]);
    z[i]+=sum;
}
// All storage and BLAS initialization precede timed solves. No production state.
struct NativeBalancedSpace {
    const int n;BalancedBlas blas;
    BalancedBuffer<double> Z,W,E,L,intermediate,a,c;
    double condition=0;Eigen::MatrixXd host_E;
    explicit NativeBalancedSpace(int count):n(count),Z(size_t(count)*8),W(size_t(count)*8),E(64),L(64),
        intermediate(count),a(8),c(8) {}
    void source(const double* bank,const std::vector<int>& slots,const double* final_x,const Eigen::MatrixXd& U) {
        balanced_require(U.rows()==slots.size()&&U.cols()==8&&U.allFinite(),"Source coefficient shape invalid");
        BalancedBuffer<int> ds(slots.size());BalancedBuffer<double> du(size_t(U.size()));
        balanced_cuda(cudaMemcpyAsync(ds.value,slots.data(),slots.size()*sizeof(int),cudaMemcpyHostToDevice,cudaStreamPerThread));
        balanced_cuda(cudaMemcpyAsync(du.value,U.data(),size_t(U.size())*sizeof(double),cudaMemcpyHostToDevice,cudaStreamPerThread));
        balanced_source_Z<<<(n*8+255)/256,256,0,cudaStreamPerThread>>>(n,int(slots.size()),bank,ds.value,final_x,du.value,Z.value);
        balanced_cuda(cudaGetLastError());balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    void target(const std::function<void(const double*,double*)>& multiply) {
        for(int j=0;j<8;++j)multiply(Z.value+size_t(j)*n,W.value+size_t(j)*n);
        const double one=1,zero=0;
        balanced_blas(cublasDgemm(blas.handle,CUBLAS_OP_T,CUBLAS_OP_N,8,8,n,&one,Z.value,n,W.value,n,&zero,E.value,8));
        host_E.resize(8,8);
        balanced_cuda(cudaMemcpyAsync(host_E.data(),E.value,64*sizeof(double),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Eigen::MatrixXd lower=native_coarse_factor(host_E,condition);
        balanced_cuda(cudaMemcpyAsync(L.value,lower.data(),64*sizeof(double),cudaMemcpyHostToDevice,cudaStreamPerThread));
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    using Profile=std::function<void(const char*,const std::function<void()>&)>;
    enum class ApplyMode { Reference,FusedProjection };
    void apply(const double* r,double* z,const std::function<void(const double*,double*)>& base,
               const Profile* profile=nullptr,ApplyMode mode=ApplyMode::Reference) {
        const double one=1,zero=0,minus=-1;
        auto stage=[&](const char* name,auto operation){if(profile)(*profile)(name,operation);else operation();};
        stage("Z_transpose_and_first_coarse_solve",[&]{
        balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_T,n,8,&one,Z.value,n,r,1,&zero,a.value,1));
        balanced_tiny_solve<<<1,1,0,cudaStreamPerThread>>>(L.value,a.value);
        });
        stage("input_copy_and_W_projection",[&]{
        if(mode==ApplyMode::FusedProjection)
            balanced_input_projection<<<(n+255)/256,256,0,cudaStreamPerThread>>>(n,r,W.value,a.value,intermediate.value);
        else {
        balanced_cuda(cudaMemcpyAsync(intermediate.value,r,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_N,n,8,&minus,W.value,n,a.value,1,&one,intermediate.value,1));
        }
        });
        stage("native_MAS",[&]{base(intermediate.value,z);});
        stage("W_transpose_and_second_coarse_solve",[&]{
        balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_T,n,8,&one,W.value,n,z,1,&zero,c.value,1));
        balanced_tiny_solve<<<1,1,0,cudaStreamPerThread>>>(L.value,c.value);
        });
        stage("coefficient_difference_and_Z_projection",[&]{
        if(mode==ApplyMode::FusedProjection)
            balanced_output_projection<<<(n+255)/256,256,0,cudaStreamPerThread>>>(n,Z.value,a.value,c.value,z);
        else {
        balanced_difference<<<1,8,0,cudaStreamPerThread>>>(a.value,c.value);
        balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_N,n,8,&one,Z.value,n,c.value,1,&one,z,1));
        }
        });
        balanced_cuda(cudaGetLastError());
    }
};
} // namespace gipc
