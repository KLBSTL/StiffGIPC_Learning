// Standalone, frozen-input cost screen. No StiffGIPC headers or production state.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <nlohmann/json.hpp>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using Json=nlohmann::json;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
#define CU(x) do{auto s=(x);if(s!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(s));}while(0)
#define BL(x) do{auto s=(x);if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(#x)+": "+std::to_string(int(s)));}while(0)
#define SO(x) do{auto s=(x);if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error(std::string(#x)+": "+std::to_string(int(s)));}while(0)
#define SP(x) do{auto s=(x);if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(std::string(#x)+": "+std::to_string(int(s)));}while(0)
double millis(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
constexpr double one=1.,zero=0.,minus_one=-1.;

struct Device {
    cudaStream_t stream{};cublasHandle_t blas{};cusolverDnHandle_t solver{};cusparseHandle_t sparse{};
    std::vector<void*> allocations;size_t bytes=0;
    Device(){CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));BL(cublasCreate(&blas));
        SO(cusolverDnCreate(&solver));SP(cusparseCreate(&sparse));BL(cublasSetStream(blas,stream));
        SO(cusolverDnSetStream(solver,stream));SP(cusparseSetStream(sparse,stream));
        BL(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));BL(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));}
    template<class T>T* alloc(size_t n){need(n>0&&n<((size_t(1)<<30)/sizeof(T)),"Allocation size guard");
        T* p=nullptr;CU(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));allocations.push_back(p);bytes+=n*sizeof(T);
        need(bytes<(size_t(1)<<30),"1GiB explicit-allocation budget");return p;}
    template<class T>std::vector<T> host(const T* p,size_t n){std::vector<T> v(n);
        CU(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));return v;}
    ~Device(){if(stream)cudaStreamSynchronize(stream);if(sparse)cusparseDestroy(sparse);if(solver)cusolverDnDestroy(solver);
        if(blas)cublasDestroy(blas);for(void* p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
};
template<class T>std::vector<T> load(const fs::path& folder,const Json& meta,const std::string& name){
    const auto& a=meta.at("arrays").at(name);size_t n=1;
    for(auto& x:a.at("shape"))n*=x.get<size_t>();
    need(n>0&&n<(size_t(1)<<29),"Invalid input dimensions");
    const auto path=folder/a.at("file").get<std::string>();
    need(fs::file_size(path)==n*sizeof(T),"Input byte count: "+name);
    std::vector<T> v(n);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
    need(bool(f),"Input read: "+name);return v;
}
template<class T>T* upload(Device& d,const std::vector<T>& v){auto p=d.alloc<T>(v.size());
    CU(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,d.stream));
    // Input vectors are temporary host objects, so complete their copies here.
    CU(cudaStreamSynchronize(d.stream));return p;}
template<class T>void save(const fs::path& path,const std::vector<T>& v){need(!fs::exists(path),"Output already exists");
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));need(bool(f),"Output write");}
struct Sparse {
    cusparseSpMatDescr_t descriptor{};int n;int* rows;int* cols;double* values;
    Sparse(Device& d,const fs::path& folder,const Json& m,const std::string& label,int size):n(size){
        auto r=load<int>(folder,m,label+"_rows");auto c=load<int>(folder,m,label+"_cols");auto v=load<double>(folder,m,label+"_values");
        need(r.size()==size_t(n+1)&&r.front()==0&&r.back()==int(v.size())&&v.size()==c.size(),"CSR shape mismatch");
        for(int i=0;i<n;++i)need(r[i]<=r[i+1],"Nonmonotone CSR offsets");
        for(int j:c)need(j>=0&&j<n,"CSR column out of bounds");
        rows=upload(d,r);cols=upload(d,c);values=upload(d,v);
        SP(cusparseCreateCsr(&descriptor,n,n,v.size(),rows,cols,values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));}
    ~Sparse(){if(descriptor)cusparseDestroySpMat(descriptor);}
};
struct Product {
    Device& d;Sparse& a;cusparseDnMatDescr_t input{},output{};void* workspace{};
    Product(Device& dev,Sparse& matrix,int width,double* x,double* y):d(dev),a(matrix){
        SP(cusparseCreateDnMat(&input,a.n,width,a.n,x,CUDA_R_64F,CUSPARSE_ORDER_COL));
        SP(cusparseCreateDnMat(&output,a.n,width,a.n,y,CUDA_R_64F,CUSPARSE_ORDER_COL));size_t bytes=0;
        SP(cusparseSpMM_bufferSize(d.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a.descriptor,input,&zero,output,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,&bytes));
        workspace=d.alloc<char>(std::max(size_t(1),bytes));}
    void run(){SP(cusparseSpMM(d.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a.descriptor,input,&zero,output,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,workspace));}
    ~Product(){if(input)cusparseDestroyDnMat(input);if(output)cusparseDestroyDnMat(output);}
};
__global__ void restriction(int count,const int* row,const int* col,const double* weights,const double* input,double* output){
    const int t=int(blockIdx.x*blockDim.x+threadIdx.x);if(t>=3*count)return;
    const int node=t/3,axis=t%3;double value=0;
    for(int k=row[node];k<row[node+1];++k)value+=weights[k]*input[3*col[k]+axis];output[t]=value;
}
template<int Size>__global__ void local_inverse(const double* matrix,const double* input,double* output){
    __shared__ double v[Size];int row=threadIdx.x;
    if(row<Size)v[row]=input[blockIdx.x*Size+row];__syncthreads();
    if(row<Size){double sum=0;for(int k=0;k<Size;++k)sum+=matrix[blockIdx.x*Size*Size+row+Size*k]*v[k];output[blockIdx.x*Size+row]=sum;}
}
__global__ void difference(int n,int m,const double* final_x,const double* bank,double* s){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(n)*m)s[i]=final_x[i%n]-bank[i];
}
__global__ void symmetric(int n,double* matrix){int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*n){int r=i%n,c=i/n;if(r<c){double value=.5*(matrix[r+n*c]+matrix[c+n*r]);matrix[r+n*c]=matrix[c+n*r]=value;}}}
__global__ void select_modes(int m,int r,const double* eigenvectors,double* u){int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*r)u[i]=eigenvectors[(i%m)+m*(m-1-i/m)];}
__global__ void diagonals(int n,int r,const double* q,double* diagonal){if(threadIdx.x<r)diagonal[threadIdx.x]=q[threadIdx.x+n*threadIdx.x];}
__global__ void tiny_solve(int r,const double* l,double* x){if(threadIdx.x==0){
    for(int i=0;i<r;++i){double v=x[i];for(int j=0;j<i;++j)v-=l[i+r*j]*x[j];x[i]=v/l[i+r*i];}
    for(int i=r-1;i>=0;--i){double v=x[i];for(int j=i+1;j<r;++j)v-=l[j+r*i]*x[j];x[i]=v/l[i+r*i];}}}
__global__ void coefficients(int r,const double* a,double* b){if(threadIdx.x<r)b[threadIdx.x]=a[threadIdx.x]-b[threadIdx.x];}

struct Mas {
    Device& d;int n,nodes,mapped,blocks,offset,abd_offset;
    int *rr,*rc,*tr,*tc;double *rv,*tv,*inverse,*abd,*restricted,*local;
    Mas(Device& dev,const fs::path& folder,const Json& m):d(dev),n(m.at("n")),nodes(m.at("mas_nodes")),
        mapped(m.at("mas_rows")),blocks(m.at("mas_blocks")),offset(m.at("mas_offset")),abd_offset(m.at("abd_offset")){
        need(3*mapped==48*blocks&&offset>=0&&offset+3*nodes<=n&&abd_offset>=0&&abd_offset+12<=n,
             "MAS/ABD dimensions invalid");
        auto R=load<int>(folder,m,"R_rows"),C=load<int>(folder,m,"R_cols");auto V=load<double>(folder,m,"R_values");
        auto T=load<int>(folder,m,"RT_rows"),U=load<int>(folder,m,"RT_cols");auto W=load<double>(folder,m,"RT_values");
        need(R.size()==size_t(mapped+1)&&T.size()==size_t(nodes+1)&&R.front()==0&&T.front()==0&&
             R.back()==int(C.size())&&T.back()==int(U.size())&&C.size()==V.size()&&U.size()==W.size(),"MAS CSR shape");
        for(int i:C)need(i>=0&&i<nodes,"R index");for(int i:U)need(i>=0&&i<mapped,"RT index");
        rr=upload(d,R);rc=upload(d,C);rv=upload(d,V);tr=upload(d,T);tc=upload(d,U);tv=upload(d,W);
        auto matrices=load<double>(folder,m,"mas_inverse"),ai=load<double>(folder,m,"abd_inverse");
        need(matrices.size()==size_t(blocks)*48*48&&ai.size()==144,"MAS dense matrix shape");
        inverse=upload(d,matrices);abd=upload(d,ai);restricted=d.alloc<double>(3*mapped);local=d.alloc<double>(3*mapped);}
    void apply(const double* input,double* output){
        CU(cudaMemcpyAsync(output,input,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
        local_inverse<12><<<1,16,0,d.stream>>>(abd,input+abd_offset,output+abd_offset);
        restriction<<<(3*mapped+255)/256,256,0,d.stream>>>(mapped,rr,rc,rv,input+offset,restricted);
        local_inverse<48><<<blocks,64,0,d.stream>>>(inverse,restricted,local);
        restriction<<<(3*nodes+255)/256,256,0,d.stream>>>(nodes,tr,tc,tv,local,output+offset);CU(cudaGetLastError());}
};
Json timed(Device& d,const std::function<void()>& f){cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
    CU(cudaStreamSynchronize(d.stream));auto start=Clock::now();CU(cudaEventRecord(a,d.stream));f();
    CU(cudaEventRecord(b,d.stream));CU(cudaEventSynchronize(b));float elapsed=0;CU(cudaEventElapsedTime(&elapsed,a,b));double wall=millis(start);
    CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));return Json{{"gpu_ms",elapsed},{"wall_ms",wall}};}
double relative(const std::vector<double>& x,const std::vector<double>& y){need(x.size()==y.size(),"Compare vector size");
    long double diff=0,norm=0;for(size_t i=0;i<x.size();++i){need(std::isfinite(x[i])&&std::isfinite(y[i]),"Nonfinite output");
        long double e=(long double)x[i]-y[i];diff+=e*e;norm+=(long double)y[i]*y[i];}return std::sqrt(double(diff/std::max(norm,1.e-300L)));}
struct Replay {
    cudaGraph_t graph{};cudaGraphExec_t executable{};Device& d;double capture_ms;
    Replay(Device& dev,const std::function<void()>& f):d(dev){auto start=Clock::now();
        CU(cudaStreamBeginCapture(d.stream,cudaStreamCaptureModeThreadLocal));f();CU(cudaStreamEndCapture(d.stream,&graph));
        CU(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));capture_ms=millis(start);}
    void run(){CU(cudaGraphLaunch(executable,d.stream));}
    ~Replay(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};

