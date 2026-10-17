// Standalone, frozen-input cost screen. No StiffGIPC headers or production state.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <nlohmann/json.hpp>
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

void run(const fs::path& folder,const fs::path& output){
    need(fs::is_directory(output)&&fs::is_empty(output),"Fresh empty output directory required");
    Json m;std::ifstream(folder/"manifest.json")>>m;need(m.at("schema")=="p5_pod_gpu_input.v1","Input ABI");
    const int n=m.at("n"),width=m.at("history_width"),iterations=m.at("source_iterations");
    need(n>16&&width>=16&&width<=128&&iterations>=width&&iterations<=2000,"Dimension budget");
    size_t free_before,total;CU(cudaMemGetInfo(&free_before,&total));need(free_before>size_t(2)*1024*1024*1024,"2GiB free memory required");
    auto initialization=Clock::now();Device d;Sparse source(d,folder,m,"source",n),target(d,folder,m,"target",n);Mas mas(d,folder,m);
    auto all=load<double>(folder,m,"iterates"),last=load<double>(folder,m,"final_source_x"),probes=load<double>(folder,m,"probes");
    need(all.size()==size_t(n)*iterations&&last.size()==size_t(n)&&probes.size()==size_t(n)*3,"History/probe shape");
    auto writes=load<int>(folder,m,"history_writes");need(writes.size()%2==0,"History copy ABI");
    for(size_t i=0;i<writes.size();i+=2)need(writes[i]>=0&&writes[i]<iterations&&writes[i+1]>=0&&writes[i+1]<width,"History copy bounds");
    double* X=upload(d,all);double* final_x=upload(d,last);double* input=upload(d,probes);
    double* bank=d.alloc<double>(size_t(n)*width);double* S=d.alloc<double>(size_t(n)*width);double* AS=d.alloc<double>(size_t(n)*width);
    double* G=d.alloc<double>(width*width);double* eigen=d.alloc<double>(width);double* U=d.alloc<double>(width*16);
    double* Z=d.alloc<double>(size_t(n)*16);double* W=d.alloc<double>(size_t(n)*16);double* E=d.alloc<double>(256);
    double* tau=d.alloc<double>(16);double* diag=d.alloc<double>(16);int* info=d.alloc<int>(1);
    double* intermediate=d.alloc<double>(n);double* answer=d.alloc<double>(n);double* a=d.alloc<double>(16);double* b=d.alloc<double>(16);
    Product source_product(d,source,width,S,AS);
    int workspace_size=1,value;
    SO(cusolverDnDsyevd_bufferSize(d.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,width,G,width,eigen,&value));workspace_size=std::max(workspace_size,value);
    for(int rank:{8,16}){
        SO(cusolverDnDgeqrf_bufferSize(d.solver,n,rank,Z,n,&value));workspace_size=std::max(workspace_size,value);
        SO(cusolverDnDorgqr_bufferSize(d.solver,n,rank,rank,Z,n,tau,&value));workspace_size=std::max(workspace_size,value);
        SO(cusolverDnDpotrf_bufferSize(d.solver,CUBLAS_FILL_MODE_LOWER,rank,E,rank,&value));workspace_size=std::max(workspace_size,value);}
    double* workspace=d.alloc<double>(workspace_size);CU(cudaStreamSynchronize(d.stream));
    Json result={{"schema","p5_pod_cost.v1"},{"n",n},{"history_width",width},{"history_copy_calls",writes.size()/2},
        {"explicit_device_bytes",d.bytes},{"initialization_and_upload_wall_ms",millis(initialization)},
        {"scope","Frozen coefficient FP64 MAS; isolated source history copy replay and setup/apply, NOT native PCG speed"},
        {"ranks",Json::array()}};
    int version=0;CU(cudaRuntimeGetVersion(&version));result["cuda_runtime_version"]=version;
    cudaDeviceProp property;CU(cudaGetDeviceProperties(&property,0));result["device"]=property.name;
    auto check_info=[&](const char* label){need(d.host(info,1)[0]==0,std::string(label)+" solver info !=0");};
    auto capture=[&](){for(size_t i=0;i<writes.size();i+=2)
        CU(cudaMemcpyAsync(bank+size_t(n)*writes[i+1],X+size_t(n)*writes[i],size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));};
    auto make_S=[&](){difference<<<unsigned((size_t(n)*width+255)/256),256,0,d.stream>>>(n,width,final_x,bank,S);CU(cudaGetLastError());};
    auto gram=[&](){BL(cublasDgemm(d.blas,CUBLAS_OP_T,CUBLAS_OP_N,width,width,n,&one,S,n,AS,n,&zero,G,width));
        symmetric<<<(width*width+255)/256,256,0,d.stream>>>(width,G);CU(cudaGetLastError());};
    // All library initialization and the first call are reported separately.
    for(int rank:{8,16}){
        Product target_product(d,target,rank,Z,W);
        Json entry={{"rank",rank},{"component_times",Json::object()},{"pipeline_samples",Json::array()},
                    {"apply_samples",Json::array()}};
        auto eigenvectors=[&](){SO(cusolverDnDsyevd(d.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,width,G,width,eigen,workspace,workspace_size,info));
            check_info("syevd");auto ev=d.host(eigen,width);
            need(std::isfinite(ev.back())&&ev.back()>0&&ev[width-rank]>1e-10*ev.back(),"Insufficient independent POD modes");
            select_modes<<<(width*rank+255)/256,256,0,d.stream>>>(width,rank,G,U);CU(cudaGetLastError());
            BL(cublasDgemm(d.blas,CUBLAS_OP_N,CUBLAS_OP_N,n,rank,width,&one,S,n,U,width,&zero,Z,n));};
        auto qr=[&](){SO(cusolverDnDgeqrf(d.solver,n,rank,Z,n,tau,workspace,workspace_size,info));check_info("geqrf");
            diagonals<<<1,32,0,d.stream>>>(n,rank,Z,diag);CU(cudaGetLastError());auto v=d.host(diag,rank);double low=1e300,high=0;
            for(double t:v){need(std::isfinite(t),"Nonfinite QR diagonal");low=std::min(low,std::abs(t));high=std::max(high,std::abs(t));}
            need(high>0&&low>1e-10*high,"QR rank deficiency; do not invent completion");
            SO(cusolverDnDorgqr(d.solver,n,rank,rank,Z,n,tau,workspace,workspace_size,info));check_info("orgqr");};
        auto factor=[&](){BL(cublasDgemm(d.blas,CUBLAS_OP_T,CUBLAS_OP_N,rank,rank,n,&one,Z,n,W,n,&zero,E,rank));
            symmetric<<<1,256,0,d.stream>>>(rank,E);CU(cudaGetLastError());
            SO(cusolverDnDpotrf(d.solver,CUBLAS_FILL_MODE_LOWER,rank,E,rank,workspace,workspace_size,info));check_info("potrf");};
        auto pipeline=[&](){capture();make_S();source_product.run();gram();eigenvectors();qr();target_product.run();factor();};
        entry["first_pipeline"]=timed(d,pipeline);
        entry["component_times"]["history_d2d"]=timed(d,capture);
        entry["component_times"]["S_final_minus_history"]=timed(d,make_S);
        entry["component_times"]["source_A_S"]=timed(d,[&](){source_product.run();});
        entry["component_times"]["source_S_t_A_S"]=timed(d,gram);
        save(output/("gram_r"+std::to_string(rank)+".bin"),d.host(G,width*width));
        entry["component_times"]["eigensolve_extract"]=timed(d,eigenvectors);
        entry["component_times"]["QR_and_rank_check"]=timed(d,qr);
        entry["component_times"]["target_A_Z"]=timed(d,[&](){target_product.run();});
        entry["component_times"]["target_E_factor_check"]=timed(d,factor);
        for(int sample=0;sample<3;++sample)entry["pipeline_samples"].push_back(timed(d,pipeline));
        save(output/("Z_r"+std::to_string(rank)+".bin"),d.host(Z,size_t(n)*rank));
        save(output/("W_r"+std::to_string(rank)+".bin"),d.host(W,size_t(n)*rank));
        save(output/("factor_r"+std::to_string(rank)+".bin"),d.host(E,rank*rank));
        save(output/("eigen_r"+std::to_string(rank)+".bin"),d.host(eigen,width));
        if(rank==8)save(output/"history_bank.bin",d.host(bank,size_t(n)*width));
        auto balanced=[&](const double* r){
            BL(cublasDgemv(d.blas,CUBLAS_OP_T,n,rank,&one,Z,n,r,1,&zero,a,1));
            tiny_solve<<<1,1,0,d.stream>>>(rank,E,a);
            CU(cudaMemcpyAsync(intermediate,r,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
            BL(cublasDgemv(d.blas,CUBLAS_OP_N,n,rank,&minus_one,W,n,a,1,&one,intermediate,1));
            mas.apply(intermediate,answer);
            BL(cublasDgemv(d.blas,CUBLAS_OP_T,n,rank,&one,W,n,answer,1,&zero,b,1));
            tiny_solve<<<1,1,0,d.stream>>>(rank,E,b);coefficients<<<1,32,0,d.stream>>>(rank,a,b);
            BL(cublasDgemv(d.blas,CUBLAS_OP_N,n,rank,&one,Z,n,b,1,&one,answer,1));CU(cudaGetLastError());};
        auto expected_mas=load<double>(folder,m,"expected_MAS"),expected_balanced=load<double>(folder,m,"expected_B_"+std::to_string(rank));
        Json checks=Json::array();
        for(int probe=0;probe<3;++probe){mas.apply(input+size_t(n)*probe,answer);auto mv=d.host(answer,n);
            balanced(input+size_t(n)*probe);auto bv=d.host(answer,n);
            std::vector<double> em(expected_mas.begin()+size_t(n)*probe,expected_mas.begin()+size_t(n)*(probe+1));
            std::vector<double> eb(expected_balanced.begin()+size_t(n)*probe,expected_balanced.begin()+size_t(n)*(probe+1));
            double me=relative(mv,em),be=relative(bv,eb);need(me<5e-10&&be<1e-6,"GPU action failed independent CPU calibration");
            checks.push_back({{"probe",probe},{"MAS_relative",me},{"balanced_relative",be}});
            save(output/("balanced_r"+std::to_string(rank)+"_p"+std::to_string(probe)+".bin"),bv);
        }
        entry["action_checks"]=checks;
        auto base=[&](){mas.apply(input,answer);};auto augmented=[&](){balanced(input);};
        // Warm every call before stream capture; capture is charged separately.
        for(int i=0;i<10;++i){base();augmented();}CU(cudaStreamSynchronize(d.stream));
        Replay base_graph(d,base),augmented_graph(d,augmented);
        entry["graph_capture_ms"]={{"base",base_graph.capture_ms},{"balanced",augmented_graph.capture_ms}};
        for(int i=0;i<10;++i){base_graph.run();augmented_graph.run();}CU(cudaStreamSynchronize(d.stream));
        augmented_graph.run();auto graph_answer=d.host(answer,n);augmented();auto host_answer=d.host(answer,n);
        need(relative(graph_answer,host_answer)<1e-12,"Graph replay changed balanced action");
        for(const std::string mode:{"graph","host"})for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order){
            bool augmented_arm=((order+pair)%2)!=0;
            auto time=timed(d,[&](){for(int i=0;i<200;++i){if(mode=="graph"){
                if(augmented_arm)augmented_graph.run();else base_graph.run();
            }else {if(augmented_arm)augmented();else base();}}});
            entry["apply_samples"].push_back({{"mode",mode},{"pair",pair},{"order",order},
                {"arm",augmented_arm?"balanced":"MAS_reference"},{"repeats",200},
                {"GPU_us_per_apply",time["gpu_ms"].get<double>()*5.},{"wall_us_per_apply",time["wall_ms"].get<double>()*5.}});
        }
        result["ranks"].push_back(entry);std::cout<<"rank "<<rank<<" validated"<<std::endl;
    }
    result["explicit_device_bytes"]=d.bytes;result["passed"]=true;result["performance_certified"]=false;
    need(!fs::exists(output/"result.json"),"Result exists");std::ofstream(output/"result.json")<<result.dump(2);
}
int main(int argc,char** argv){try{need(argc==3,"Usage: p5_pod_cost INPUT_DIR EMPTY_OUTPUT_DIR");run(fs::path(argv[1]),fs::path(argv[2]));return 0;}
    catch(const std::exception& error){std::cerr<<error.what()<<std::endl;
        if(argc==3&&!fs::exists(fs::path(argv[2])/"failure.json"))std::ofstream(fs::path(argv[2])/"failure.json")<<Json{{"passed",false},{"error",error.what()}}.dump(2);return 1;}}
