// Standalone research diagnostic; native kernels retain upstream MPL provenance.
#include <solver/MASPreconditioner.cuh>
#include <linear_system/utils/spmv.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include "native_kernels.cuh"
#include "cholesky.cuh"
using Json=nlohmann::json;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buffer
{
    T* p=nullptr;size_t n=0;
    explicit Buffer(size_t count):n(count){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    Buffer(const Buffer&)=delete;
    void upload(const std::vector<T>& v){if(v.size()!=n)throw std::runtime_error("buffer size");if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    void zero(){if(n)check(cudaMemset(p,0,n*sizeof(T)));}
};
template<class T> std::vector<T> read_file(const std::string& name)
{
    std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("cannot read "+name);
    auto bytes=f.tellg();if(bytes%sizeof(T))throw std::runtime_error("unaligned "+name);
    std::vector<T> v(static_cast<size_t>(bytes)/sizeof(T));f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()),bytes);if(!f)throw std::runtime_error("short read "+name);return v;
}
void save(const std::string& path,const std::vector<double>& v)
{std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*8);if(!f)throw std::runtime_error("write failed");}
double host_dot(const std::vector<double>& a,const std::vector<double>& b)
{long double v=0;for(size_t i=0;i<a.size();++i)v+=static_cast<long double>(a[i])*b[i];return static_cast<double>(v);}
Json diff(const std::vector<double>& a,const std::vector<double>& b)
{
    if(a.size()!=b.size())throw std::runtime_error("comparison dimensions");
    long double d=0;double maximum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){double v=a[i]-b[i];d+=static_cast<long double>(v)*v;maximum=std::max(maximum,std::abs(v));unequal+=a[i]!=b[i];}
    return {{"relative",std::sqrt(static_cast<double>(d)/std::max(host_dot(b,b),1e-300))},{"max_abs",maximum},{"unequal",unequal}};
}
__global__ void abd_apply(const double* r,double* z,const Eigen::Matrix<double,12,12>* m,int bodies,int offset)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=bodies)return;
    Eigen::Map<Eigen::Matrix<double,12,1>>(z+offset+i*12)=
        m[i]*Eigen::Map<const Eigen::Matrix<double,12,1>>(r+offset+i*12);
}

int factor_fixture(const std::string& out)
{
    std::vector<__GEIGEN__::MasMatrixSymT> matrices(4);
    for(auto& m:matrices)for(auto& v:m.M)v.setZero();
    auto set=[&](int b,int r,int c,double x){int br=r/3,bc=c/3;matrices[b].M[BANKSIZE*br-br*(br+1)/2+bc](r%3,c%3)=x;};
    for(int i=0;i<CHOL_N;++i){set(0,i,i,1);set(1,i,i,2);set(2,i,i,1);set(3,i,i,1);}
    for(int i=0;i+1<CHOL_N;++i)
    {set(1,i,i+1,i/3==(i+1)/3?.3:.25);if(i/3==(i+1)/3)set(1,i+1,i,.2);}
    set(0,CHOL_N-1,CHOL_N-1,0); // Native padded-diagonal convention.
    set(2,0,0,-1);set(3,0,0,std::numeric_limits<double>::quiet_NaN());
    Buffer<__GEIGEN__::MasMatrixSymT> input(4);input.upload(matrices);
    Buffer<double> factors(4*CHOL_N*CHOL_N);Buffer<int> status(4);
    symmetric_cholesky<<<4,CHOL_N>>>(input.p,factors.p,status.p);
    auto codes=status.read();if(codes!=std::vector<int>({0,0,1,1}))throw std::runtime_error("factor fixture pivot classification");
    Buffer<Eigen::Vector3d> rhs(2*BANKSIZE);Buffer<double3> output(2*BANKSIZE);
    std::vector<Eigen::Vector3d> values(2*BANKSIZE);
    for(int i=0;i<2*CHOL_N;++i)reinterpret_cast<double*>(values.data())[i]=1./(i+1);
    rhs.upload(values);cholesky_action<<<2,CHOL_N>>>(factors.p,rhs.p,output.p);auto first=output.read();
    const double* x=reinterpret_cast<const double*>(first.data());double error=0;
    for(int i=0;i<2*CHOL_N;++i)
    {
        double ax=x[i];if(i>=CHOL_N){ax=2*x[i];if(i>CHOL_N)ax+=.25*x[i-1];if(i+1<2*CHOL_N)ax+=.25*x[i+1];}
        error=std::max(error,std::abs(ax-1./(i+1)));
    }
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));cholesky_action<<<2,CHOL_N>>>(factors.p,rhs.p,output.p);
    cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    bool identical=true;for(int i=0;i<3;++i){check(cudaGraphLaunch(exec,cudaStreamPerThread));auto current=output.read();identical&=std::memcmp(first.data(),current.data(),first.size()*sizeof(double3))==0;}
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    Json result={{"pivot_codes",codes},{"max_abs_residual",error},{"graph_bitwise",identical},{"passed",error<1e-12&&identical}};
    if(std::filesystem::exists(out))throw std::runtime_error("preserve fixture result");std::ofstream(out)<<result.dump(2);
    return result["passed"].get<bool>()?0:1;
}

int main(int argc,char** argv)
try
{
    if(argc==3 && std::string(argv[1])=="--selftest")return factor_fixture(argv[2]);
    if(argc!=4 && argc!=7)throw std::runtime_error("mas_replay snapshot_prefix output_directory mode[5..7] [tolerance max_iter_ratio repeats]");
    const std::string prefix=argv[1],out=argv[2];const int mode=std::stoi(argv[3]);
    if(mode<5||mode>7)throw std::runtime_error("v42 modes: 5 old inverse64, 6 Cholesky atomic R, 7 Cholesky deterministic R");
    std::vector<double> tolerances={1e-4,1e-14};
    double max_iter_ratio=.3;int repeats=2;
    if(argc==7){tolerances={std::stod(argv[4])};max_iter_ratio=std::stod(argv[5]);repeats=std::stoi(argv[6]);}
    if(!(max_iter_ratio>0 && max_iter_ratio<=1) || repeats<1 || repeats>3)throw std::runtime_error("diagnostic bounds");
    for(double tol:tolerances)if(!(tol>0 && tol<1))throw std::runtime_error("tolerance bounds");
    std::filesystem::create_directories(out);
    if(std::filesystem::exists(out+"/result.json"))throw std::runtime_error("preserve existing results");
    Json meta;std::ifstream(prefix+"_meta.json")>>meta;
    Json local,abd;for(auto item:meta["local_preconditioners"]){if(item["kind"]=="MAS_full_owned_buffers")local=item;else if(item["kind"]=="abd")abd=item;}
    if(local.is_null())throw std::runtime_error("complete MAS required");
    const int n=meta["dofs"],nb=meta["blocks"],nodes=local["dimensions"][0],mapped=local["dimensions"][1];
    const int levels=local["dimensions"][2],clusters=local["dimensions"][4],offset=3*local["offset"].get<int>();
    if(offset+3*nodes!=n)throw std::runtime_error("MAS range");
    auto rhs=read_file<double>(prefix+"_rhs.bin");Buffer<double> b(n),x(n),r(n),z(n),p(n),ap(n),diagnostic_ax(n);b.upload(rhs);
    auto rows=read_file<int>(prefix+"_rows.bin"),cols=read_file<int>(prefix+"_cols.bin");
    auto mats=read_file<Eigen::Matrix3d>(prefix+"_values.bin");
    if(rhs.size()!=n||rows.size()!=nb||cols.size()!=nb||mats.size()!=nb)throw std::runtime_error("matrix dimensions");
    Buffer<int> dr(nb),dc(nb);Buffer<Eigen::Matrix3d> dv(nb);dr.upload(rows);dc.upload(cols);dv.upload(mats);
#define LOAD(TYPE,NAME) auto h_##NAME=read_file<TYPE>(prefix+"_mas_"#NAME".bin"); Buffer<TYPE> NAME(h_##NAME.size()); NAME.upload(h_##NAME)
    LOAD(int,d_goingNext);LOAD(int,d_prefixOriginal);LOAD(unsigned int,d_fineConnectMask);
    LOAD(int,d_partId_map_real);LOAD(int,d_real_map_partId);LOAD(__GEIGEN__::itable,d_coarseTable);
    LOAD(__GEIGEN__::MasMatrixSymT,d_inverseMatMas);LOAD(__GEIGEN__::MasMatrixSymf,d_precondMatMas);
#undef LOAD
    Buffer<__GEIGEN__::MasMatrixSymT> inverse64(clusters/BANKSIZE);
    Buffer<__GEIGEN__::MasMatrixSymf> regenerated32(clusters/BANKSIZE);
    // Exact native geometry: 96 threads, two BANKSIZE=16 blocks per CTA.
    __inverse6_P96x96<<<(clusters*3+95)/96,96>>>(inverse64.p,d_inverseMatMas.p,clusters*3);
    __inverse6_P96x96<<<(clusters*3+95)/96,96>>>(regenerated32.p,d_inverseMatMas.p,clusters*3);
    check(cudaDeviceSynchronize());
    auto regenerated=regenerated32.read();
    bool inverse32_identical=regenerated.size()==h_d_precondMatMas.size() && std::memcmp(regenerated.data(),h_d_precondMatMas.data(),regenerated.size()*sizeof(regenerated[0]))==0;
    const int local_blocks=clusters/BANKSIZE;
    Buffer<double> factors(mode>=6?local_blocks*CHOL_N*CHOL_N:0);Buffer<int> factor_status(mode>=6?local_blocks:0);
    if(mode>=6)
    {
        symmetric_cholesky<<<local_blocks,CHOL_N>>>(d_inverseMatMas.p,factors.p,factor_status.p);
        auto status=factor_status.read();Json failed=Json::array();
        for(int i=0;i<local_blocks;++i)if(status[i])failed.push_back({{"block",i},{"pivot",status[i]-1}});
        std::ofstream(out+"/factor_status.json")<<Json({{"blocks",local_blocks},{"failed",failed}}).dump(2);
        if(!failed.empty())throw std::runtime_error("Nonpositive/nonfinite Cholesky pivot; no regularization applied");
        save(out+"/factors.bin",factors.read());
    }
    std::vector<std::vector<int>> contributors(clusters);
    for(int i=0;i<mapped;++i)if(h_d_partId_map_real[i]>=0)
    {if(h_d_partId_map_real[i]>=nodes)throw std::runtime_error("fine map range");contributors[i].push_back(h_d_partId_map_real[i]);}
    const int* coarse=reinterpret_cast<const int*>(h_d_coarseTable.data());
    for(int i=0;i<nodes;++i)for(int level=0;level<levels-1;++level)
    {
        int cluster=coarse[i*6+level];if(cluster<0||cluster>=clusters)throw std::runtime_error("coarse map range");
        contributors[cluster].push_back(i);
    }
    std::vector<int> starts(clusters+1),indices;
    for(int i=0;i<clusters;++i){std::sort(contributors[i].begin(),contributors[i].end());indices.insert(indices.end(),contributors[i].begin(),contributors[i].end());starts[i+1]=indices.size();}
    Buffer<int> restriction_starts(starts.size()),restriction_nodes(indices.size());restriction_starts.upload(starts);restriction_nodes.upload(indices);
    Buffer<Eigen::Vector3f> lr32(clusters);Buffer<Eigen::Vector3d> lr64(clusters);
    Buffer<float3> lz32(clusters);Buffer<double3> lz64(clusters);
    using Mat12=Eigen::Matrix<double,12,12>;
    std::vector<Mat12> abd_host;int abd_offset=0;
    if(!abd.is_null()){abd_host=read_file<Mat12>(prefix+"_"+abd["buffer"].get<std::string>()+".bin");abd_offset=3*abd["offset"].get<int>();}
    Buffer<Mat12> abd_matrix(abd_host.size());abd_matrix.upload(abd_host);
    auto apply=[&](const double* input,double* output)
    {
        const bool wide_r=mode>=4;
        if(mode<=1)lz32.zero();else lz64.zero();
        if(mode==7)
        {
            deterministic_restrict<<<(clusters*3+255)/256,256>>>(input+offset,lr64.p,restriction_starts.p,restriction_nodes.p,clusters);
        }
        else if(wide_r)
        {
            check(cudaMemset(lr64.p+mapped,0,(clusters-mapped)*sizeof(Eigen::Vector3d)));
            __buildMultiLevelR_optimized_new<<<(mapped+255)/256,256>>>((const double3*)(input+offset),lr64.p,d_goingNext.p,d_prefixOriginal.p,d_fineConnectMask.p,d_partId_map_real.p,levels,mapped);
        }
        else
        {
            check(cudaMemset(lr32.p+mapped,0,(clusters-mapped)*sizeof(Eigen::Vector3f)));
            __buildMultiLevelR_optimized_new<<<(mapped+255)/256,256>>>((const double3*)(input+offset),lr32.p,d_goingNext.p,d_prefixOriginal.p,d_fineConnectMask.p,d_partId_map_real.p,levels,mapped);
        }
        const int count=clusters*BANKSIZE,blocks=(count+BANKSIZE*BANKSIZE-1)/(BANKSIZE*BANKSIZE);
        if(mode<=1)_schwarzLocalXSym6<__GEIGEN__::MasMatrixSymf,float,float><<<blocks,BANKSIZE*BANKSIZE>>>(mode?regenerated32.p:d_precondMatMas.p,lr32.p,lz32.p,count);
        if(mode==2)_schwarzLocalXSym6<__GEIGEN__::MasMatrixSymf,float,double><<<blocks,BANKSIZE*BANKSIZE>>>(d_precondMatMas.p,lr32.p,lz64.p,count);
        if(mode==3)_schwarzLocalXSym6<__GEIGEN__::MasMatrixSymT,float,double><<<blocks,BANKSIZE*BANKSIZE>>>(inverse64.p,lr32.p,lz64.p,count);
        if(mode==4)_schwarzLocalXSym6<__GEIGEN__::MasMatrixSymf,double,double><<<blocks,BANKSIZE*BANKSIZE>>>(d_precondMatMas.p,lr64.p,lz64.p,count);
        if(mode==5)_schwarzLocalXSym6<__GEIGEN__::MasMatrixSymT,double,double><<<blocks,BANKSIZE*BANKSIZE>>>(inverse64.p,lr64.p,lz64.p,count);
        if(mode>=6)cholesky_action<<<local_blocks,CHOL_N>>>(factors.p,lr64.p,lz64.p);
        if(mode<=1)__collectFinalZ_new<<<(nodes+255)/256,256>>>((double3*)(output+offset),lz32.p,d_coarseTable.p,d_real_map_partId.p,levels,nodes);
        else __collectFinalZ_new<<<(nodes+255)/256,256>>>((double3*)(output+offset),lz64.p,d_coarseTable.p,d_real_map_partId.p,levels,nodes);
        if(!abd_host.empty())abd_apply<<<(abd_host.size()*12+255)/256,256>>>(input,output,abd_matrix.p,abd_host.size(),abd_offset);
    };
    gipc::Spmv spmv;
    auto multiply=[&](const double* input,double* output){spmv.warp_reduce_sym_spmv(1,dv.p,dr.p,dc.p,nb,{input,n},0,{output,n});};
    Buffer<double> partials((n+255)/256),scalar(1);
    size_t scratch_size=0;check(cub::DeviceReduce::Sum(nullptr,scratch_size,partials.p,scalar.p,partials.n));Buffer<unsigned char> scratch(scratch_size);
    auto dot=[&](const double* a,const double* b){PCG_vdv_Reduction<<<(n+255)/256,256>>>(partials.p,a,b,n);check(cub::DeviceReduce::Sum(scratch.p,scratch_size,partials.p,scalar.p,partials.n));return scalar.read()[0];};
    const char* names[]={"captured_native","regenerated_native","captured_wide","inverse64_r32","captured_r64","full64","cholesky_atomic_r","cholesky_deterministic_r"};
    Json result={{"mode",names[mode]},{"dofs",n},{"max_iter_ratio",max_iter_ratio},{"snapshot_prefix",prefix},{"native_inverse_recomputed_bitwise",inverse32_identical},
        {"scope","Fixed system host-loop replay, no simulation or speed claim"},{"operators",Json::array()},{"solves",Json::array()},
        {"native_inverse_snapshot_available",!h_d_precondMatMas.empty()},{"symmetric_factor_input",mode>=6},{"deterministic_restriction",mode==7},
        {"operator_threshold",1e-10},{"operator_repeat_passed",true}};
    std::vector<std::vector<double>> inputs,outputs;
    for(int k=0;k<10;++k)
    {
        auto input=read_file<double>(prefix+"_v"+std::to_string(k)+".bin");r.upload(input);apply(r.p,z.p);auto value=z.read();
        auto reference=read_file<double>(prefix+"_Mv"+std::to_string(k)+".bin");
        Json rec={{"index",k},{"versus_v37",diff(value,reference)},{"v_Mv",host_dot(input,value)},{"repeats",Json::array()}};
        for(int i=0;i<3;++i){apply(r.p,z.p);rec["repeats"].push_back(diff(z.read(),value));}
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));apply(r.p,z.p);
        cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));rec["graph_repeats"]=Json::array();
        for(int i=0;i<3;++i){check(cudaGraphLaunch(exec,cudaStreamPerThread));rec["graph_repeats"].push_back(diff(z.read(),value));}
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
        for(const char* field:{"repeats","graph_repeats"})for(const auto& d:rec[field])
            if(!std::isfinite(d["relative"].get<double>()) || d["relative"].get<double>()>1e-10)result["operator_repeat_passed"]=false;
        save(out+"/Mv"+std::to_string(k)+".bin",value);inputs.push_back(input);outputs.push_back(value);result["operators"].push_back(rec);
    }
    result["bilinear"]=Json::array();
    for(int k=0;k<10;k+=2)
    {
        double a=host_dot(inputs[k],outputs[k+1]),b=host_dot(inputs[k+1],outputs[k]);
        double scale=std::sqrt(host_dot(inputs[k],inputs[k])*host_dot(outputs[k+1],outputs[k+1]))+std::sqrt(host_dot(inputs[k+1],inputs[k+1])*host_dot(outputs[k],outputs[k]));
        result["bilinear"].push_back(std::abs(a-b)/std::max(scale,1e-300));
    }
    for(double tol:tolerances)for(int repeat=0;repeat<repeats;++repeat)
    {
        x.zero();r.upload(rhs);apply(r.p,z.p);double rho=dot(r.p,z.p),rho0=rho;check(cudaMemcpy(p.p,z.p,n*8,cudaMemcpyDeviceToDevice));
        const int limit=static_cast<int>(max_iter_ratio*n);int iteration=0;std::string reason="iteration_limit";
        Json history=Json::array();double min_pap=INFINITY,min_rho=rho;
        auto sample=[&](int k,double pap)
        {
            multiply(x.p,diagnostic_ax.p);auto ax=diagnostic_ax.read(),recursive=r.read();
            long double true_rr=0,rec_rr=0,drift=0;
            for(int i=0;i<n;++i){long double residual=rhs[i]-ax[i],error=recursive[i]-residual;
                true_rr+=residual*residual;rec_rr+=static_cast<long double>(recursive[i])*recursive[i];drift+=error*error;}
            history.push_back({{"iteration",k},{"rho_before_update",rho},{"rho_ratio_before_update",rho/rho0},{"pAp",pap},
                {"true_residual",std::sqrt(static_cast<double>(true_rr)/host_dot(rhs,rhs))},
                {"recursive_residual",std::sqrt(static_cast<double>(rec_rr)/host_dot(rhs,rhs))},
                {"residual_drift_over_b",std::sqrt(static_cast<double>(drift)/host_dot(rhs,rhs))}});
        };
        auto begin=std::chrono::steady_clock::now();
        for(iteration=1;iteration<limit;++iteration)
        {
            multiply(p.p,ap.p);double pap=dot(p.p,ap.p);
            min_pap=std::min(min_pap,pap);min_rho=std::min(min_rho,rho);
            if(!(pap>0)||!(rho>0)||!std::isfinite(pap)||!std::isfinite(rho)){reason="breakdown";break;}
            update_vector_dx_r<<<(n+255)/256,256>>>(x.p,r.p,p.p,ap.p,rho/pap,n);
            if(iteration==1 || iteration%250==0 || iteration==limit-1 || std::abs(rho)<=tol*rho0)sample(iteration,pap);
            // Match native previous-rho stopping AFTER the update.
            if(std::abs(rho)<=tol*rho0){reason="rho_stop";break;}
            apply(r.p,z.p);double next=dot(r.p,z.p);
            update_vector_c<<<(n+255)/256,256>>>(p.p,z.p,next/rho,n);rho=next;
        }
        check(cudaDeviceSynchronize());auto solution=x.read();multiply(x.p,ap.p);auto ax=ap.read();
        long double rr=0;for(int i=0;i<n;++i){long double v=rhs[i]-ax[i];rr+=v*v;}
        const std::string file="x_"+std::to_string(result["solves"].size())+".bin";save(out+"/"+file,solution);
        result["solves"].push_back({{"tolerance",tol},{"repeat",repeat+1},{"iterations",iteration},{"reason",reason},
            {"iteration_limit",limit},{"history",history},{"min_pAp",min_pap},{"min_rho",min_rho},
            {"rho_initial",rho0},{"rho_stop",rho},{"true_relative_residual",std::sqrt(static_cast<double>(rr)/host_dot(rhs,rhs))},
            {"diagnostic_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},{"file",file}});
        std::ofstream(out+"/result.json")<<result.dump(2);
    }
    std::cout<<names[mode]<<" complete\n";return 0;
}
catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
