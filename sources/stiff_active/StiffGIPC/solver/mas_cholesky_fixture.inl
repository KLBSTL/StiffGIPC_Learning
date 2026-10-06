#include <cmath>

int MASPreconditioner::cholesky_fixture(const char* raw_prefix,const char* raw_output)
{
    if(!raw_output)throw std::runtime_error("Cholesky fixture output required");
    const std::string prefix=raw_prefix,out=raw_output;
    if(std::filesystem::exists(out))throw std::runtime_error("Preserve Cholesky fixture");std::filesystem::create_directories(out);
    gipc::Json meta;std::ifstream(prefix+"_meta.json")>>meta;gipc::Json local;
    for(const auto& p:meta["local_preconditioners"])if(p["kind"]=="MAS_full_owned_buffers")local=p;
    MASPreconditioner m;m.totalNodes=local["dimensions"][0];m.totalMapNodes=local["dimensions"][1];m.levelnum=local["dimensions"][2];
    m.totalNumberClusters=local["dimensions"][4];m.precision_initialized=true;
    const int offset=3*local["offset"].get<int>(),n=3*m.totalNodes,count=m.totalNumberClusters/BANKSIZE;
    auto load=[&](auto& buffer,const char* name)
    {using T=std::remove_pointer_t<decltype(buffer.data())>;std::ifstream f(prefix+"_mas_"+name+".bin",std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixture input");size_t bytes=f.tellg();std::vector<T> h(bytes/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(h.data()),bytes);if(!f||bytes%sizeof(T))throw std::runtime_error("fixture size");buffer.resize_discard(h.size());CUDA_SAFE_CALL(cudaMemcpy(buffer.data(),h.data(),bytes,cudaMemcpyHostToDevice));};
#define LOAD_CHOL(name) load(m.name,#name)
    LOAD_CHOL(d_inverseMatMas);LOAD_CHOL(d_coarseTable);LOAD_CHOL(d_partId_map_real);LOAD_CHOL(d_real_map_partId);
    LOAD_CHOL(d_goingNext);LOAD_CHOL(d_prefixOriginal);LOAD_CHOL(d_fineConnectMask);
#undef LOAD_CHOL
    m.d_multiLevelR.resize_discard(m.totalNumberClusters);m.d_multiLevelZ.resize_discard(m.totalNumberClusters);
    m.d_multiLevelR64.resize_discard(m.totalNumberClusters);m.d_multiLevelZ64.resize_discard(m.totalNumberClusters);
    m.d_precondMatMas.resize_discard(count);
    __inverse6_P96x96<<<(m.totalNumberClusters*3+95)/96,96>>>(m.d_precondMatMas.data(),m.d_inverseMatMas.data(),m.totalNumberClusters*3);
    m.prepare_cholesky();
    auto save=[&](const std::string& name,const auto& buffer)
    {using T=std::remove_pointer_t<decltype(buffer.data())>;std::vector<std::remove_const_t<T>> h(buffer.size());CUDA_SAFE_CALL(cudaMemcpy(h.data(),buffer.data(),h.size()*sizeof(T),cudaMemcpyDeviceToHost));std::ofstream f(out+"/"+name,std::ios::binary);f.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(T));};
    save("factors.bin",m.d_cholesky);
    if(m.d_factor_inverse.size())save("inverse_factor.bin",m.d_factor_inverse);
    cudatool::DeviceBuffer<double> input(n),output(n);
    auto read_output=[&](){std::vector<double> h(n);CUDA_SAFE_CALL(cudaMemcpy(h.data(),output.data(),n*8,cudaMemcpyDeviceToHost));return h;};
    gipc::Json result={{"passed",true},{"graph_bitwise",true},
        {"factor_action",gipc::mas_factor_action_mode()},
        {"factor_inverse_prepared",m.d_factor_inverse.size()!=0},
        {"block_width",CHOL_N},{"phases",gipc::Json::array()}};
    std::vector<std::uintptr_t> off_key;
    for(int phase=0;phase<3;++phase)
    {
        m.cholesky=phase==1;m.wide_apply=m.cholesky;
        auto key=m.graph_signature();if(phase==0)off_key=key;
        if((phase==1 && key==off_key)||(phase==2 && key!=off_key))throw std::runtime_error("Cholesky mode signature");
        for(int k=0;k<10;++k)
        {
            std::vector<double> values(n);std::ifstream f(prefix+"_v"+std::to_string(k)+".bin",std::ios::binary);f.seekg(offset*8);f.read(reinterpret_cast<char*>(values.data()),n*8);if(!f)throw std::runtime_error("probe read");
            CUDA_SAFE_CALL(cudaMemcpy(input.data(),values.data(),n*8,cudaMemcpyHostToDevice));
            m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));auto first=read_output();
            save("p"+std::to_string(phase)+"_v"+std::to_string(k)+".bin",output);
            CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
            cudaGraph_t graph;cudaGraphExec_t exec;CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&graph));CUDA_SAFE_CALL(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            for(int repeat=0;repeat<3;++repeat)
            {CUDA_SAFE_CALL(cudaGraphLaunch(exec,cudaStreamPerThread));auto now=read_output();if(phase==1 && std::memcmp(first.data(),now.data(),n*8)!=0)result["graph_bitwise"]=false;}
            CUDA_SAFE_CALL(cudaGraphExecDestroy(exec));CUDA_SAFE_CALL(cudaGraphDestroy(graph));
        }
        result["phases"].push_back({{"phase",phase},{"cholesky",m.cholesky}});
    }
    m.cholesky=true;m.wide_apply=true;m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));auto before=read_output();auto old_key=m.graph_signature();
    m.d_cholesky.reserve(m.d_cholesky.capacity()+1);m.d_multiLevelR64.reserve(m.d_multiLevelR64.capacity()+1);m.d_multiLevelZ64.reserve(m.d_multiLevelZ64.capacity()+1);m.d_restriction_starts.reserve(m.d_restriction_starts.capacity()+1);m.d_restriction_nodes.reserve(m.d_restriction_nodes.capacity()+1);
    m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));auto after=read_output();
    result["growth_signature_changed"]=old_key!=m.graph_signature();result["growth_bitwise"]=std::memcmp(before.data(),after.data(),n*8)==0;
    result["off_signature_restored"]=true;

    // Grow B alone so a change in some unrelated buffer cannot hide a missing
    // B pointer in the Graph signature. Rebuild capture after the growth.
    result["factor_inverse_growth_checked"]=m.d_factor_inverse.size()!=0;
    bool inverse_growth_passed=true;
    if(m.d_factor_inverse.size())
    {
        const auto before_growth=read_output();const auto inverse_key=m.graph_signature();
        m.d_factor_inverse.reserve(m.d_factor_inverse.capacity()+1);
        m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
        const auto after_growth=read_output();
        const bool signature_changed=inverse_key!=m.graph_signature();
        const bool same=std::memcmp(before_growth.data(),after_growth.data(),n*8)==0;
        CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
        m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
        cudaGraph_t graph;cudaGraphExec_t exec;
        CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        CUDA_SAFE_CALL(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        bool replay_same=true;
        for(int repeat=0;repeat<3;++repeat)
        {
            CUDA_SAFE_CALL(cudaGraphLaunch(exec,cudaStreamPerThread));const auto values=read_output();
            replay_same&=std::memcmp(before_growth.data(),values.data(),n*8)==0;
        }
        CUDA_SAFE_CALL(cudaGraphExecDestroy(exec));CUDA_SAFE_CALL(cudaGraphDestroy(graph));
        result["factor_inverse_growth_signature_changed"]=signature_changed;
        result["factor_inverse_growth_bitwise"]=same;
        result["factor_inverse_growth_graph_bitwise"]=replay_same;
        inverse_growth_passed=signature_changed&&same&&replay_same;
    }

    // A zero RHS must remain exactly zero for both launches and Graph replay.
    CUDA_SAFE_CALL(cudaMemset(input.data(),0,n*sizeof(double)));
    m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
    const auto zero_host=read_output();
    auto all_zero=[](const std::vector<double>& values)
    {return std::all_of(values.begin(),values.end(),[](double value){return value==0.0;});};
    result["zero_rhs_host"]=all_zero(zero_host);
    CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
    m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
    cudaGraph_t zero_graph;cudaGraphExec_t zero_exec;
    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&zero_graph));
    CUDA_SAFE_CALL(cudaGraphInstantiate(&zero_exec,zero_graph,nullptr,nullptr,0));
    bool zero_replay=true;
    for(int repeat=0;repeat<3;++repeat)
    {
        CUDA_SAFE_CALL(cudaGraphLaunch(zero_exec,cudaStreamPerThread));const auto values=read_output();
        zero_replay&=all_zero(values)&&std::memcmp(zero_host.data(),values.data(),n*8)==0;
    }
    CUDA_SAFE_CALL(cudaGraphExecDestroy(zero_exec));CUDA_SAFE_CALL(cudaGraphDestroy(zero_graph));
    result["zero_rhs_graph"]=zero_replay;

    // Invalid inputs are tested using private one-block buffers. Every owned
    // buffer in m is hashed before and after to prove the fixture does not
    // replace valid factors or leave fault data for subsequent work.
    auto owned_digest=[&]()
    {
        gipc::Json digest=gipc::Json::object();
        m.diagnostic_buffers([&](const char* name,const void* data,size_t length,size_t item_bytes)
        {
            const size_t bytes=length*item_bytes;std::vector<unsigned char> host(bytes);
            if(bytes)CUDA_SAFE_CALL(cudaMemcpy(host.data(),data,bytes,cudaMemcpyDeviceToHost));
            std::uint64_t hash=14695981039346656037ull;
            for(auto value:host){hash^=value;hash*=1099511628211ull;}
            digest[name]={{"bytes",bytes},{"fnv1a64",hash}};
        });
        return digest;
    };
    const auto fault_before=owned_digest();
    {
        cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymT> bad_matrix(1);
        cudatool::DeviceBuffer<double> private_factor(CHOL_N*CHOL_N),private_inverse(CHOL_N*CHOL_N);
        cudatool::DeviceBuffer<int> private_status(1);
        __GEIGEN__::MasMatrixSymT host_matrix;
        CUDA_SAFE_CALL(cudaMemcpy(&host_matrix,m.d_inverseMatMas.data(),sizeof(host_matrix),cudaMemcpyDeviceToHost));
        host_matrix.M[0](0,0)=-1.0;
        CUDA_SAFE_CALL(cudaMemcpy(bad_matrix.data(),&host_matrix,sizeof(host_matrix),cudaMemcpyHostToDevice));
        symmetric_cholesky<<<1,CHOL_N>>>(bad_matrix.data(),private_factor.data(),private_status.data());
        int pivot_status=0;
        CUDA_SAFE_CALL(cudaMemcpy(&pivot_status,private_status.data(),sizeof(int),cudaMemcpyDeviceToHost));
        result["invalid_pivot_status"]=pivot_status;
        result["invalid_pivot_detected"]=pivot_status==1;

        std::vector<double> tiny_diagonal(CHOL_N*CHOL_N,0.0);
        for(int i=0;i<CHOL_N;++i)tiny_diagonal[i*CHOL_N+i]=1.0;
        tiny_diagonal[0]=std::numeric_limits<double>::denorm_min();
        CUDA_SAFE_CALL(cudaMemcpy(private_factor.data(),tiny_diagonal.data(),tiny_diagonal.size()*sizeof(double),cudaMemcpyHostToDevice));
        invert_cholesky_factor<<<1,CHOL_N>>>(private_factor.data(),private_inverse.data(),private_status.data());
        int inverse_status=0;
        CUDA_SAFE_CALL(cudaMemcpy(&inverse_status,private_status.data(),sizeof(int),cudaMemcpyDeviceToHost));
        std::vector<double> observed(CHOL_N*CHOL_N);
        CUDA_SAFE_CALL(cudaMemcpy(observed.data(),private_inverse.data(),observed.size()*sizeof(double),cudaMemcpyDeviceToHost));
        bool observed_nonfinite=false;
        for(double value:observed)observed_nonfinite|=!std::isfinite(value);
        result["nonfinite_inverse_status"]=inverse_status;
        result["nonfinite_inverse_detected"]=inverse_status>0&&observed_nonfinite;
    }
    result["fault_probe_main_buffers_unchanged"]=fault_before==owned_digest();
    result["passed"]=result["graph_bitwise"].get<bool>()
        &&result["growth_signature_changed"].get<bool>()&&result["growth_bitwise"].get<bool>()
        &&inverse_growth_passed&&result["zero_rhs_host"].get<bool>()&&zero_replay
        &&result["invalid_pivot_detected"].get<bool>()&&result["nonfinite_inverse_detected"].get<bool>()
        &&result["fault_probe_main_buffers_unchanged"].get<bool>();
    std::ofstream(out+"/fixture.json")<<result.dump(2);return result["passed"].get<bool>()?0:1;
}
