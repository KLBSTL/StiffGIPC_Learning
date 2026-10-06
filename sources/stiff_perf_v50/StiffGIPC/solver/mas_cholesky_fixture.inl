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
    cudatool::DeviceBuffer<double> input(n),output(n);
    auto read_output=[&](){std::vector<double> h(n);CUDA_SAFE_CALL(cudaMemcpy(h.data(),output.data(),n*8,cudaMemcpyDeviceToHost));return h;};
    gipc::Json result={{"passed",true},{"graph_bitwise",true},{"phases",gipc::Json::array()}};std::vector<std::uintptr_t> off_key;
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
    result["off_signature_restored"]=true;result["passed"]=result["graph_bitwise"].get<bool>() && result["growth_signature_changed"].get<bool>() && result["growth_bitwise"].get<bool>();
    std::ofstream(out+"/fixture.json")<<result.dump(2);return result["passed"].get<bool>()?0:1;
}
