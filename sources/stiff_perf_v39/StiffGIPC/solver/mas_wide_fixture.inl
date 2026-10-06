// Diagnostic entry into the actual MAS class, without simulation initialization.
#include <fstream>
#include <filesystem>
#include <cstring>
#include <type_traits>
#include <gipc/utils/json.h>

int MASPreconditioner::replay_fixture(const char* raw_prefix,const char* raw_output)
{
    if(!raw_output)throw std::runtime_error("MAS fixture output required");
    const std::string prefix=raw_prefix,out=raw_output;
    std::filesystem::create_directories(out);
    if(std::filesystem::exists(out+"/fixture.json"))throw std::runtime_error("Preserve existing MAS fixture");
    gipc::Json meta;std::ifstream(prefix+"_meta.json")>>meta;
    gipc::Json local;for(const auto& p:meta["local_preconditioners"])if(p["kind"]=="MAS_full_owned_buffers")local=p;
    if(local.is_null())throw std::runtime_error("MAS fixture needs complete snapshot");
    MASPreconditioner m;
    m.totalNodes=local["dimensions"][0];m.totalMapNodes=local["dimensions"][1];m.levelnum=local["dimensions"][2];
    m.collision_node_Offset=local["dimensions"][3];m.totalNumberClusters=local["dimensions"][4];
    m.h_clevelSize={local["dimensions"][5],local["dimensions"][6]};m.neighborListSize=local["dimensions"][7];
    m.precision_initialized=true;
    const int offset=3*local["offset"].get<int>(),n=3*m.totalNodes;
    auto load=[&](auto& buffer,const char* name)
    {
        using T=std::remove_pointer_t<decltype(buffer.data())>;
        std::ifstream f(prefix+"_mas_"+name+".bin",std::ios::binary|std::ios::ate);
        if(!f)throw std::runtime_error("MAS fixture read failed");
        const size_t bytes=f.tellg();if(bytes%sizeof(T))throw std::runtime_error("MAS fixture size mismatch");
        std::vector<T> host(bytes/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(host.data()),bytes);
        if(!f)throw std::runtime_error("MAS fixture short read");
        buffer.resize_discard(host.size());if(bytes)CUDA_SAFE_CALL(cudaMemcpy(buffer.data(),host.data(),bytes,cudaMemcpyHostToDevice));
    };
#define MAS_LOAD(name) load(m.name,#name)
    MAS_LOAD(d_goingNext);MAS_LOAD(d_prefixOriginal);MAS_LOAD(d_fineConnectMask);
    MAS_LOAD(d_partId_map_real);MAS_LOAD(d_real_map_partId);MAS_LOAD(d_coarseTable);MAS_LOAD(d_precondMatMas);
#undef MAS_LOAD
    m.d_multiLevelR.resize_discard(m.totalNumberClusters);m.d_multiLevelZ.resize_discard(m.totalNumberClusters);
    m.d_multiLevelR64.resize_discard(m.totalNumberClusters);m.d_multiLevelZ64.resize_discard(m.totalNumberClusters);
    cudatool::DeviceBuffer<double> input(n),output(n);
    auto download=[&](){std::vector<double> h(n);CUDA_SAFE_CALL(cudaMemcpy(h.data(),output.data(),n*sizeof(double),cudaMemcpyDeviceToHost));return h;};
    auto difference=[](const auto& a,const auto& b)
    {long double d=0,s=0;for(size_t i=0;i<a.size();++i){long double v=a[i]-b[i];d+=v*v;s+=static_cast<long double>(b[i])*b[i];}return std::sqrt(static_cast<double>(d/std::max(s,1e-300L)));};
    auto save=[&](const std::string& name,const auto& values)
    {std::ofstream f(out+"/"+name,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));if(!f)throw std::runtime_error("MAS fixture write failed");};
    gipc::Json result={{"nodes",m.totalNodes},{"global_scalar_offset",offset},{"phases",gipc::Json::array()}};
    std::vector<std::uintptr_t> off_key,previous;
    for(int phase=0;phase<3;++phase)
    {
        m.wide_apply=phase==1;
        auto key=m.graph_signature();
        if(phase==0)off_key=key;
        if(phase>0 && key==previous)throw std::runtime_error("MAS precision missing from graph signature");
        previous=key;
        if(phase==2 && key!=off_key)throw std::runtime_error("MAS off signature did not restore");
        gipc::Json record={{"wide_apply",m.wide_apply},{"probes",gipc::Json::array()}};
        for(int k=0;k<10;++k)
        {
            std::ifstream f(prefix+"_v"+std::to_string(k)+".bin",std::ios::binary);
            std::vector<double> values(n);f.seekg(offset*sizeof(double));f.read(reinterpret_cast<char*>(values.data()),n*sizeof(double));
            if(!f)throw std::runtime_error("MAS probe short read");
            CUDA_SAFE_CALL(cudaMemcpy(input.data(),values.data(),n*sizeof(double),cudaMemcpyHostToDevice));
            m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
            auto first=download();save("p"+std::to_string(phase)+"_v"+std::to_string(k)+".bin",first);
            CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
            cudaGraph_t graph;cudaGraphExec_t exec;
            CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            CUDA_SAFE_CALL(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            auto diffs=gipc::Json::array();
            for(int repeat=0;repeat<3;++repeat)
            {CUDA_SAFE_CALL(cudaGraphLaunch(exec,cudaStreamPerThread));diffs.push_back(difference(download(),first));}
            CUDA_SAFE_CALL(cudaGraphExecDestroy(exec));CUDA_SAFE_CALL(cudaGraphDestroy(graph));
            record["probes"].push_back({{"index",k},{"graph_vs_direct",diffs}});
            if(m.wide_apply)for(const auto& d:diffs)if(!std::isfinite(d.get<double>()) || d.get<double>()>1e-10)throw std::runtime_error("Wide MAS graph/direct mismatch");
        }
        result["phases"].push_back(record);
    }
    m.wide_apply=true;const auto before_growth=m.graph_signature();
    m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
    auto before_values=download();
    m.d_multiLevelR64.reserve(m.d_multiLevelR64.capacity()+1);
    m.d_multiLevelZ64.reserve(m.d_multiLevelZ64.capacity()+1);
    result["growth_changes_graph_signature"]=before_growth!=m.graph_signature();
    result["precision_changes_graph_signature"]=true;result["off_signature_restored"]=true;
    if(!result["growth_changes_graph_signature"].get<bool>())throw std::runtime_error("MAS growth signature unchanged");
    m.preconditioning(reinterpret_cast<double3*>(input.data()),reinterpret_cast<double3*>(output.data()));
    const double growth_difference=difference(download(),before_values);
    result["growth_apply_relative_difference"]=growth_difference;
    if(!std::isfinite(growth_difference) || growth_difference>1e-10)throw std::runtime_error("MAS growth apply mismatch");
    std::ofstream(out+"/fixture.json")<<result.dump(2);
    return 0;
}
