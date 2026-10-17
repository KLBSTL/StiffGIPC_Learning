namespace full_apply_probe {
using Json=gipc::Json;namespace fs=std::filesystem;using Clock=std::chrono::steady_clock;
using gipc::balanced_cuda;using gipc::balanced_require;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
Json read_json(const fs::path& p){Json j;std::ifstream f(p);f>>j;balanced_require(bool(f),"JSON read failed");return j;}
void write_json(const fs::path& p,const Json& j){balanced_require(!fs::exists(p),"Output exists");std::ofstream f(p);f<<j.dump(2);f.close();balanced_require(bool(f),"JSON write failed");}
template<class T>std::vector<T> load(const fs::path& p,size_t count){
    balanced_require(fs::file_size(p)==count*sizeof(T),"Input byte size mismatch");std::vector<T> a(count);
    std::ifstream f(p,std::ios::binary);if(count)f.read(reinterpret_cast<char*>(a.data()),count*sizeof(T));
    balanced_require(bool(f),"Binary read failed");return a;
}
template<class T>std::vector<T> download(const T* p,size_t count){std::vector<T> a(count);
    if(count)balanced_cuda(cudaMemcpy(a.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return a;}
std::string save(const fs::path& p,const std::vector<double>& a){balanced_require(!fs::exists(p),"Binary output exists");
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(double));f.close();
    balanced_require(bool(f),"Binary output failed");return fs::absolute(p).generic_string();}
double relative(const std::vector<double>& a,const std::vector<double>& b){
    balanced_require(a.size()==b.size(),"Vector sizes differ");long double d=0,s=0;
    for(size_t i=0;i<a.size();++i){balanced_require(std::isfinite(a[i])&&std::isfinite(b[i]),"Nonfinite output");
        const long double delta=static_cast<long double>(a[i])-b[i];d+=delta*delta;s+=static_cast<long double>(b[i])*b[i];}
    return s>0?std::sqrt(double(d/s)):std::sqrt(double(d));
}
struct Saved {
    std::string name;void* p;std::vector<unsigned char> bytes;
    Saved(std::string name,void* p,size_t size):name(std::move(name)),p(p),bytes(download(static_cast<unsigned char*>(p),size)){}
    void restore()const{if(!bytes.empty())balanced_cuda(cudaMemcpyAsync(p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice,cudaStreamPerThread));}
    bool equal()const{return bytes==download(static_cast<unsigned char*>(p),bytes.size());}
};
struct Events {
    cudaEvent_t first=nullptr,last=nullptr;
    Events(){balanced_cuda(cudaEventCreate(&first));try{balanced_cuda(cudaEventCreate(&last));}catch(...){cudaEventDestroy(first);throw;}}
    ~Events(){if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);}
    double ms()const{float value=0;balanced_cuda(cudaEventElapsedTime(&value,first,last));return value;}
};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;Json info;
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
    void capture(const std::function<void()>& operation){
        const auto start=Clock::now();balanced_cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        try{gipc::CostCaptureGuard guard;operation();balanced_cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));}
        catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(cudaStreamPerThread,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);throw;}
        balanced_cuda(cudaGraphInstantiate(&exec,graph,0));
        size_t count=0;balanced_cuda(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
        balanced_cuda(cudaGraphGetNodes(graph,nodes.data(),&count));Json types=Json::object();
        for(auto node:nodes){cudaGraphNodeType type;balanced_cuda(cudaGraphNodeGetType(node,&type));const std::string key=std::to_string(int(type));types[key]=types.value(key,0)+1;}
        info={{"nodes",count},{"types_cuda_enum",types},{"capture_instantiate_host_ms",elapsed(start)}};
    }
};
Json device(){int index=0;balanced_cuda(cudaGetDevice(&index));cudaDeviceProp p{};balanced_cuda(cudaGetDeviceProperties(&p,index));
    std::ostringstream uuid;uuid<<std::hex<<std::setfill('0');for(unsigned char b:p.uuid.bytes)uuid<<std::setw(2)<<unsigned(b);
    return {{"name",p.name},{"index",index},{"uuid_hex",uuid.str()}};}
}

#include "graph_tiny_info.inl"
#include "cost_profile.h"
int MASPreconditioner::tiny_prefetch_fixture(const char* manifest_path,const char* raw_output,const char* action_raw) {
    const std::string action=action_raw;
    gipc::balanced_require(action=="contracts"||action=="measure"||action=="profile","Unknown action");
    const bool measure=action=="measure";
    using namespace full_apply_probe;using Space=gipc::NativeBalancedSpace;using Mode=Space::ApplyMode;
    const auto manifest=read_json(manifest_path);const fs::path out=raw_output;
    balanced_require(fs::is_directory(out)&&!fs::exists(out/"data.json"),"Fresh output directory required");
    Json result={{"passed",false},{"device",device()},{"cases",Json::array()},{"samples",Json::array()},
        {"private_graph_on_native_per_thread_stream",true},{"production_state_exists",false}};
    const std::array<std::string,3> methods={"original","cached","MAS"};
    const std::array<std::string,3> executions={"host16","graph16","graph1x16"};
    for(const auto& item:manifest.at("cases")) {
        const std::string name=item.at("name"),probe=item.at("probe"),prefix=item.at("prefix");
        if(probe!="rhs")continue;
        const int n=item.at("n");balanced_require(n>0&&n<100000,"Bounded size required");
        const Json meta=read_json(prefix+"_meta.json");balanced_require(meta.at("dofs")==n,"Global dimension differs");
        Json local,abd;for(const auto& p:meta.at("local_preconditioners")) {
            if(p.at("kind")=="MAS_full_owned_buffers"){balanced_require(local.is_null(),"Multiple MAS owners");local=p;}
            else if(p.at("kind")=="abd"){balanced_require(abd.is_null(),"Multiple ABD owners");abd=p;}
            else throw std::runtime_error("Unexpected local preconditioner");
        }
        balanced_require(!local.is_null()&&!abd.is_null()&&!meta.at("buffers").contains("diag_inverse"),"Unsupported global preconditioner");
        balanced_require(!local.at("wide_apply").get<bool>()&&!local.at("inverse64").get<bool>()&&!local.at("cholesky").get<bool>(),"Legacy MAS required");
        MASPreconditioner m;auto dim=local.at("dimensions").get<std::vector<int>>();balanced_require(dim.size()==8,"MAS dimensions differ");
        m.totalNodes=dim[0];m.totalMapNodes=dim[1];m.levelnum=dim[2];m.collision_node_Offset=dim[3];
        m.totalNumberClusters=dim[4];m.h_clevelSize={dim[5],dim[6]};m.neighborListSize=dim[7];m.precision_initialized=true;
        const int mas_offset=3*local.at("offset").get<int>(),abd_offset=3*abd.at("offset").get<int>(),abd_count=abd.at("count");
        balanced_require(abd_offset==0&&abd_count==1&&mas_offset==12&&mas_offset+3*m.totalNodes==n,"Unexpected local layout");
        balanced_require(m.totalNodes>0&&m.totalMapNodes>=m.totalNodes&&m.totalNumberClusters>=m.totalMapNodes,"Invalid MAS sizes");
        auto load=[&](auto& buffer,const char* key){
            using T=std::remove_pointer_t<decltype(buffer.data())>;const auto& spec=local.at("buffers").at(key);
            const size_t count=spec.at("count");balanced_require(spec.at("item_bytes")==sizeof(T),"MAS item width differs");
            const auto host=full_apply_probe::load<T>(prefix+"_mas_"+key+".bin",count);buffer.resize_discard(count);
            if(count)balanced_cuda(cudaMemcpy(buffer.data(),host.data(),count*sizeof(T),cudaMemcpyHostToDevice));
        };
        #include "load_buffers.inc"
        const auto signature=m.graph_signature();
        cudatool::DeviceBuffer<gipc::Matrix12x12> inverse(abd_count);
        const auto inverse_host=full_apply_probe::load<gipc::Matrix12x12>(prefix+"_"+abd.at("buffer").get<std::string>()+".bin",abd_count);
        balanced_cuda(cudaMemcpy(inverse.data(),inverse_host.data(),inverse_host.size()*sizeof(gipc::Matrix12x12),cudaMemcpyHostToDevice));
        gipc::BalancedBuffer<double> r(n),z(n);
        constexpr size_t workspace_bytes=4*1024*1024,guard_bytes=256;
        gipc::BalancedBuffer<unsigned char> workspaceStorage(workspace_bytes+2*guard_bytes);Space space(n);
        auto upload=[&](double* target,const std::string& path,size_t count){auto host=full_apply_probe::load<double>(path,count);
            balanced_cuda(cudaMemcpy(target,host.data(),count*sizeof(double),cudaMemcpyHostToDevice));};
        upload(r.value,item.at("files").at("r"),n);upload(space.Z.value,item.at("Z"),size_t(n)*8);
        upload(space.W.value,item.at("W"),size_t(n)*8);upload(space.L.value,item.at("L"),64);
        balanced_cuda(cudaMemset(z.value,0,n*sizeof(double)));balanced_cuda(cudaMemset(space.intermediate.value,0,n*sizeof(double)));
        balanced_cuda(cudaMemset(space.a.value,0,8*sizeof(double)));balanced_cuda(cudaMemset(space.c.value,0,8*sizeof(double)));
        std::vector<Saved> scratch,readonly;
        m.diagnostic_buffers([&](const char* key,const void* p,size_t count,size_t width){
            auto& bank=std::string(key)=="d_multiLevelR"||std::string(key)=="d_multiLevelZ"?scratch:readonly;
            bank.emplace_back(key,const_cast<void*>(p),count*width);
        });
        scratch.emplace_back("output",z.value,n*sizeof(double));scratch.emplace_back("balanced_intermediate",space.intermediate.value,n*sizeof(double));
        scratch.emplace_back("balanced_a",space.a.value,8*sizeof(double));scratch.emplace_back("balanced_c",space.c.value,8*sizeof(double));
        readonly.emplace_back("r",r.value,n*sizeof(double));readonly.emplace_back("Z",space.Z.value,size_t(n)*8*sizeof(double));
        readonly.emplace_back("W",space.W.value,size_t(n)*8*sizeof(double));readonly.emplace_back("L",space.L.value,64*sizeof(double));
        readonly.emplace_back("ABD_inverse",inverse.data(),abd_count*sizeof(gipc::Matrix12x12));
        auto reset=[&](){const auto start=Clock::now();for(const auto& s:scratch)s.restore();balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));return elapsed(start);};
        auto check_state=[&](){balanced_require(m.graph_signature()==signature,"MAS signature changed");
            for(const auto& s:readonly)balanced_require(s.equal(),("Read-only changed: "+s.name).c_str());
            for(const auto& s:scratch)balanced_require(s.equal(),("Restore failed: "+s.name).c_str());};
        auto abd_action=[&](const double* input,double* output){
            cudatool::CDenseVectorView<double> rv{input+abd_offset,12*abd_count};
            cudatool::DenseVectorView<double> zv{output+abd_offset,12*abd_count};
            LaunchCudaKernal_default(abd_count,256,0,abd_preconditioner_apply_kernel,abd_count,rv.cviewer(),zv.viewer(),inverse.view().viewer());
        };
        NativeCostProfile costs;
        auto base=[&](const double* input,double* output){
            costs.call("quality_native_MAS",[&]{
            balanced_cuda(cudaMemcpyAsync(output,input,n*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            abd_action(input,output);m.preconditioning(reinterpret_cast<const double3*>(input+mas_offset),reinterpret_cast<double3*>(output+mas_offset));
            });
        };
        unsigned char* workspace=workspaceStorage.value+guard_bytes;
        balanced_require(reinterpret_cast<std::uintptr_t>(workspace)%256==0,"Workspace alignment");
        balanced_cuda(cudaMemset(workspaceStorage.value,0x5a,workspace_bytes+2*guard_bytes));
        balanced_cuda(cudaMemset(workspace,0xa5,workspace_bytes));
        gipc::balanced_blas(cublasSetWorkspace(space.blas.handle,workspace,workspace_bytes));
        auto guards=[&]{
            for(auto b:download(workspaceStorage.value,guard_bytes))balanced_require(b==0x5a,"Workspace left guard");
            for(auto b:download(workspace+workspace_bytes,guard_bytes))balanced_require(b==0x5a,"Workspace right guard");
        };
        // Immutable block COO and native kernels; no CSR conversion on the GPU.
        const int count=meta.at("blocks"),blocks=(n+255)/256;int max_iter=2001;
        gipc::BalancedBuffer<int> rows(count),cols(count);
        gipc::BalancedBuffer<Eigen::Matrix3d> values(count);
        auto row_host=full_apply_probe::load<int>(prefix+"_rows.bin",count);
        auto col_host=full_apply_probe::load<int>(prefix+"_cols.bin",count);
        auto val_host=full_apply_probe::load<Eigen::Matrix3d>(prefix+"_values.bin",count);
        balanced_cuda(cudaMemcpy(rows.value,row_host.data(),count*sizeof(int),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(cols.value,col_host.data(),count*sizeof(int),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(values.value,val_host.data(),count*sizeof(Eigen::Matrix3d),cudaMemcpyHostToDevice));
        readonly.emplace_back("rows",rows.value,count*sizeof(int));readonly.emplace_back("cols",cols.value,count*sizeof(int));
        readonly.emplace_back("values",values.value,count*sizeof(Eigen::Matrix3d));
        gipc::BalancedBuffer<double> b(n),work_x(n),p(n),Ap(n),quality_r(n),quality_z(n),residual_ax(n),dot_partials(blocks),reduction_result(1);
        upload(b.value,prefix+"_rhs.bin",n);
        balanced_require(download(b.value,n)==download(r.value,n),"RHS differs from sealed apply input");
        // r becomes the recurrence scratch, so remove only its read-only entry.
        readonly.erase(std::remove_if(readonly.begin(),readonly.end(),[](const Saved& s){return s.name=="r";}),readonly.end());
        readonly.emplace_back("b",b.value,n*sizeof(double));
        gipc::Spmv spmv;
        auto multiply=[&](const double* input,double* output){
            costs.call("SpMV",[&]{spmv.warp_reduce_sym_spmv(1.,values.value,rows.value,cols.value,count,
                cudatool::CDenseVectorView<double>{input,n},0.,cudatool::DenseVectorView<double>{output,n});
            balanced_cuda(cudaGetLastError());});
        };
        auto check=[](cudaError_t status){balanced_cuda(status);};
        auto milliseconds=[&](Clock::time_point start){return elapsed(start);};
        auto dot=[&](const double* a,const double* c){return costs.call("dot_with_host_readback",[&]{return My_PCG_General_v_v_Reduction_Algorithm(dot_partials.value,a,c,reduction_result.value,n);});};
        const double rhs_norm2=dot(b.value,b.value);
        reset();base(b.value,quality_z.value);const double common_bMb=dot(b.value,quality_z.value);
        balanced_require(std::isfinite(rhs_norm2)&&rhs_norm2>0&&std::isfinite(common_bMb)&&common_bMb>0,"Invalid RHS quadratic");
        multiply(b.value,residual_ax.value);
        Json c={{"name",name},{"n",n},{"rhs_norm2",rhs_norm2},{"common_bMb",common_bMb},
            {"native_A_b_file",save(out/(name+"_native_A_b.bin"),download(residual_ax.value,n))},
            {"native_MAS_b_file",save(out/(name+"_native_MAS_b.bin"),download(quality_z.value,n))},
            {"graphs",Json::array()},{"budget_checks",Json::array()},{"warm",Json::array()},{"runs",Json::array()},{"zero",Json::array()}};
        auto direct=[&](int method){
            if(method==2)base(r.value,z.value);
            else space.apply(r.value,z.value,base,nullptr,Mode::FusedProjection,method?2:0);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::unique_ptr<Graph>,3> graphs;
        for(int method=0;method<3;++method){
            reset();direct(method);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            graphs[method]=std::make_unique<Graph>();graphs[method]->capture([&]{direct(method);});
            size_t opaque=0;auto info=tiny_launch_info(*graphs[method],method==1?2:0,&opaque);
            balanced_require(info.size()==(method==2?0:2)&&opaque==(method==2?0:4),"Wrong PCG apply graph");
            c["graphs"].push_back({{"method",methods[method]},{"tiny",info},{"opaque",opaque},{"info",graphs[method]->info}});
        }
        int current_method=0,execution=0;bool current_cooldown=false;
        auto op=[&](int method){
            costs.call("recurrence_apply",[&]{if(execution)balanced_cuda(cudaGraphLaunch(graphs[method]->exec,cudaStreamPerThread));else direct(method);});
        };
        const std::string balanced_fault="none";
        using gipc::NativeBalancedRejected;using gipc::native_quality_check_due;
        #include "recurrence.inl"
        auto run=[&](int arm,int mode,const std::string& tag){
            const int method=arm%2?1:2;current_method=method;execution=mode;current_cooldown=arm>=2;
            double reset_ms=reset();auto t=Clock::now();
            balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));reset_ms+=elapsed(t);
            Events events;Json output;costs.stages=Json::object();costs.enabled=action=="profile";
            const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));
            solve_candidate(method!=2,true,method!=2,output);
            balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));balanced_cuda(cudaEventSynchronize(events.last));
            const double host_ms=elapsed(start),event_ms=events.ms();costs.enabled=false;
            balanced_require(costs.depth==0,"Profiler depth leak");
            const Json profile={{"enabled",action=="profile"},{"stages",costs.stages}};const auto export_start=Clock::now();
            auto x=download(work_x.value,n);for(double value:x)balanced_require(std::isfinite(value),"Nonfinite solution");
            const std::string file=save(out/(tag+"_x.bin"),x);
            // Explicit terminal vector and native MAS action are outside timing.
            multiply(work_x.value,residual_ax.value);
            audit_residual<<<blocks,256,0,cudaStreamPerThread>>>(quality_r.value,b.value,residual_ax.value,n);
            base(quality_r.value,quality_z.value);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            auto residual=download(quality_r.value,n),weighted=download(quality_z.value,n);
            const auto rf=save(out/(tag+"_residual.bin"),residual),mf=save(out/(tag+"_MAS_residual.bin"),weighted);
            const double terminal_l2=std::sqrt(dot(quality_r.value,quality_r.value)/rhs_norm2);
            const double terminal_mas=std::sqrt(dot(quality_r.value,quality_z.value)/common_bMb);
            balanced_require(std::isfinite(terminal_l2)&&std::isfinite(terminal_mas)&&terminal_l2<=.0095&&terminal_mas<=.0095,"Post-solve explicit gate failed");
            const double export_ms=elapsed(export_start);
            reset();check_state();guards();
            return Json{{"arm",arm},{"policy",current_cooldown?"cooldown2":"legacy"},{"profile",profile},{"method",methods[method]},{"execution",mode?"graph_apply1":"host"},{"solve_host_ms",host_ms},{"solve_event_ms",event_ms},
                {"reset_host_ms",reset_ms},{"export_audit_host_ms",export_ms},{"solution_file",file},{"residual_file",rf},{"MAS_residual_file",mf},
                {"terminal_l2",terminal_l2},{"terminal_MAS",terminal_mas},{"result",output}};
        };
        // Algebra-only zero contracts; the nonzero-RHS PCG recurrence is not run.
        for(int mode=0;mode<2;++mode)for(int method=0;method<3;++method){
            balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            balanced_cuda(cudaMemsetAsync(r.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            execution=mode;op(method);multiply(work_x.value,residual_ax.value);
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            for(double value:download(residual_ax.value,n))balanced_require(value==0,"A zero input not zero");
            for(double value:download(z.value,n))balanced_require(value==0,"Preconditioner zero input not zero");
            const auto file=save(out/(name+"_zero_m"+std::to_string(method)+"_e"+std::to_string(mode)+".bin"),download(z.value,n));
            c["zero"].push_back({{"execution",mode?"graph_apply1":"host"},{"method",methods[method]},{"output_file",file},{"algebra_only",true}});
        }
// Negative budget cases run before successful contracts to expose state leaks.
if(action=="contracts") {
    max_iter=3;
    for(int mode=0;mode<2;++mode)for(int arm=0;arm<4;++arm) {
        const int method=arm%2?1:2;current_method=method;execution=mode;current_cooldown=arm>=2;
        reset();balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));Json failed;bool rejected=false;
        try{solve_candidate(method!=2,true,method!=2,failed);}
        catch(const NativeBalancedRejected& error){
            balanced_require(std::string(error.what())=="Native diagnostic iteration budget reached","Wrong budget failure");rejected=true;
        }
        balanced_require(rejected&&!failed.at("converged").get<bool>()&&failed.at("iterations")==2
            &&failed.at("last_quality_iteration")==2&&failed.at("trace").back().at("forced_quality_check").get<bool>(),"Budget did not force explicit rejection");
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
        const auto file=save(out/(name+"_budget_a"+std::to_string(arm)+"_e"+std::to_string(mode)+"_x.bin"),download(work_x.value,n));
        reset();check_state();guards();c["budget_checks"].push_back({{"arm",arm},{"method",methods[method]},
            {"policy",current_cooldown?"cooldown2":"legacy"},{"execution",mode?"graph_apply1":"host"},
            {"solution_file",file},{"result",failed},{"state_restored",true}});
    }
    max_iter=2001;
}
for(int mode=0;mode<2;++mode)for(int arm=0;arm<4;++arm)
    c["warm"].push_back(run(arm,mode,name+"_warm_a"+std::to_string(arm)+"_e"+std::to_string(mode)));
if(measure) {
    const int comparison_arms[6][2]={{0,0},{2,2},{0,1},{2,3},{0,2},{1,3}};
    const char* comparison_names[6]={"AA_legacy","AA_cooldown","legacy_net","cooldown_net","MAS_policy","cached_policy"};
    for(int round=0;round<3;++round)for(int mo=0;mo<2;++mo)for(int q=0;q<4;++q)
        for(int co=0;co<6;++co)for(int pos=0;pos<4;++pos) {
            const int mode=(mo+round)%2,comparison=(co+q+round)%6;
            const int label=int(pos==0||pos==3)^((round+q)%2),arm=comparison_arms[comparison][label];
            const std::string tag=name+"_r"+std::to_string(round+1)+"_e"+std::to_string(mode)+"_q"+std::to_string(q+1)+"_c"+std::to_string(comparison)+"_p"+std::to_string(pos+1);
            auto row=run(arm,mode,tag);row["round"]=round+1;row["quartet"]=q+1;row["position"]=pos+1;
            row["comparison"]=comparison_names[comparison];row["label"]=label?"B":"A";c["runs"].push_back(row);
        }
}
        reset();check_state();guards();c["readonly_and_scratch_unchanged"]=true;result["cases"].push_back(c);
    }
    result["passed"]=true;result["action"]=action;result["expected_measured_solves"]=measure?1152:0;
    write_json(out/"data.json",result);return 0;
}
