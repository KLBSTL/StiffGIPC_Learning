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

int MASPreconditioner::full_apply_graph_fixture(const char* manifest_path,const char* raw_output,bool measure) {
    using namespace full_apply_probe;using Space=gipc::NativeBalancedSpace;using Mode=Space::ApplyMode;
    const auto manifest=read_json(manifest_path);const fs::path out=raw_output;
    balanced_require(fs::is_directory(out)&&!fs::exists(out/(measure?"measurements.json":"contracts.json")),"Fresh output directory required");
    Json result={{"passed",false},{"device",device()},{"cases",Json::array()},{"samples",Json::array()},
        {"private_graph_on_native_per_thread_stream",true},{"production_state_exists",false}};
    const std::array<std::string,3> methods={"MAS","reference","fused"};
    const std::array<std::string,3> executions={"host16","graph16","graph1x16"};
    for(const auto& item:manifest.at("cases")) {
        const std::string name=item.at("name"),probe=item.at("probe"),prefix=item.at("prefix");
        if(measure&&probe=="zero")continue;
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
        gipc::BalancedBuffer<double> r(n),z(n);Space space(n);
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
        auto base=[&](const double* input,double* output){
            balanced_cuda(cudaMemcpyAsync(output,input,n*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            abd_action(input,output);m.preconditioning(reinterpret_cast<const double3*>(input+mas_offset),reinterpret_cast<double3*>(output+mas_offset));
        };
        auto op=[&](int method,const Space::Profile* profile=nullptr){
            if(method==0)base(r.value,z.value);
            else space.apply(r.value,z.value,base,profile,method==2?Mode::FusedProjection:Mode::Reference);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::vector<double>,3> cpu={full_apply_probe::load<double>(item.at("files").at("MAS_expected"),n),
            full_apply_probe::load<double>(item.at("files").at("balanced_expected"),n),{}};cpu[2]=cpu[1];
        std::array<std::vector<double>,3> host_reference;
        Json case_record={{"name",name},{"probe",probe},{"n",n},{"graphs",Json::array()},{"warm",Json::array()},
            {"stage_profiles",Json::array()},{"restoration_passed",false}};
        std::array<std::array<std::unique_ptr<Graph>,2>,3> graphs;
        auto validate=[&](int method,const std::vector<double>& output){const double error=relative(output,cpu[method]);
            balanced_require(std::isfinite(error)&&error<=1e-3,"Native action differs from CPU reference");return error;};
        // Initialize every library/kernel before capture. Host reference uses same final 16 calls.
        for(int method=0;method<3;++method){reset();for(int k=0;k<16;++k)op(method);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            host_reference[method]=download(z.value,n);validate(method,host_reference[method]);
            for(int g=0;g<2;++g){reset();graphs[method][g]=std::make_unique<Graph>();const int count=g?16:1;
                graphs[method][g]->capture([&](){for(int k=0;k<count;++k)op(method);});
                reset();const auto start=Clock::now();balanced_cuda(cudaGraphLaunch(graphs[method][g]->exec,cudaStreamPerThread));
                balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));auto graph_info=graphs[method][g]->info;
                graph_info["first_launch_host_ms"]=elapsed(start);graph_info["calls"]=count;graph_info["method"]=methods[method];
                validate(method,download(z.value,n));case_record["graphs"].push_back(graph_info);
            }
        }
        auto execute=[&](int method,int mode){
            if(mode==0)for(int k=0;k<16;++k)op(method);
            else if(mode==1)balanced_cuda(cudaGraphLaunch(graphs[method][1]->exec,cudaStreamPerThread));
            else for(int k=0;k<16;++k)balanced_cuda(cudaGraphLaunch(graphs[method][0]->exec,cudaStreamPerThread));
        };
        for(int method=0;method<3;++method)for(int mode=0;mode<3;++mode){reset();execute(method,mode);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            const auto values=download(z.value,n);const double error=validate(method,values),diff=relative(values,host_reference[method]);
            balanced_require(diff<=1e-3,"Host/Graph native action differs");Json warm={{"method",methods[method]},{"execution",executions[mode]},
                {"CPU_relative",error},{"host_relative",diff},{"output",save(out/(name+"_warm_"+methods[method]+"_"+executions[mode]+".bin"),values)}};
            if(probe=="rhs"){
                const auto historical=full_apply_probe::load<double>(item.at("historical_b").at(methods[method]),n);
                const double difference=relative(values,historical);balanced_require(difference<=1e-3,"Historical native action calibration failed");
                warm["historical_native_relative"]=difference;
            }
            case_record["warm"].push_back(warm);
        }
        if(measure){Events events;
            for(int round=0;round<3;++round)for(int group=0;group<6;++group)for(int order=0;order<9;++order){
                const int code=(order+group*5+round*3)%9,method=code/3,mode=code%3;const double reset_ms=reset();
                const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));execute(method,mode);
                balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));const double submit_ms=elapsed(start);
                balanced_cuda(cudaEventSynchronize(events.last));const double host_ms=elapsed(start),event_ms=events.ms();
                const auto check_start=Clock::now();const auto values=download(z.value,n);const double error=validate(method,values);
                const double difference=relative(values,host_reference[method]);balanced_require(difference<=1e-3,"Measured Host/Graph action mismatch");
                const std::string tag=name+"_r"+std::to_string(round+1)+"_g"+std::to_string(group+1)+"_"+methods[method]+"_"+executions[mode];
                const std::string file=save(out/(tag+".bin"),values);
                result["samples"].push_back({{"name",name},{"round",round+1},{"group",group+1},{"order",order+1},{"method",methods[method]},
                    {"execution",executions[mode]},{"calls",16},{"host_ms",host_ms},{"submit_ms",submit_ms},{"event_ms",event_ms},
                    {"reset_ms",reset_ms},{"verify_export_ms",elapsed(check_start)},{"CPU_relative",error},{"host_relative",difference},{"output",file}});
            }
            // Instrumented diagnosis, kept outside every paired measurement.
            for(int method=1;method<3;++method){reset();Json stages=Json::array();std::vector<std::unique_ptr<Events>> markers;
                Space::Profile profile=[&](const char* stage,const std::function<void()>& operation){
                    markers.push_back(std::make_unique<Events>());auto& e=*markers.back();balanced_cuda(cudaEventRecord(e.first,cudaStreamPerThread));
                    const auto start=Clock::now();operation();const double submit=elapsed(start);balanced_cuda(cudaEventRecord(e.last,cudaStreamPerThread));
                    stages.push_back({{"stage",stage},{"submit_ms",submit}});
                };
                op(method,&profile);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));validate(method,download(z.value,n));
                for(size_t i=0;i<markers.size();++i)stages[i]["event_ms"]=markers[i]->ms();
                case_record["stage_profiles"].push_back({{"method",methods[method]},{"stages",stages},{"instrumented",true}});
            }
            reset();Json internal=Json::array();std::vector<std::unique_ptr<Events>> markers;
            auto stage=[&](const char* label,const std::function<void()>& operation){
                markers.push_back(std::make_unique<Events>());auto& e=*markers.back();balanced_cuda(cudaEventRecord(e.first,cudaStreamPerThread));
                const auto start=Clock::now();operation();const double submit=elapsed(start);balanced_cuda(cudaEventRecord(e.last,cudaStreamPerThread));
                internal.push_back({{"stage",label},{"submit_ms",submit}});
            };
            stage("global_identity_copy",[&]{balanced_cuda(cudaMemcpyAsync(z.value,r.value,n*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));});
            stage("ABD",[&]{abd_action(r.value,z.value);});
            stage("MAS_workspace_zero",[&]{
                balanced_cuda(cudaMemsetAsync(m.d_multiLevelR.data()+m.totalMapNodes,0,(m.totalNumberClusters-m.totalMapNodes)*sizeof(Eigen::Vector3f),cudaStreamPerThread));
                balanced_cuda(cudaMemsetAsync(m.d_multiLevelZ.data(),0,m.totalNumberClusters*sizeof(Precision_T3),cudaStreamPerThread));
            });
            stage("MAS_restrict",[&]{m.BuildMultiLevelR(reinterpret_cast<const double3*>(r.value+mas_offset));});
            stage("MAS_local_action",[&]{m.SchwarzLocalXSym_block3();});
            stage("MAS_prolong",[&]{m.CollectFinalZ(reinterpret_cast<double3*>(z.value+mas_offset));});
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));const auto values=download(z.value,n);validate(0,values);
            balanced_require(relative(values,host_reference[0])<=1e-3,"Instrumented internal MAS differs");
            for(size_t i=0;i<markers.size();++i)internal[i]["event_ms"]=markers[i]->ms();
            case_record["instrumented_MAS_internal"]={{"stages",internal},{"output",save(out/(name+"_instrumented_MAS.bin"),values)},
                {"scope","Separate dispatch of original zero/restrict/local/prolong methods; not additive to uninstrumented timings"}};
        }
        // Deliberate scratch poison and post-launch exception must restore all original bytes.
        reset();for(auto& s:scratch)if(!s.bytes.empty())balanced_cuda(cudaMemsetAsync(s.p,0xa5,s.bytes.size(),cudaStreamPerThread));
        reset();check_state();bool caught=false;
        try{execute(2,1);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));throw std::runtime_error("probe injected after Graph apply");}
        catch(const std::runtime_error& e){balanced_require(std::string(e.what())=="probe injected after Graph apply","Wrong injected exception");caught=true;reset();}
        balanced_require(caught,"Failure contract did not execute");check_state();
        case_record["restoration_passed"]=true;case_record["post_graph_exception_restored"]=true;
        case_record["readonly_buffers_checked"]=readonly.size();case_record["scratch_buffers_checked"]=scratch.size();
        result["cases"].push_back(case_record);
    }
    balanced_require(result["cases"].size()==(measure?4:6),"Case count differs");
    balanced_require(result["samples"].size()==(measure?648:0),"Measured sample count differs");
    result["passed"]=true;write_json(out/(measure?"measurements.json":"contracts.json"),result);return 0;
}
