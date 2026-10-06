// Included inside fixed_system_study. One frozen A/b/FP32 inverse/map;
// candidate switches only between complete solves, never inside a PCG solve.
const auto saved_signature=graph_signature();
std::vector<Float> saved_z(z.size());
CUDA_SAFE_CALL(cudaMemcpy(saved_z.data(),z.data(),saved_z.size()*sizeof(Float),cudaMemcpyDeviceToHost));
const auto saved_captures=captures,saved_hits=cache_hits,saved_invalidations=invalidations;
struct PrivateCache
{
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    std::vector<std::uintptr_t> key;SizeT limit=0;Float tol=0;
    ~PrivateCache(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
    void exchange(cudaGraph_t& g,cudaGraphExec_t& e,std::vector<std::uintptr_t>& k,SizeT& l,Float& t)
    {std::swap(graph,g);std::swap(exec,e);key.swap(k);std::swap(limit,l);std::swap(tol,t);}
} caches[2];
auto solve_arm=[&](int mode,int arm)
{
    LegacyRestrictionArm restriction(arm!=0);
    if(mode==0)return pcg(x,b,max_iter);
    auto& cache=caches[arm];
    cache.exchange(graph,graph_exec,captured_key,captured_max_iter,captured_tol);
    try
    {
        const auto iterations=pcg_graph(x,b,max_iter);
        cache.exchange(graph,graph_exec,captured_key,captured_max_iter,captured_tol);
        return iterations;
    }
    catch(...){cache.exchange(graph,graph_exec,captured_key,captured_max_iter,captured_tol);throw;}
};
auto read_vector=[](const auto& view)
{
    std::vector<Float> values(view.size());
    CUDA_SAFE_CALL(cudaMemcpy(values.data(),view.data(),values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    return values;
};
auto save_vector=[&](const std::string& suffix,const std::vector<Float>& values)
{
    const auto path=prefix+suffix+".bin";
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
    if(!file)throw std::runtime_error("Failed legacy restriction diagnostic export");
    return path;
};
auto difference=[](const std::vector<Float>& values,const std::vector<Float>& reference)
{
    double error=0,norm=0;size_t different=0;
    for(size_t i=0;i<values.size();++i)
    {error+=(values[i]-reference[i])*(values[i]-reference[i]);norm+=reference[i]*reference[i];
     different+=std::memcmp(&values[i],&reference[i],sizeof(Float))!=0;}
    return Json{{"relative_l2",norm>0?std::sqrt(error/norm):std::sqrt(error)},
                {"different_components",different}};
};
auto true_residual=[&]()
{
    spmv(cudatool::CDenseVectorView<Float>{x.data(),int(x.size())},Ap.view());
    const auto ax=read_vector(Ap.view()),rhs=read_vector(b);
    long double error=0,norm=0;
    for(size_t i=0;i<rhs.size();++i){const long double delta=(long double)rhs[i]-ax[i];error+=delta*delta;norm+=(long double)rhs[i]*rhs[i];}
    return norm>0?std::sqrt(double(error/norm)):std::sqrt(double(error));
};
study["legacy_restriction_pairs"]=Json::array();study["preconditioner_probes"]=Json::array();
study["rho_tolerance"]=m_config.global_tol_rate;
study["scope"]="Default rho; 3 interleaved pairs per host/Graph; separate caches warm once and preserve production cache. CSR build cost included in optimistic full-linear estimate, distribution excluded. No strict-rho sweep.";
study["prepare_with_map_ms"]=saved_info.value("diagnostic_prepare_with_map_ms",0.0);
double map_ms=0;
for(const auto& local:before["local_preconditioners"])
    if(local.value("kind",std::string{})=="MAS_full_owned_buffers")map_ms+=local.value("restriction_map_prepare_ms",0.0);
study["map_prepare_ms"]=map_ms;
study["atomic_prepare_estimate_ms"]=std::max(0.0,study["prepare_with_map_ms"].get<double>()-map_ms);
try
{
    // Complete M includes global/ABD contributions. Export b, zero and two
    // bounded independent trigonometric vectors, three observations per arm.
    const auto rhs=read_vector(b);
    DeviceDenseVector input;input.resize(b.size());
    for(int probe=0;probe<4;++probe)
    {
        auto values=rhs;
        if(probe>0)for(size_t i=0;i<values.size();++i)
            values[i]=probe==1?0.0:probe==2?std::sin(double(i+1)*0.731):std::cos(double(i+1)*0.319);
        const auto input_file=save_vector("_probe"+std::to_string(probe)+"_r",values);
        CUDA_SAFE_CALL(cudaMemcpy(input.data(),values.data(),values.size()*sizeof(Float),cudaMemcpyHostToDevice));
        for(int arm=0;arm<2;++arm)
        {
            LegacyRestrictionArm restriction(arm!=0);
            std::vector<Float> first;
            for(int repeat=0;repeat<3;++repeat)
            {
                apply_preconditioner(z.view(),input.cview());
                const auto output=read_vector(z.view());if(repeat==0)first=output;
                long double quadratic=0;bool finite=true;
                for(size_t i=0;i<values.size();++i){quadratic+=(long double)values[i]*output[i];finite&=std::isfinite(output[i]);}
                study["preconditioner_probes"].push_back({{"probe",probe},{"arm",arm==0?"atomic":"ordered"},
                    {"repeat",repeat+1},{"input_file",input_file},{"output_file",save_vector("_probe"+std::to_string(probe)+"_a"+std::to_string(arm)+"_r"+std::to_string(repeat+1),output)},
                    {"finite",finite},{"quadratic",double(quadratic)},{"repeat_difference",difference(output,first)}});
                if(!finite || (probe!=1&&quadratic<=0))throw std::runtime_error("Legacy preconditioner probe failed finite/positive quadratic check");
            }
        }
    }
    for(int mode=0;mode<2;++mode)
    {
        for(int arm=0;arm<2;++arm)
        {
            x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const auto begin=std::chrono::steady_clock::now();
            const auto iterations=solve_arm(mode,arm);CUDA_SAFE_CALL(cudaDeviceSynchronize());
            study["warmups"].push_back({{"mode",mode==0?"host":"graph"},{"arm",arm==0?"atomic":"ordered"},
                {"iterations",iterations},{"ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()},
                {"capture_ms",mode?info.value("graph_capture_instantiate_host_ms",0.0):0.0}});
        }
        for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order)
        {
            const int arm=(order+pair%2)%2;
            x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const auto begin=std::chrono::steady_clock::now();
            const auto iterations=solve_arm(mode,arm);CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            const auto output=read_vector(x);const auto actual_residual=true_residual();
            bool finite=std::all_of(output.begin(),output.end(),[](Float value){return std::isfinite(value);});
            const double prepare=study[arm?"prepare_with_map_ms":"atomic_prepare_estimate_ms"].get<double>();
            study["legacy_restriction_pairs"].push_back({{"mode",mode==0?"host":"graph"},{"pair",pair+1},
                {"arm",arm==0?"atomic":"ordered"},{"iterations",iterations},{"solve_ms",elapsed},
                {"optimistic_full_linear_ms",prepare+elapsed},{"finite",finite},{"true_relative_residual",actual_residual},
                {"rho_initial",info.value("rho_initial",0.0)},{"rho_stop",info.value("rho_stop",0.0)},
                {"graph_cache_hit",mode?info.value("graph_cache_hit",false):false},
                {"solution_file",save_vector("_solve_m"+std::to_string(mode)+"_a"+std::to_string(arm)+"_p"+std::to_string(pair+1),output)}});
            if(iterations>=max_iter || !finite || !std::isfinite(actual_residual))throw std::runtime_error("Legacy restriction default PCG failed");
        }
    }
}
catch(const std::exception& e){study["diagnostic_error"]=e.what();}
CUDA_SAFE_CALL(cudaDeviceSynchronize());
captures=saved_captures;cache_hits=saved_hits;invalidations=saved_invalidations;
m_config=saved_config;
CUDA_SAFE_CALL(cudaMemcpy(x.data(),primary.data(),primary.size()*sizeof(Float),cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(z.data(),saved_z.data(),saved_z.size()*sizeof(Float),cudaMemcpyHostToDevice));
study["system_unchanged"]=snapshot_operator_identity(before)==snapshot_operator_identity(snapshot_system(""));
study["scratch_not_part_of_operator_identity"]={"d_multiLevelR","d_multiLevelZ","d_multiLevelR64","d_multiLevelZ64"};
const auto restored_x=read_vector(x),restored_z=read_vector(z.view());
study["primary_restored_bitwise"]=std::memcmp(restored_x.data(),primary.data(),primary.size()*sizeof(Float))==0;
study["z_restored_bitwise"]=std::memcmp(restored_z.data(),saved_z.data(),saved_z.size()*sizeof(Float))==0;
study["graph_signature_restored"]=graph_signature()==saved_signature;
info=saved_info;info["fixed_study_file"]=prefix+"_study.json";
std::ofstream file(prefix+"_study.json");file<<study.dump(2);file.close();
if(!file || study.contains("diagnostic_error") || !study["system_unchanged"].get<bool>() ||
   !study["primary_restored_bitwise"].get<bool>() || !study["z_restored_bitwise"].get<bool>() ||
   !study["graph_signature_restored"].get<bool>())throw std::runtime_error("Legacy restriction study failed; details retained");
