// Explicit diagnostic only: freeze the local MAS action produced by the old
// path, then compare the final collect and complete (ABD + FEM) rho. Re-running
// the atomic legacy restriction is not used as an equality oracle.
namespace gipc
{
void PCGSolver::mas_dot_fixed_study(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,const std::string& prefix)
{
    if(!mas_dot_supported || mas_dot_partial_count<=0)
        throw std::runtime_error("Fused MAS dot study unavailable: "+mas_dot_reason);
    CostSampleGuard sample("frozen_operator_probe");
    CostScope scope("diagnostic.mas_fused_dot_including_audit");
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const auto saved_info=info;
    const bool saved_effective=mas_dot_effective;
    const auto before=snapshot_system(prefix);
    const auto signature=graph_signature(),saved_key=captured_key;
    const auto saved_graph=graph;const auto saved_exec=graph_exec;
    struct SavedBuffer{void* address;std::vector<unsigned char> bytes;};
    std::vector<SavedBuffer> saved;
    auto save=[&](void* address,size_t bytes){
        if(!bytes)return;
        SavedBuffer entry{address,std::vector<unsigned char>(bytes)};
        CUDA_SAFE_CALL(cudaMemcpy(entry.bytes.data(),address,bytes,cudaMemcpyDeviceToHost));
        saved.push_back(std::move(entry));
    };
    save(x.data(),x.size()*sizeof(Float));
    save(z.data(),z.size()*sizeof(Float));save(r.data(),r.size()*sizeof(Float));
    save(p.data(),p.size()*sizeof(Float));save(Ap.data(),Ap.size()*sizeof(Float));
    save(reduction_result.data(),reduction_result.size()*sizeof(Float));
    save(graph_scalars.data(),graph_scalars.size()*sizeof(Float));
    save(graph_reduce_storage.data(),graph_reduce_storage.size());
    save(mas_dot_partials.data(),mas_dot_partials.size()*sizeof(Float));
    save(mas_dot_reduce_storage.data(),mas_dot_reduce_storage.size());
    mas_dot_scratch(save);
    std::vector<Float> rhs(b.size());
    CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),b.data(),rhs.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    auto restore=[&](){for(const auto& entry:saved)
        CUDA_SAFE_CALL(cudaMemcpy(entry.address,entry.bytes.data(),entry.bytes.size(),cudaMemcpyHostToDevice));};
    Json study={{"scope","Frozen local MAS action; final collect and all final-z components"},
        {"full_PCG_speed_or_quality_certified",false},{"system",before},
        {"partial_count",mas_dot_partial_count},{"cases",Json::array()},{"passed",true}};
    attach_solve_context(study);
    cudatool::DeviceBuffer<Float> input(b.size());
    cudaGraph_t probe_graph=nullptr;cudaGraphExec_t probe_exec=nullptr;
    cudatool::DeviceBuffer<unsigned char> old_graph_storage;
    cudatool::DeviceBuffer<Float> old_graph_scalars;
    bool graph_buffers_swapped=false;
    bool capturing=false;
    auto cleanup=[&](){
        if(probe_exec){cudaGraphExecDestroy(probe_exec);probe_exec=nullptr;}
        if(probe_graph){cudaGraphDestroy(probe_graph);probe_graph=nullptr;}
    };
    auto restore_graph_buffers=[&](){
        if(graph_buffers_swapped)
        {
            graph_reduce_storage=std::move(old_graph_storage);
            graph_scalars=std::move(old_graph_scalars);
            graph_buffers_swapped=false;
        }
    };
    try
    {
        for(int test=0;test<3;++test)
        {
            std::vector<Float> values=rhs;
            if(test==1)std::fill(values.begin(),values.end(),0.0);
            if(test==2)for(size_t i=0;i<values.size();++i)
                values[i]=(static_cast<int>(i%17)-8)*0.125;
            CUDA_SAFE_CALL(cudaMemcpy(input.data(),values.data(),values.size()*sizeof(Float),cudaMemcpyHostToDevice));
            const cudatool::CDenseVectorView<Float> v{input.data(),static_cast<int>(values.size())};
            apply_preconditioner(z.view(),v);
            const double legacy_rho=My_PCG_General_v_v_Reduction_Algorithm(p.data(),input.data(),
                z.data(),reduction_result.data(),static_cast<int>(values.size()));
            std::vector<Float> reference(values.size());
            CUDA_SAFE_CALL(cudaMemcpy(reference.data(),z.data(),reference.size()*sizeof(Float),cudaMemcpyDeviceToHost));
            long double cpu_rho=0,sum_abs=0;
            for(size_t i=0;i<values.size();++i)
            {const long double product=static_cast<long double>(values[i])*reference[i];cpu_rho+=product;sum_abs+=std::abs(product);}
            const double tolerance=64*std::numeric_limits<double>::epsilon()*static_cast<double>(sum_abs)+1e-300;
            Json item={{"input",test==0?"actual_rhs":test==1?"zero_rhs":"signed_tail_pattern"},
                {"scalar_count",values.size()},{"legacy_rho",legacy_rho},
                {"cpu_long_double_rho",static_cast<double>(cpu_rho)},
                {"rho_roundoff_bound",tolerance},{"runs",Json::array()}};
            for(int mode=0;mode<3;++mode)
            {
                if(mode==0)apply_mas_dot(z.view(),v,reduction_result.data(),true);
                else
                {
                    if(!probe_exec)
                    {
                        CostCaptureGuard guard;
                        CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                        capturing=true;
                        apply_mas_dot(z.view(),v,reduction_result.data(),true);
                        CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&probe_graph));
                        capturing=false;
                        CUDA_SAFE_CALL(cudaGraphInstantiate(&probe_exec,probe_graph,nullptr,nullptr,0));
                    }
                    CUDA_SAFE_CALL(cudaGraphLaunch(probe_exec,cudaStreamPerThread));
                }
                double rho=0;
                std::vector<Float> actual(values.size());
                CUDA_SAFE_CALL(cudaMemcpy(&rho,reduction_result.data(),sizeof(rho),cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(actual.data(),z.data(),actual.size()*sizeof(Float),cudaMemcpyDeviceToHost));
                const bool exact=std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(Float))==0;
                const double error=std::abs(static_cast<double>(static_cast<long double>(rho)-cpu_rho));
                const bool valid=exact && std::isfinite(rho) && error<=tolerance;
                item["runs"].push_back({{"execution",mode==0?"host_launch":mode==1?"graph_first":"graph_replay"},
                    {"z_bitwise_equal",exact},{"rho",rho},{"rho_absolute_error",error},{"passed",valid}});
                if(!valid)study["passed"]=false;
            }
            study["cases"].push_back(item);
        }
        cleanup();
        // Exercise the real new initial-preconditioner/rho path, including
        // the existing host and conditional-PCG zero guards. The Graph zero
        // branch exits before any production graph invalidation/capture.
        CUDA_SAFE_CALL(cudaMemsetAsync(input.data(),0,input.size()*sizeof(Float),cudaStreamPerThread));
        const cudatool::CDenseVectorView<Float> zero{input.data(),static_cast<int>(input.size())};
        // Keep existing allocations (and production graph references) alive.
        // A host primary solve may not have any Graph scratch allocated yet.
        old_graph_storage=std::move(graph_reduce_storage);
        old_graph_scalars=std::move(graph_scalars);
        graph_buffers_swapped=true;
        mas_dot_effective=true;
        study["full_fused_PCG_zero_rhs"]=Json::array();
        for(int execution=0;execution<2;++execution)
        {
            x.buffer_view().fill(0);
            const auto iterations=execution==0?pcg(x,zero,10):pcg_graph(x,zero,10);
            bool finite_zero=true;
            for(const auto* data:{x.data(),z.data(),r.data()})
            {
                std::vector<Float> values(x.size());
                CUDA_SAFE_CALL(cudaMemcpy(values.data(),data,values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
                for(double value:values)finite_zero=finite_zero && std::isfinite(value) && value==0;
            }
            const bool valid=iterations==0 && finite_zero;
            study["full_fused_PCG_zero_rhs"].push_back({{"execution",execution==0?"host":"conditional_graph"},
                {"iterations",iterations},{"x_z_r_finite_zero",finite_zero},{"passed",valid}});
            if(!valid)study["passed"]=false;
        }
        mas_dot_effective=saved_effective;
        restore_graph_buffers();
        cost_trace_iteration(-1);
        restore();
    }
    catch(...)
    {
        if(capturing){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(cudaStreamPerThread,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}
        cleanup();mas_dot_effective=saved_effective;restore_graph_buffers();
        cost_trace_iteration(-1);restore();info=saved_info;throw;
    }
    bool restored=true;
    for(const auto& entry:saved)
    {
        std::vector<unsigned char> after(entry.bytes.size());
        CUDA_SAFE_CALL(cudaMemcpy(after.data(),entry.address,after.size(),cudaMemcpyDeviceToHost));
        restored=restored && after==entry.bytes;
    }
    std::vector<Float> rhs_after(rhs.size());
    CUDA_SAFE_CALL(cudaMemcpy(rhs_after.data(),b.data(),rhs_after.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    const auto after=snapshot_system("");
    const bool rhs_equal=std::memcmp(rhs.data(),rhs_after.data(),rhs.size()*sizeof(Float))==0;
    const bool graph_equal=signature==graph_signature() && saved_key==captured_key && saved_graph==graph && saved_exec==graph_exec;
    study["all_work_buffers_restored_bitwise"]=restored;
    study["rhs_unchanged_bitwise"]=rhs_equal;
    study["system_and_MAS_scratch_unchanged"]=before==after;
    study["production_graph_unchanged"]=graph_equal;
    study["passed"]=study["passed"].get<bool>() && restored && rhs_equal && before==after && graph_equal;
    info=saved_info;info["mas_fused_dot_study"]=study;
    info["fixed_study_file"]=prefix+"_mas_dot_study.json";
    std::ofstream file(prefix+"_mas_dot_study.json");file<<study.dump(2);file.close();
    if(!file || !study["passed"].get<bool>())throw std::runtime_error("Protected MAS final-z/rho study failed");
}
}
