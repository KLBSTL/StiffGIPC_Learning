// Diagnostics only. No reassembly; optional restriction arithmetic comparison
// preserves A/b/factors and changes summation order strictly between solves.
namespace gipc
{
void PCGSolver::fixed_system_study(cudatool::DenseVectorView<Float> x,
                                  cudatool::CDenseVectorView<Float> b,
                                  SizeT max_iter, SizeT primary_iter)
{
    const int frame=total_Frames+1;
    const int direction=Statistics::instance().at_current_frame()["newton"].size();
    auto selected=[](const char* variable,int value,const char* fallback)
    {
        const char* raw=std::getenv(variable);
        std::stringstream stream(raw?raw:fallback);std::string item;
        while(std::getline(stream,item,','))if(std::stoi(item)==value)return true;
        return false;
    };
    if(!selected("GIPC_FIXED_STUDY_FRAMES",frame,"1,22,40,70,100") ||
       !selected("GIPC_FIXED_STUDY_DIRECTIONS",direction,"1,8"))return;
    const std::string prefix=std::string(std::getenv("GIPC_FIXED_STUDY_DIR"))+"/f"+
        std::to_string(frame)+"_n"+std::to_string(direction);
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const auto saved_info=info;const auto saved_config=m_config;
    std::vector<Float> primary(x.size());
    CUDA_SAFE_CALL(cudaMemcpy(primary.data(),x.data(),primary.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    const auto before=snapshot_system(prefix);
    Json study={{"frame",frame},{"direction",direction},{"system",before},
                {"zero_initial_guess",true},{"runs",Json::array()}};
    attach_solve_context(study);
    if(const char* enabled=std::getenv("GIPC_FIXED_MAS_STAGE_STUDY");enabled&&std::string(enabled)=="1")
    {
        const auto saved_signature=graph_signature();
        std::vector<Float> saved_z(z.size());
        CUDA_SAFE_CALL(cudaMemcpy(saved_z.data(),z.data(),saved_z.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        {
            MasStageProbeRequest probe(prefix);
            apply_preconditioner(z.view(),b);
            if(!probe.dispatched)throw std::runtime_error("MAS stage probe was not dispatched");
            study["mas_stage_probe"]=probe.result;
        }
        CUDA_SAFE_CALL(cudaMemcpy(z.data(),saved_z.data(),saved_z.size()*sizeof(Float),cudaMemcpyHostToDevice));
        const auto after=snapshot_system("");
        study["stage_only"]=true;
        study["zero_initial_guess"]=nullptr; // no diagnostic PCG solve in this branch
        study["system_after"]=after;
        study["full_snapshot_unchanged_including_scratch"]=before==after;
        study["system_unchanged"]=snapshot_operator_identity(before)==snapshot_operator_identity(after);
        std::vector<Float> restored(x.size()),restored_z(z.size());
        CUDA_SAFE_CALL(cudaMemcpy(restored.data(),x.data(),restored.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(restored_z.data(),z.data(),restored_z.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        study["primary_restored_bitwise"]=std::memcmp(primary.data(),restored.data(),primary.size()*sizeof(Float))==0;
        study["z_restored_bitwise"]=std::memcmp(saved_z.data(),restored_z.data(),saved_z.size()*sizeof(Float))==0;
        study["graph_signature_restored"]=graph_signature()==saved_signature;
        info=saved_info;info["fixed_study_file"]=prefix+"_study.json";
        std::ofstream file(prefix+"_study.json");file<<study.dump(2);file.close();
        if(!file || !study["full_snapshot_unchanged_including_scratch"].get<bool>() ||
           !study["primary_restored_bitwise"].get<bool>() || !study["z_restored_bitwise"].get<bool>() ||
           !study["graph_signature_restored"].get<bool>())
            throw std::runtime_error("MAS stage probe did not restore production state");
        return;
    }
    auto download=[&]()
    {
        std::vector<Float> values(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(values.data(),x.data(),values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        return values;
    };
    auto difference=[](const std::vector<Float>& a,const std::vector<Float>& ref)
    {
        double d=0,n=0;size_t unequal=0;
        for(size_t i=0;i<a.size();++i){d+=(a[i]-ref[i])*(a[i]-ref[i]);n+=ref[i]*ref[i];unequal+=std::memcmp(&a[i],&ref[i],sizeof(Float))!=0;}
        return Json{{"relative",n>0?std::sqrt(d/n):std::sqrt(d)},{"bitwise_different",unequal}};
    };
    auto residual=[&]()
    {
        spmv(cudatool::CDenseVectorView<Float>{x.data(),static_cast<int>(x.size())},Ap.view());
        std::vector<Float> ax(b.size()),rhs(b.size());
        CUDA_SAFE_CALL(cudaMemcpy(ax.data(),Ap.data(),ax.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),b.data(),rhs.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        long double rr=0,bb=0;
        for(size_t i=0;i<ax.size();++i){long double d=static_cast<long double>(rhs[i])-ax[i];rr+=d*d;bb+=static_cast<long double>(rhs[i])*rhs[i];}
        return bb>0?std::sqrt(static_cast<double>(rr/bb)):std::sqrt(static_cast<double>(rr));
    };
    // Batch asynchronous operators with fixed b; include their normal launch costs.
    auto operator_cost=[&](auto operation)
    {
        for(int i=0;i<4;++i)operation();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<128;++i)operation();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/128;
    };
    study["operator_mean_us"]={{"spmv",operator_cost([&](){spmv(b,Ap.view());})},
        {"preconditioner",operator_cost([&](){apply_preconditioner(z.view(),b);})}};
    auto repeat_operator=[&](auto operation,auto output)
    {
        operation();std::vector<Float> first(output.size()),again(output.size());
        CUDA_SAFE_CALL(cudaMemcpy(first.data(),output.data(),first.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        Json records=Json::array();
        for(int repeat=0;repeat<3;++repeat)
        {
            operation();CUDA_SAFE_CALL(cudaMemcpy(again.data(),output.data(),again.size()*sizeof(Float),cudaMemcpyDeviceToHost));
            records.push_back(difference(again,first));
        }
        return records;
    };
    study["operator_repeats"]={{"spmv",repeat_operator([&](){spmv(b,Ap.view());},Ap.view())},
        {"preconditioner",repeat_operator([&](){apply_preconditioner(z.view(),b);},z.view())}};
    const int fixed=std::max(1,std::min(64,static_cast<int>(primary_iter)));
    const bool compact=std::getenv("GIPC_FIXED_STUDY_COMPACT") && std::string(std::getenv("GIPC_FIXED_STUDY_COMPACT"))=="1";
    study["compact_default_strict_comparison"]=compact;
    // Separate equal-work timing from native stopping and tighter residual probes.
    for(int phase=0;phase<4;++phase)
    {
        if(compact && phase!=1 && phase!=3)continue;
        const double tolerances[]={1e-4,1e-4,1e-8,compact?1e-16:1e-12};
        m_config.global_tol_rate=tolerances[phase];
        diagnostic_fixed_iterations=phase==0?fixed:0;
        std::vector<Float> first_host;
        for(int mode=0;mode<(fused_diag_update_available()?3:2);++mode)
        {
            diagnostic_fused_override=mode==2?1:0;
            std::vector<Float> first;
            for(int repeat=compact?0:-1;repeat<(compact?2:3);++repeat)
            {
                x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const auto start=std::chrono::steady_clock::now();
                SizeT iterations=0;std::string error;
                try{iterations=mode==0?pcg(x,b,max_iter):pcg_graph(x,b,max_iter);}
                catch(const std::exception& e){error=e.what();}
                CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                if(repeat<0)continue; // graph capture/warmup excluded from steady solve timing
                auto values=download();if(repeat==0)first=values;
                if(mode==0 && repeat==0)first_host=values;
                const double true_residual=residual();
                Json record={{"phase",phase==0?"fixed_iterations":"rho_tolerance"},
                    {"rho_tolerance",m_config.global_tol_rate},{"fixed_iterations",diagnostic_fixed_iterations},
                    {"mode",mode==0?"host":mode==1?"graph":"graph_fused"},{"repeat",repeat+1},
                    {"iterations",iterations},{"limit",iterations>=max_iter},{"error",error},{"solve_ms",ms},
                    {"true_relative_residual",true_residual},{"repeat_difference",difference(values,first)},
                    {"host_difference",difference(values,first_host)}};
                if(mode>0)record["graph_cache_hit"]=info.value("graph_cache_hit",false);
                study["runs"].push_back(record);
                // Keep the first solution of each arm/precision for independent CPU residual checks.
                if(repeat==0)
                {
                    std::ofstream file(prefix+"_p"+std::to_string(phase)+"_m"+std::to_string(mode)+"_x.bin",std::ios::binary);
                    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
                }
            }
        }
    }
    diagnostic_fixed_iterations=0;diagnostic_fused_override=-1;m_config=saved_config;
    if(const char* enabled=std::getenv("GIPC_FIXED_RESTRICT_STUDY");enabled&&std::string(enabled)=="1")
    {
        study["restriction_pairs"]=Json::array();
        study["restriction_pair_scope"]="Same A/b/Cholesky/CSR, default rho; 3 interleaved pairs per execution mode; one warmup per arm; prepare cost is common and excluded";
        for(int mode=0;mode<2;++mode)
        {
            std::vector<Float> reference;
            for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order)
            {
                const bool warp=(order+(pair%2))%2!=0;
                MasRestrictionStudyArm arm(warp);
                auto solve=[&](){return mode==0?pcg(x,b,max_iter):pcg_graph(x,b,max_iter);};
                x.buffer_view().fill(0);solve();CUDA_SAFE_CALL(cudaDeviceSynchronize());
                x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const auto start=std::chrono::steady_clock::now();
                const auto iterations=solve();CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                const auto values=download();
                if(pair==0&&!warp)reference=values;
                const double actual_residual=residual();
                study["restriction_pairs"].push_back({{"mode",mode==0?"host":"graph"},
                    {"pair",pair+1},{"restriction",warp?"warp":"serial"},{"rho_tolerance",m_config.global_tol_rate},
                    {"iterations",iterations},{"solve_ms",elapsed},{"true_relative_residual",actual_residual},
                    {"serial_difference",difference(values,reference)},
                    {"rho_initial",info.value("rho_initial",0.0)},{"rho_stop",info.value("rho_stop",0.0)}});
                if(pair==0)
                {
                    std::ofstream file(prefix+"_restrict_"+(warp?"warp":"serial")+"_m"+std::to_string(mode)+"_x.bin",std::ios::binary);
                    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
                }
                if(iterations>=max_iter||!std::isfinite(actual_residual))
                    throw std::runtime_error("Fixed restriction study did not converge");
            }
        }
    }
    if(mas_factor_action_study_enabled())
    {
        study["factor_action_pairs"]=Json::array();
        study["factor_action_pair_scope"]="Same A/b/L/CSR, default rho; 3 interleaved pairs in host/Graph. B is precomputed before the study; its extra preparation cost is measured separately, not hidden in a speedup claim.";
        for(int mode=0;mode<2;++mode)
        {
            std::vector<Float> reference;
            for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order)
            {
                const bool inverse=(order+(pair%2))%2!=0;
                MasFactorActionStudyArm arm(inverse);
                auto solve=[&](){return mode==0?pcg(x,b,max_iter):pcg_graph(x,b,max_iter);};
                x.buffer_view().fill(0);solve();CUDA_SAFE_CALL(cudaDeviceSynchronize());
                x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const auto start=std::chrono::steady_clock::now();
                const auto iterations=solve();CUDA_SAFE_CALL(cudaDeviceSynchronize());
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                const auto values=download();
                if(pair==0&&!inverse)reference=values;
                const double actual_residual=residual();
                study["factor_action_pairs"].push_back({{"mode",mode==0?"host":"graph"},
                    {"pair",pair+1},{"factor_action",inverse?"factor_inverse":"triangular"},
                    {"rho_tolerance",m_config.global_tol_rate},{"iterations",iterations},
                    {"solve_ms",elapsed},{"true_relative_residual",actual_residual},
                    {"triangular_difference",difference(values,reference)},
                    {"rho_initial",info.value("rho_initial",0.0)},{"rho_stop",info.value("rho_stop",0.0)}});
                std::ofstream file(prefix+"_factor_"+(inverse?"inverse":"triangular")+"_m"+
                    std::to_string(mode)+"_r"+std::to_string(pair+1)+"_x.bin",std::ios::binary);
                file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
                if(iterations>=max_iter||!std::isfinite(actual_residual))
                    throw std::runtime_error("Fixed factor action study did not converge");
            }
        }
    }
    const auto after=snapshot_system("");
    study["system_after"]=after;
    study["full_snapshot_unchanged_including_scratch"]=before==after;
    study["operator_equality_excludes_scratch"]={"d_multiLevelR","d_multiLevelZ","d_multiLevelR64","d_multiLevelZ64"};
    study["system_unchanged"]=snapshot_operator_identity(before)==snapshot_operator_identity(after);
    CUDA_SAFE_CALL(cudaMemcpy(x.data(),primary.data(),primary.size()*sizeof(Float),cudaMemcpyHostToDevice));
    const auto restored=download();
    study["primary_restored_bitwise"]=std::memcmp(primary.data(),restored.data(),primary.size()*sizeof(Float))==0;
    std::ofstream file(prefix+"_study.json");file<<study.dump(2);file.close();
    info=saved_info;info["fixed_study_file"]=prefix+"_study.json";
    if(!study["system_unchanged"].get<bool>() || !study["primary_restored_bitwise"].get<bool>())
        throw std::runtime_error("Fixed-system study changed system or primary solution");
}
}
