namespace gipc {
void PCGSolver::divergence_study(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,SizeT max_iter,const std::string& prefix)
{
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const auto original_info=info;const auto original_config=m_config;
    const auto before=snapshot_system(prefix);
    std::vector<double> primary(x.size()),rhs(b.size());
    CUDA_SAFE_CALL(cudaMemcpy(primary.data(),x.data(),primary.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),b.data(),rhs.size()*sizeof(double),cudaMemcpyDeviceToHost));
    Json report={{"system",before},{"n",b.size()},{"runs",Json::array()},
        {"stages",{"initial_r_z_p","p_Ap_r_after_dot","x_r_p_after_update",
                   "r_z_x_after_preconditioner","p_z_r_after_direction_update"}},
        {"trace_layout","[iteration,stage,3*n+8]; unwritten slots are NaN; scalars k,stage,rho,pAp,alpha,beta,rho_new,valid"}};
    attach_solve_context(report);
    std::vector<int> counts;
    const char* values=std::getenv("GIPC_DIVERGENCE_COUNTS");
    std::stringstream stream(values?values:"1,2,4,8,16,32,64,0");std::string item;
    while(std::getline(stream,item,','))counts.push_back(std::stoi(item));
    const int repeats=std::getenv("GIPC_DIVERGENCE_REPEATS")?
        std::clamp(std::atoi(std::getenv("GIPC_DIVERGENCE_REPEATS")),2,10):5;
    for(int count:counts) {
        diagnostic_fixed_iterations=count;
        m_config.global_tol_rate=1e-4;m_config.graph_chunk_iterations=1;
        diagnostic_fused_override=0;
        for(int repeat=-1;repeat<repeats;++repeat)for(int order=0;order<2;++order) {
            const int mode=(order+std::max(0,repeat))%2;
            divergence_trace_enabled=count==0&&repeat>=0;
            x.buffer_view().fill(0);CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const auto begin=std::chrono::steady_clock::now();
            const SizeT iterations=mode?pcg_graph(x,b,max_iter):pcg(x,b,max_iter);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if(repeat<0)continue;
            const std::string tag=prefix+"_k"+std::to_string(count)+"_m"+std::to_string(mode)+"_r"+std::to_string(repeat+1);
            std::vector<double> solution(x.size()),ax(x.size());
            CUDA_SAFE_CALL(cudaMemcpy(solution.data(),x.data(),solution.size()*sizeof(double),cudaMemcpyDeviceToHost));
            divergence_trace_save(tag+"_trace.bin",b.size());
            spmv(cudatool::CDenseVectorView<Float>{x.data(),static_cast<int>(x.size())},Ap.view());
            CUDA_SAFE_CALL(cudaMemcpy(ax.data(),Ap.data(),ax.size()*sizeof(double),cudaMemcpyDeviceToHost));
            long double rr=0,bb=0;
            for(size_t i=0;i<rhs.size();++i){long double d=rhs[i]-ax[i];rr+=d*d;bb+=static_cast<long double>(rhs[i])*rhs[i];}
            std::ofstream out(tag+"_x.bin",std::ios::binary);
            out.write(reinterpret_cast<const char*>(solution.data()),solution.size()*sizeof(double));out.close();
            if(!out)throw std::runtime_error("Cannot write divergence solution");
            report["runs"].push_back({{"count",count},{"mode",mode?"graph":"host"},{"repeat",repeat+1},
                {"iterations",iterations},{"ms",ms},{"true_relative_residual",std::sqrt(double(rr/std::max(bb,1e-300L)))},
                {"rho_initial",info.value("rho_initial",0.)},{"rho_stop",info.value("rho_stop",0.)},
                {"file",tag},{"trace_steps",divergence_trace_enabled?divergence_trace_steps:0}});
        }
    }
    divergence_trace_enabled=false;diagnostic_fixed_iterations=0;diagnostic_fused_override=-1;
    m_config=original_config;release_graph();
    const auto after=snapshot_system("");
    report["operator_unchanged"]=snapshot_operator_identity(before)==snapshot_operator_identity(after);
    std::vector<double> rhs_after(b.size());
    CUDA_SAFE_CALL(cudaMemcpy(rhs_after.data(),b.data(),rhs_after.size()*sizeof(double),cudaMemcpyDeviceToHost));
    report["rhs_unchanged_bitwise"]=std::memcmp(rhs.data(),rhs_after.data(),rhs.size()*sizeof(double))==0;
    CUDA_SAFE_CALL(cudaMemcpy(x.data(),primary.data(),primary.size()*sizeof(double),cudaMemcpyHostToDevice));
    std::vector<double> restored(x.size());
    CUDA_SAFE_CALL(cudaMemcpy(restored.data(),x.data(),restored.size()*sizeof(double),cudaMemcpyDeviceToHost));
    report["primary_restored_bitwise"]=std::memcmp(restored.data(),primary.data(),primary.size()*sizeof(double))==0;
    info=original_info;
    std::ofstream out(prefix+"_divergence.json");out<<report.dump(2);out.close();
    if(!out || !report["operator_unchanged"].get<bool>() || !report["rhs_unchanged_bitwise"].get<bool>() ||
       !report["primary_restored_bitwise"].get<bool>())throw std::runtime_error("Divergence probe modified primary system");
}
}
