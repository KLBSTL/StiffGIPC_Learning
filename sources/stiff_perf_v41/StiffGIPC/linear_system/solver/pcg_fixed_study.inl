// Diagnostics only. No reassembly or preconditioner change between solves.
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
    // Separate equal-work timing from native stopping and tighter residual probes.
    for(int phase=0;phase<4;++phase)
    {
        const double tolerances[]={1e-4,1e-4,1e-8,1e-12};
        m_config.global_tol_rate=tolerances[phase];
        diagnostic_fixed_iterations=phase==0?fixed:0;
        std::vector<Float> first_host;
        for(int mode=0;mode<(fused_diag_update_available()?3:2);++mode)
        {
            diagnostic_fused_override=mode==2?1:0;
            std::vector<Float> first;
            for(int repeat=-1;repeat<3;++repeat)
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
    const auto after=snapshot_system("");
    study["system_unchanged"]=before==after;
    CUDA_SAFE_CALL(cudaMemcpy(x.data(),primary.data(),primary.size()*sizeof(Float),cudaMemcpyHostToDevice));
    const auto restored=download();
    study["primary_restored_bitwise"]=std::memcmp(primary.data(),restored.data(),primary.size()*sizeof(Float))==0;
    std::ofstream file(prefix+"_study.json");file<<study.dump(2);file.close();
    info=saved_info;info["fixed_study_file"]=prefix+"_study.json";
    if(!study["system_unchanged"].get<bool>() || !study["primary_restored_bitwise"].get<bool>())
        throw std::runtime_error("Fixed-system study changed system or primary solution");
}
}
