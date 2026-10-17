// Included inside the protected native_history_study try block.
for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order)
{
    const bool capture=((pair+order)%2)!=0;
    const auto reset_start=Clock::now();
    restore_m();check(cudaMemsetAsync(work_x.data(),0,n*sizeof(Float),cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));info=Json::object();
    const double reset_ms=milliseconds(reset_start);
    NativePcgHistory sink([&](int slot) {
        check(cudaMemcpyAsync(history_bank.data()+size_t(slot)*n,work_x.data(),
                              size_t(n)*sizeof(Float),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    });
    diagnostic_history=capture?&sink:nullptr;
    check(cudaEventRecord(events.begin,cudaStreamPerThread));
    const auto solve_start=Clock::now();
    const auto iterations=pcg(work_x.view(),b,max_iter);
    diagnostic_history=nullptr;
    check(cudaEventRecord(events.end,cudaStreamPerThread));check(cudaEventSynchronize(events.end));
    const double solve_ms=milliseconds(solve_start);
    float event_ms=0;check(cudaEventElapsedTime(&event_ms,events.begin,events.end));
    const auto export_start=Clock::now();
    const auto solution=read(work_x.data(),work_x.size());
    spmv(work_x.cview(),residual_ax.view());const auto ax=read(residual_ax.data(),residual_ax.size());
    long double rr=0;bool finite=true;
    for(size_t i=0;i<solution.size();++i) {
        finite=finite&&std::isfinite(solution[i])&&std::isfinite(ax[i]);
        const long double error=static_cast<long double>(rhs[i])-ax[i];rr+=error*error;
    }
    const double residual=std::sqrt(static_cast<double>(rr/rhs_norm2));
    Json row={{"pair",pair+1},{"order",order+1},{"capture",capture},{"iterations",iterations},
        {"solve_host_ms",solve_ms},{"solve_event_ms",event_ms},{"reset_host_ms",reset_ms},
        {"pcg",info},{"true_relative_residual",std::isfinite(residual)?Json(residual):Json(nullptr)}};
    const std::string stem=prefix+"_host_p"+std::to_string(pair+1)+(capture?"_capture":"_control");
    auto save=[&](const std::string& suffix,const void* data,size_t bytes) {
        const std::string path=stem+suffix;
        std::ifstream existing(path,std::ios::binary);
        if(existing.good())throw std::runtime_error("Native history output already exists");
        std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(data),bytes);file.close();
        if(!file)throw std::runtime_error("Native history output write failed");return path;
    };
    row["solution_file"]=save("_x.bin",solution.data(),solution.size()*sizeof(Float));
    if(capture) {
        if(sink.rows.size()!=static_cast<size_t>(iterations))throw std::runtime_error("Native history missing scalar rows");
        std::vector<std::pair<int,int>> slots;
        for(int i=0;i<NativePcgHistory::capacity;++i)if(sink.slot_iterations[i]>=0)slots.emplace_back(sink.slot_iterations[i],i);
        std::sort(slots.begin(),slots.end());std::vector<int> indices;std::vector<Float> bank;
        bank.reserve(size_t(n)*slots.size());indices.reserve(slots.size());
        for(const auto& entry:slots) {
            indices.push_back(entry.first);auto column=read(history_bank.data()+size_t(entry.second)*n,n);
            bank.insert(bank.end(),column.begin(),column.end());
        }
        std::vector<double> scalars;scalars.reserve(3*sink.rows.size());
        for(const auto& values:sink.rows)scalars.insert(scalars.end(),values.begin(),values.end());
        row["history_file"]=save("_history.bin",bank.data(),bank.size()*sizeof(Float));
        row["indices_file"]=save("_indices.bin",indices.data(),indices.size()*sizeof(int));
        row["scalars_file"]=save("_scalars.bin",scalars.data(),scalars.size()*sizeof(double));
        row["history_width"]=indices.size();row["scalar_columns"]={"alpha","pre_update_rho","p_Ap"};
        row["history_layout"]="column-major full coordinates, pre-update x, zero-based indices";
    }
    row["export_and_residual_host_ms"]=milliseconds(export_start);
    const bool passed=finite&&std::isfinite(residual)&&iterations<max_iter
        &&!info.contains("breakdown")&&!info.value("iteration_limit",false);
    row["passed"]=passed;study["runs"].push_back(row);
    if(!passed)throw std::runtime_error("Native history solve/finite contract failed");
    if(capture&&inject_failure)throw std::runtime_error("Native history injected failure after captured solve");
}
