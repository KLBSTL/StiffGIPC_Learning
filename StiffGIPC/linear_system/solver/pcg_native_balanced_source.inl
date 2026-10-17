// Included inside native_history_study's protected try block, n1 only.
for(int order=0;order<2;++order) {
    const bool capture=order==1;
    const auto reset_start=Clock::now();restore_m();
    check(cudaMemsetAsync(work_x.data(),0,size_t(n)*sizeof(Float),cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));info=Json::object();
    const double reset_ms=milliseconds(reset_start);
    NativePcgHistory sink([&](int slot) {
        check(cudaMemcpyAsync(history_bank.data()+size_t(slot)*n,work_x.data(),size_t(n)*sizeof(Float),
                              cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    });
    diagnostic_history=capture?&sink:nullptr;
    const auto solve_start=Clock::now();check(cudaEventRecord(events.begin,cudaStreamPerThread));
    const auto iterations=pcg(work_x.view(),b,max_iter);diagnostic_history=nullptr;
    check(cudaEventRecord(events.end,cudaStreamPerThread));check(cudaEventSynchronize(events.end));
    const double solve_ms=milliseconds(solve_start);float event_ms=0;
    check(cudaEventElapsedTime(&event_ms,events.begin,events.end));
    const auto export_start=Clock::now();const auto solution=read(work_x.data(),n);
    spmv(work_x.cview(),residual_ax.view());const auto ax=read(residual_ax.data(),n);
    long double rr=0;bool finite=true;
    for(int i=0;i<n;++i){finite=finite&&std::isfinite(solution[i])&&std::isfinite(ax[i]);
        const long double t=static_cast<long double>(rhs[i])-ax[i];rr+=t*t;}
    const double residual=std::sqrt(static_cast<double>(rr/rhs_norm2));
    Json row={{"pair",1},{"order",order+1},{"capture",capture},{"iterations",iterations},
        {"solve_host_ms",solve_ms},{"solve_event_ms",event_ms},{"reset_host_ms",reset_ms},{"pcg",info},
        {"true_relative_residual",std::isfinite(residual)?Json(residual):Json(nullptr)}};
    const std::string stem=prefix+(capture?"_source_capture":"_source_control");
    auto save=[&](const std::string& suffix,const void* data,size_t bytes) {
        const std::string path=stem+suffix;std::ifstream existing(path,std::ios::binary);
        if(existing.good())throw std::runtime_error("Native balanced output exists");
        std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(data),bytes);file.close();
        if(!file)throw std::runtime_error("Native balanced source write failed");return path;
    };
    row["solution_file"]=save("_x.bin",solution.data(),solution.size()*sizeof(Float));
    std::vector<int> indices,slots;
    if(capture) {
        if(sink.rows.size()!=static_cast<size_t>(iterations))throw std::runtime_error("Missing source scalars");
        std::vector<std::pair<int,int>> retained;
        for(int i=0;i<NativePcgHistory::capacity;++i)if(sink.slot_iterations[i]>=0)
            retained.emplace_back(sink.slot_iterations[i],i);
        std::sort(retained.begin(),retained.end());std::vector<Float> bank;bank.reserve(size_t(n)*retained.size());
        for(const auto& item:retained) {
            indices.push_back(item.first);slots.push_back(item.second);
            auto column=read(history_bank.data()+size_t(item.second)*n,n);bank.insert(bank.end(),column.begin(),column.end());
        }
        std::vector<double> scalar;scalar.reserve(3*sink.rows.size());
        for(const auto& values:sink.rows)scalar.insert(scalar.end(),values.begin(),values.end());
        row["history_file"]=save("_history.bin",bank.data(),bank.size()*sizeof(Float));
        row["indices_file"]=save("_indices.bin",indices.data(),indices.size()*sizeof(int));
        row["scalars_file"]=save("_scalars.bin",scalar.data(),scalar.size()*sizeof(double));
        row["history_width"]=indices.size();
    }
    row["export_and_residual_host_ms"]=milliseconds(export_start);
    row["passed"]=finite&&std::isfinite(residual)&&iterations<max_iter&&!info.contains("breakdown");
    study["runs"].push_back(row);
    if(!row["passed"].get<bool>())throw std::runtime_error("Native balanced source solve failed");
    if(capture) {
        const auto build_start=Clock::now();
        try {
            const auto eig_start=Clock::now();const auto U=native_stat_coefficients(indices,sink.rows);
            study["source_eigen_host_ms"]=milliseconds(eig_start);
            const auto allocation_start=Clock::now();pending_space=std::make_unique<NativeBalancedSpace>(n);
            study["source_BLAS_and_allocation_host_ms"]=milliseconds(allocation_start);
            const auto projection_start=Clock::now();pending_space->source(history_bank.data(),slots,work_x.data(),U);
            study["source_Z_formation_host_ms"]=milliseconds(projection_start);
            study["source_build_host_ms"]=milliseconds(build_start);study["source_space_built"]=true;
            const auto basis_export_start=Clock::now();const auto basis=read(pending_space->Z.value,size_t(n)*8);
            for(double v:basis)if(!std::isfinite(v))throw std::runtime_error("Nonfinite source Z");
            study["source_Z_file"]=save("_Z.bin",basis.data(),basis.size()*sizeof(double));
            study["source_basis_identity"]=digest(pending_space->Z.value,size_t(n)*8*sizeof(double));
            study["source_Z_export_host_ms"]=milliseconds(basis_export_start);
        } catch(const NativeBalancedRejected& error) {
            pending_space.reset();pending_rejection=error.what();study["source_space_built"]=false;
            study["source_rejection"]=pending_rejection;study["source_build_host_ms"]=milliseconds(build_start);
        }
        if(inject_failure)throw std::runtime_error("Native history injected failure after captured solve");
    }
}
