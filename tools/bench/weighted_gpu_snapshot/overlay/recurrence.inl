gipc::research::WeightedCheckSchedule policy_schedule;
auto quality_probe=[&](Json& point,Json& output) {
    const auto qstart=Clock::now();multiply(work_x.value,residual_ax.value);
    costs.call("explicit_residual_vector",[&]{audit_residual<<<blocks,256,0,cudaStreamPerThread>>>(quality_r.value,b.value,residual_ax.value,n);
    check(cudaGetLastError());});base(quality_r.value,quality_z.value);
    const double rr=dot(quality_r.value,quality_r.value),qm=dot(quality_r.value,quality_z.value);
    balanced_require(std::isfinite(rr)&&rr>=0&&std::isfinite(qm)&&qm>=0,"Native diagnostic invalid true quadratic");
    const double l2=std::sqrt(rr/static_cast<double>(rhs_norm2)),weighted=std::sqrt(qm/common_bMb);
    #include "quality_event.inl"
    point["quality_checked"]=true;point["true_relative_l2"]=l2;point["common_native_MAS_relative"]=weighted;
    output["quality_host_ms"]=output["quality_host_ms"].get<double>()+milliseconds(qstart);
    output["quality_checks"]=output["quality_checks"].get<int>()+1;
    output["last_quality_iteration"]=point.at("iteration");
    if(l2<=.0095&&weighted<=.0095) {
        output["converged"]=true;output["true_relative_l2"]=l2;output["common_native_MAS_relative"]=weighted;
        return true;
    }
    return false;
};
auto solve_candidate=[&](bool augmented,bool screened,bool fused,Json& output) {
    output={{"augmented",augmented},{"converged",false},{"trace",Json::array()},
        {"quality_checks",0},{"quality_host_ms",0.},{"residual_restarts",0},{"preconditioner_calls",0},
        {"screened",screened},{"fused_apply",fused},{"screen_checks",0},{"screen_host_ms",0.},{"last_quality_iteration",0}};
    policy_schedule=gipc::research::WeightedCheckSchedule{};
    output["policy"]=current_cooldown?"cooldown2":"legacy";output["quality_events"]=Json::array();
    output["schedule_queries"]=0;output["rhs_norm2"]=static_cast<double>(rhs_norm2);
    costs.call("copy_rhs",[&]{check(cudaMemcpyAsync(r.value,b.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));});
    auto action=[&]() {
        output["preconditioner_calls"]=output["preconditioner_calls"].get<int>()+1;
        op(current_method);
        if(augmented&&balanced_fault=="after_apply") {
            check(cudaStreamSynchronize(cudaStreamPerThread));
            throw std::runtime_error("Native balanced injected failure after apply");
        }
        if(augmented&&fused&&balanced_fault=="after_fused_apply") {
            check(cudaStreamSynchronize(cudaStreamPerThread));
            throw std::runtime_error("Native balanced injected failure after fused apply");
        }
    };
    action();double rho=dot(r.value,z.value),rho0=rho;
    balanced_require(std::isfinite(rho)&&rho>0,"Native diagnostic nonpositive initial rho");
    output["rho_initial"]=rho;costs.call("copy_initial_direction",[&]{check(cudaMemcpyAsync(p.value,z.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));});bool quality_active=false;
    const int limit=std::min<int>(int(max_iter)-1,2000);output["iteration_limit"]=limit;
    for(int iteration=1;iteration<=limit;++iteration) {
        multiply(p.value,Ap.value);const double curvature=dot(p.value,Ap.value);
        balanced_require(std::isfinite(curvature)&&curvature>0,"Native diagnostic nonpositive curvature");
        const double alpha=rho/curvature;
        balanced_require(std::isfinite(alpha)&&alpha>0,"Native diagnostic invalid alpha");
        costs.call("vector_update_x_r",[&]{update_vector_dx_r<<<blocks,256,0,cudaStreamPerThread>>>(work_x.value,r.value,p.value,Ap.value,alpha,n);
        check(cudaGetLastError());});
        Json point={{"iteration",iteration},{"pre_update_rho",rho},{"p_Ap",curvature},{"alpha",alpha},
            {"quality_checked",false},{"true_relative_l2",nullptr},{"common_native_MAS_relative",nullptr},
            {"recursive_relative_l2",nullptr},{"recursive_norm_rr",nullptr},{"recursive_norm_evaluated",false},
            {"policy_cooldown_before",current_cooldown&&policy_schedule.cooling_down()},{"policy_cooldown_after",current_cooldown&&policy_schedule.cooling_down()},{"schedule_due",false},{"screened_out",false},{"forced_quality_check",false}};
        quality_active=quality_active||rho<=1e-4*rho0;
        if(quality_active||iteration==limit) {
            bool due=true;
            if(screened) {
                #include "screen_block.inl"
            }
            point["schedule_due"]=due;
            if(due&&quality_probe(point,output)) {
                output["trace"].push_back(point);output["converged"]=true;output["iterations"]=iteration;
                return;
            }
        }
        output["trace"].push_back(point);output["iterations"]=iteration;
        action();const double next=dot(r.value,z.value);
        if(next==0) {
            output["zero_rho_true_quality_checked"]=true;
            output["trace"].back()["screened_out"]=false;
            output["trace"].back()["forced_quality_check"]=true;
            if(quality_probe(output["trace"].back(),output))return;
        }
        balanced_require(std::isfinite(next)&&next>0,"Native diagnostic invalid next rho");
        const double beta=next/rho;balanced_require(std::isfinite(beta),"Native diagnostic invalid beta");
        costs.call("vector_update_direction",[&]{update_vector_c<<<blocks,256,0,cudaStreamPerThread>>>(p.value,z.value,beta,n);
        check(cudaGetLastError());});rho=next;
    }
    throw NativeBalancedRejected("Native diagnostic iteration budget reached");
};
