// n2 only. Both arms have identical explicit quality checks after a rho candidate.
const char* fault_raw=std::getenv("GIPC_NATIVE_BALANCED_FAULT");
const std::string balanced_fault=fault_raw?fault_raw:"none";
if(balanced_fault!="none"&&balanced_fault!="coarse_reject"&&balanced_fault!="after_apply"&&balanced_fault!="after_fused_apply")
    throw std::runtime_error("Invalid GIPC_NATIVE_BALANCED_FAULT");
study["fault_selector"]=balanced_fault;
const char* quality_raw=std::getenv("GIPC_NATIVE_BALANCED_QUALITY_STUDY");
if(quality_raw&&std::string(quality_raw)!="0"&&std::string(quality_raw)!="1")
    throw std::runtime_error("GIPC_NATIVE_BALANCED_QUALITY_STUDY must be 0 or 1");
const bool quality_study=quality_raw&&std::string(quality_raw)=="1";
const char* apply_raw=std::getenv("GIPC_NATIVE_BALANCED_APPLY_STUDY");
if(apply_raw&&std::string(apply_raw)!="0"&&std::string(apply_raw)!="1")
    throw std::runtime_error("GIPC_NATIVE_BALANCED_APPLY_STUDY must be 0 or 1");
const bool apply_study=apply_raw&&std::string(apply_raw)=="1";
if(apply_study&&!quality_study)throw std::runtime_error("Apply study requires quality study selector");
if(balanced_fault=="after_fused_apply"&&!apply_study)throw std::runtime_error("Fused fault requires apply study");
using ApplyMode=NativeBalancedSpace::ApplyMode;
study["quality_frequency_study"]=quality_study;study["fused_apply_study"]=apply_study;
study["expected_runs"]=apply_study?9:(quality_study?12:6);
study["recursive_screen_tolerance"]=.012;study["maximum_quality_check_interval"]=8;
auto* space=balanced_cache.space.get();bool coarse_ready=false;
std::string rejection=balanced_cache.rejection;
auto base=[&](const double* input,double* output) {
    apply_preconditioner(cudatool::DenseVectorView<Float>{output,n},cudatool::CDenseVectorView<Float>{input,n});
};
auto multiply=[&](const double* input,double* output) {
    spmv(cudatool::CDenseVectorView<Float>{input,n},cudatool::DenseVectorView<Float>{output,n});
};
auto save=[&](const std::string& suffix,const void* data,size_t bytes) {
    const std::string path=prefix+suffix;std::ifstream existing(path,std::ios::binary);
    if(existing.good())throw std::runtime_error("Native balanced output exists");
    std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(data),bytes);file.close();
    if(!file)throw std::runtime_error("Native balanced target write failed");return path;
};
const auto coarse_start=Clock::now();
if(space) {
    try {
        if(balanced_fault=="coarse_reject")throw NativeBalancedRejected("Injected coarse rank rejection");
        space->target(multiply);coarse_ready=true;
    } catch(const NativeBalancedRejected& error){rejection=error.what();}
}
study["target_coarse_host_ms"]=milliseconds(coarse_start);
study["coarse_ready"]=coarse_ready;study["coarse_rejection"]=rejection;
if(coarse_ready) {
    study["coarse_condition"]=space->condition;const auto t=Clock::now();
    for(const auto& item:std::initializer_list<std::pair<const char*,const double*>>{
            {"Z",space->Z.value},{"W",space->W.value},{"L",space->L.value}}) {
        const size_t count=std::string(item.first)=="L"?64:size_t(n)*8;
        const auto values=read(item.second,count);
        for(double v:values)if(!std::isfinite(v))throw std::runtime_error("Nonfinite coarse export");
        study[std::string("target_")+item.first+"_file"]=save(std::string("_target_")+item.first+".bin",values.data(),count*sizeof(double));
    }
    study["target_E_file"]=save("_target_E.bin",space->host_E.data(),64*sizeof(double));
    study["coarse_export_host_ms"]=milliseconds(t);
    if(quality_study) {
        const auto verify_start=Clock::now();
        const auto initial_W=read(space->W.value,size_t(n)*8),initial_L=read(space->L.value,64);
        const auto initial_E=space->host_E;const double initial_condition=space->condition;
        const auto before_W=digest(space->W.value,size_t(n)*8*sizeof(double));
        const auto before_E=digest(space->E.value,64*sizeof(double));
        const auto before_L=digest(space->L.value,64*sizeof(double));
        Json samples=Json::array();double warm_sum=0;
        for(int repeat=0;repeat<3;++repeat) {
            const auto warm_start=Clock::now();space->target(multiply);const double warm_ms=milliseconds(warm_start);
            warm_sum+=warm_ms;const auto W=read(space->W.value,size_t(n)*8);
            long double error=0,norm=0;
            for(size_t i=0;i<W.size();++i){const long double d=static_cast<long double>(W[i])-initial_W[i];error+=d*d;norm+=static_cast<long double>(initial_W[i])*initial_W[i];}
            const double w_relative=std::sqrt(static_cast<double>(error/std::max(norm,1e-300L)));
            const double e_relative=(space->host_E-initial_E).norm()/std::max(initial_E.norm(),1e-300);
            if(!std::isfinite(w_relative)||!std::isfinite(e_relative)||w_relative>1e-10||e_relative>1e-10)
                throw std::runtime_error("Warm coarse calibration changed W/E materially");
            samples.push_back({{"repeat",repeat+1},{"host_ms",warm_ms},{"W_relative",w_relative},{"E_relative",e_relative}});
        }
        check(cudaMemcpyAsync(space->W.value,initial_W.data(),initial_W.size()*sizeof(double),cudaMemcpyHostToDevice,cudaStreamPerThread));
        check(cudaMemcpyAsync(space->E.value,initial_E.data(),64*sizeof(double),cudaMemcpyHostToDevice,cudaStreamPerThread));
        check(cudaMemcpyAsync(space->L.value,initial_L.data(),64*sizeof(double),cudaMemcpyHostToDevice,cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));space->host_E=initial_E;space->condition=initial_condition;
        const bool exact=before_W==digest(space->W.value,size_t(n)*8*sizeof(double))
            &&before_E==digest(space->E.value,64*sizeof(double))&&before_L==digest(space->L.value,64*sizeof(double));
        study["coarse_warm_calibration"]=samples;study["coarse_restored_bitwise"]=exact;
        study["coarse_warm_verification_restore_ms"]=milliseconds(verify_start)-warm_sum;
        if(!exact)throw std::runtime_error("Coarse calibration exact restoration failed");
    }
}
DeviceDenseVector quality_r(n),quality_z(n);
BalancedBuffer<double> dot_partials(blocks);
auto dot=[&](const double* a,const double* c) {
    return My_PCG_General_v_v_Reduction_Algorithm(dot_partials.value,a,c,reduction_result.data(),n);
};
restore_m();base(b.data(),quality_z.data());
const double common_bMb=dot(b.data(),quality_z.data());
balanced_require(std::isfinite(common_bMb)&&common_bMb>0,"Native reference RHS quadratic invalid");
study["common_native_bMb"]=common_bMb;
study["quality_rule"]="after previous-rho candidate: original-A true L2 and original native-MAS quadratic relative <= .0095; both arms identical";
study["quality_checks_counted_in_solve"]=true;
auto quality_probe=[&](Json& point,Json& output) {
    const auto qstart=Clock::now();multiply(work_x.data(),residual_ax.data());
    audit_residual<<<blocks,256,0,cudaStreamPerThread>>>(quality_r.data(),b.data(),residual_ax.data(),n);
    check(cudaGetLastError());base(quality_r.data(),quality_z.data());
    const double rr=dot(quality_r.data(),quality_r.data()),qm=dot(quality_r.data(),quality_z.data());
    balanced_require(std::isfinite(rr)&&rr>=0&&std::isfinite(qm)&&qm>=0,"Native diagnostic invalid true quadratic");
    const double l2=std::sqrt(rr/static_cast<double>(rhs_norm2)),weighted=std::sqrt(qm/common_bMb);
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
    r.buffer_view().copy_from(b.buffer_view());
    auto action=[&]() {
        output["preconditioner_calls"]=output["preconditioner_calls"].get<int>()+1;
        if(augmented)space->apply(r.data(),z.data(),base,nullptr,fused?ApplyMode::FusedProjection:ApplyMode::Reference);
        else base(r.data(),z.data());
        if(augmented&&balanced_fault=="after_apply") {
            check(cudaStreamSynchronize(cudaStreamPerThread));
            throw std::runtime_error("Native balanced injected failure after apply");
        }
        if(augmented&&fused&&balanced_fault=="after_fused_apply") {
            check(cudaStreamSynchronize(cudaStreamPerThread));
            throw std::runtime_error("Native balanced injected failure after fused apply");
        }
    };
    action();double rho=dot(r.data(),z.data()),rho0=rho;
    balanced_require(std::isfinite(rho)&&rho>0,"Native diagnostic nonpositive initial rho");
    output["rho_initial"]=rho;p.copy_from(z);bool quality_active=false;
    const int limit=std::min<int>(int(max_iter)-1,2000);
    for(int iteration=1;iteration<=limit;++iteration) {
        multiply(p.data(),Ap.data());const double curvature=dot(p.data(),Ap.data());
        balanced_require(std::isfinite(curvature)&&curvature>0,"Native diagnostic nonpositive curvature");
        const double alpha=rho/curvature;
        balanced_require(std::isfinite(alpha)&&alpha>0,"Native diagnostic invalid alpha");
        update_vector_dx_r<<<blocks,256,0,cudaStreamPerThread>>>(work_x.data(),r.data(),p.data(),Ap.data(),alpha,n);
        check(cudaGetLastError());
        Json point={{"iteration",iteration},{"pre_update_rho",rho},{"p_Ap",curvature},{"alpha",alpha},
            {"quality_checked",false},{"true_relative_l2",nullptr},{"common_native_MAS_relative",nullptr},
            {"recursive_relative_l2",nullptr},{"screened_out",false},{"forced_quality_check",false}};
        quality_active=quality_active||rho<=1e-4*rho0;
        if(quality_active) {
            bool due=true;
            if(screened) {
                const auto screen_start=Clock::now();const double rr=dot(r.data(),r.data());
                const int last=output["last_quality_iteration"].get<int>();
                due=native_quality_check_due(iteration,last,rr,static_cast<double>(rhs_norm2),limit);
                output["screen_host_ms"]=output["screen_host_ms"].get<double>()+milliseconds(screen_start);
                output["screen_checks"]=output["screen_checks"].get<int>()+1;
                point["recursive_relative_l2"]=std::sqrt(rr/static_cast<double>(rhs_norm2));
                point["screened_out"]=!due;
                point["forced_quality_check"]=last==0||iteration==limit||iteration-last>=8;
            }
            if(due&&quality_probe(point,output)) {
                output["trace"].push_back(point);output["converged"]=true;output["iterations"]=iteration;
                return;
            }
        }
        output["trace"].push_back(point);output["iterations"]=iteration;
        action();const double next=dot(r.data(),z.data());
        if(next==0) {
            output["zero_rho_true_quality_checked"]=true;
            output["trace"].back()["screened_out"]=false;
            output["trace"].back()["forced_quality_check"]=true;
            if(quality_probe(output["trace"].back(),output))return;
        }
        balanced_require(std::isfinite(next)&&next>0,"Native diagnostic invalid next rho");
        const double beta=next/rho;balanced_require(std::isfinite(beta),"Native diagnostic invalid beta");
        update_vector_c<<<blocks,256,0,cudaStreamPerThread>>>(p.data(),z.data(),beta,n);
        check(cudaGetLastError());rho=next;
    }
    throw NativeBalancedRejected("Native diagnostic iteration budget reached");
};
const int quality_order[3][4]={{0,1,2,3},{3,2,1,0},{2,3,0,1}};
const int apply_order[3][3]={{0,1,2},{2,1,0},{1,2,0}};
for(int pair=0;pair<3;++pair)for(int order=0;order<(apply_study?3:(quality_study?4:2));++order) {
    const int variant=apply_study?apply_order[pair][order]:(quality_study?quality_order[pair][order]:(pair+order)%2);
    const bool requested=apply_study?variant!=0:variant%2!=0;
    const bool screened=apply_study||variant>=2,fused=apply_study&&variant==2;
    const auto reset_start=Clock::now();restore_m();
    check(cudaMemsetAsync(work_x.data(),0,size_t(n)*sizeof(Float),cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));info=Json::object();
    const double reset_ms=milliseconds(reset_start);
    Json result,failed;std::string reason=requested?rejection:std::string{};
    bool used=requested&&coarse_ready;
    const auto solve_start=Clock::now();check(cudaEventRecord(events.begin,cudaStreamPerThread));
    try{solve_candidate(used,screened,used&&fused,result);}
    catch(const NativeBalancedRejected& error) {
        if(!used){study["failed_target_candidate"]=result;throw;}
        check(cudaStreamSynchronize(cudaStreamPerThread));failed=result;
        failed["host_ms"]=milliseconds(solve_start);reason=error.what();used=false;
        restore_m();check(cudaMemsetAsync(work_x.data(),0,size_t(n)*sizeof(Float),cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        try{solve_candidate(false,screened,false,result);}catch(...){study["failed_fallback_candidate"]=result;throw;}
    }
    catch(...){study["failed_target_candidate"]=result;throw;}
    check(cudaEventRecord(events.end,cudaStreamPerThread));check(cudaEventSynchronize(events.end));
    const double solve_ms=milliseconds(solve_start);float event_ms=0;
    check(cudaEventElapsedTime(&event_ms,events.begin,events.end));
    Json row={{"pair",pair+1},{"order",order+1},{"requested_balanced",requested},{"used_balanced",used},
        {"screened",screened},{"requested_fused",fused},{"used_fused",used&&fused},
        {"fallback",requested&&!used},{"fallback_reason",reason},{"failed_attempt",failed},
        {"solve_host_ms",solve_ms},{"solve_event_ms",event_ms},{"reset_host_ms",reset_ms},{"result",result}};
    const auto export_start=Clock::now();const auto values=read(work_x.data(),n);
    for(double v:values)if(!std::isfinite(v))throw std::runtime_error("Native diagnostic nonfinite solution");
    const std::string tag="_target_p"+std::to_string(pair+1)+(requested?"_balanced":"_MAS")
        +(apply_study?(fused?"_fused":"_reference"):(quality_study?(screened?"_screened":"_dense"):""));
    row["solution_file"]=save(tag+"_x.bin",values.data(),values.size()*sizeof(Float));
    row["solution_export_host_ms"]=milliseconds(export_start);row["passed"]=result.at("converged");
    study["runs"].push_back(row);
}
// Calibration is separate from the paired solves, with explicit synchronization.
Json calibration=Json::array();
std::vector<double> reference_b_action;
for(int i=0;i<4;++i)for(int arm=0;arm<(coarse_ready?(apply_study?3:2):1);++arm) {
    restore_m();check(cudaStreamSynchronize(cudaStreamPerThread));const auto t=Clock::now();
    if(arm)space->apply(b.data(),quality_z.data(),base,nullptr,arm==2?ApplyMode::FusedProjection:ApplyMode::Reference);
    else base(b.data(),quality_z.data());
    check(cudaStreamSynchronize(cudaStreamPerThread));
    calibration.push_back({{"sample",i+1},{"balanced",arm!=0},{"fused",arm==2},{"whole_apply_host_ms",milliseconds(t)}});
    if(i==0) {
        const auto values=read(quality_z.data(),n);
        const char* key=arm==2?"fused_b_action_file":(arm?"balanced_b_action_file":"MAS_b_action_file");
        const char* suffix=arm==2?"_fused_b.bin":(arm?"_balanced_b.bin":"_MAS_b.bin");
        study[key]=save(suffix,values.data(),values.size()*sizeof(Float));
        if(arm==1&&apply_study)reference_b_action=values;
    }
}
study["separate_apply_calibration"]=calibration;
if(coarse_ready) {
    Json stages=Json::array();
    NativeBalancedSpace::Profile profile=[&](const char* name,const std::function<void()>& operation) {
        check(cudaStreamSynchronize(cudaStreamPerThread));const auto t=Clock::now();operation();
        check(cudaStreamSynchronize(cudaStreamPerThread));stages.push_back({{"stage",name},{"host_ms",milliseconds(t)}});
    };
    restore_m();space->apply(b.data(),quality_z.data(),base,&profile);
    study["separate_synchronized_apply_stages"]=stages;
    if(apply_study) {
        stages=Json::array();restore_m();space->apply(b.data(),quality_z.data(),base,&profile,ApplyMode::FusedProjection);
        study["separate_fused_synchronized_apply_stages"]=stages;
    }
    study["stage_timing_scope"]="separate single apply with per-stage synchronization; not additive to paired solves or pure kernel time";
}
if(coarse_ready&&apply_study) {
    Json batches=Json::array();
    const auto before_b=digest(b.data(),size_t(n)*sizeof(double));
    const auto before_Z=digest(space->Z.value,size_t(n)*8*sizeof(double));
    const auto before_W=digest(space->W.value,size_t(n)*8*sizeof(double));
    const auto before_E=digest(space->E.value,64*sizeof(double)),before_L=digest(space->L.value,64*sizeof(double));
    for(int pair=0;pair<6;++pair)for(int order=0;order<2;++order) {
        const bool fused=(pair+order)%2!=0;restore_m();check(cudaStreamSynchronize(cudaStreamPerThread));
        const auto batch_start=Clock::now();check(cudaEventRecord(events.begin,cudaStreamPerThread));
        for(int call=0;call<16;++call)
            space->apply(b.data(),quality_z.data(),base,nullptr,fused?ApplyMode::FusedProjection:ApplyMode::Reference);
        check(cudaEventRecord(events.end,cudaStreamPerThread));check(cudaEventSynchronize(events.end));
        const double host_ms=milliseconds(batch_start);float event_ms=0;
        check(cudaEventElapsedTime(&event_ms,events.begin,events.end));
        const auto values=read(quality_z.data(),n);double error=0,norm=0;
        for(int i=0;i<n;++i){const double d=values[i]-reference_b_action[i];error+=d*d;norm+=reference_b_action[i]*reference_b_action[i];}
        const double relative=std::sqrt(error/std::max(norm,1e-300));
        if(!std::isfinite(relative)||relative>1e-3)throw std::runtime_error("Apply batch action calibration failed");
        const std::string output_file=save("_apply_batch_p"+std::to_string(pair+1)+(fused?"_fused.bin":"_reference.bin"),
            values.data(),values.size()*sizeof(double));
        batches.push_back({{"pair",pair+1},{"order",order+1},{"fused",fused},{"calls",16},
            {"host_ms",host_ms},{"event_ms",event_ms},{"action_relative",relative},{"output_file",output_file}});
    }
    const bool exact=before_b==digest(b.data(),size_t(n)*sizeof(double))
        &&before_Z==digest(space->Z.value,size_t(n)*8*sizeof(double))&&before_W==digest(space->W.value,size_t(n)*8*sizeof(double))
        &&before_E==digest(space->E.value,64*sizeof(double))&&before_L==digest(space->L.value,64*sizeof(double));
    study["apply_batches"]=batches;study["apply_batch_inputs_unchanged"]=exact;
    if(!exact)throw std::runtime_error("Apply batch changed input or coarse operator");
}
