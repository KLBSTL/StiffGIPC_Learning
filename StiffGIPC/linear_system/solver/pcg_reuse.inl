// Research-only initial guesses. The original PCG remains the default path.
namespace gipc {
__global__ void reuse_axpy(double* x,const double* u,double c,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=c*u[i];
}

// One grid covers all retained columns. Coefficients cross the PCIe boundary
// together so the existing host-side finite check precedes the direction update.
struct ReuseProjectionColumns {
    const double* column[4];
};
struct ReuseProjectionWeights {
    double value[4];
};
__device__ double reuse_block_sum(double v) {
    __shared__ double warp_sums[8];
    for(int step=16;step;step/=2)v+=__shfl_down_sync(0xffffffffu,v,step);
    if((threadIdx.x&31)==0)warp_sums[threadIdx.x/32]=v;
    __syncthreads();
    v=threadIdx.x<8?warp_sums[threadIdx.x]:0.;
    if(threadIdx.x<32)for(int step=16;step;step/=2)v+=__shfl_down_sync(0xffffffffu,v,step);
    return v;
}
__global__ void reuse_projection_partials(ReuseProjectionColumns au,const double* z,
                                         double* partial,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const double sum=reuse_block_sum(i<n?au.column[blockIdx.y][i]*z[i]:0.);
    if(threadIdx.x==0)partial[blockIdx.y*gridDim.x+blockIdx.x]=sum;
}
__global__ void reuse_projection_finish(const double* partial,double* dots,int blocks) {
    double sum=0.;
    for(int k=threadIdx.x;k<blocks;k+=blockDim.x)sum+=partial[blockIdx.x*blocks+k];
    sum=reuse_block_sum(sum);
    if(threadIdx.x==0)dots[blockIdx.x]=sum;
}
__global__ void reuse_projection_update(double* p,ReuseProjectionColumns history,
                                        ReuseProjectionWeights weights,int columns,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double direction=p[i];
    for(int j=0;j<columns;++j)direction+=weights.value[j]*history.column[j][i];
    p[i]=direction;
}
__global__ void reuse_diagonal_apply(const double* inverse,const double* x,double* y,int blocks) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=blocks)return;
    for(int i=0;i<3;++i){double v=0;for(int j=0;j<3;++j)v+=inverse[9*k+3*i+j]*x[3*k+j];y[3*k+i]=v;}
}
void PCGSolver::reuse_begin_frame() {
    reuse_history.clear();reuse_history_sources.clear();reuse_pending.reset();reuse_solve_ordinal=0;reuse_pending_ms=0;
    reuse_pending_b.clear();reuse_pending_au.clear();reuse_source_b.clear();reuse_source_au.clear();
}
void PCGSolver::reuse_discard() {reuse_pending.reset();reuse_pending_ms=0;reuse_pending_b.clear();reuse_pending_au.clear();}
void PCGSolver::reuse_stage(cudatool::CDenseVectorView<Float> x,cudatool::CDenseVectorView<Float> b,bool capture_source) {
    CostScope scope("pcg.reuse.history_stage");
    CUDA_SAFE_CALL(cudaDeviceSynchronize());const auto start=std::chrono::steady_clock::now();
    reuse_pending=std::make_unique<DeviceDenseVector>();reuse_pending->resize(x.size());
    reuse_pending->buffer_view().copy_from(x.buffer_view());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    reuse_pending_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    reuse_pending_source=Json::object();attach_solve_context(reuse_pending_source);
    reuse_pending_source["solve_ordinal"]=reuse_solve_ordinal;
    reuse_pending_source["stage_ms"]=reuse_pending_ms;
    const auto o=PcgReuseOptions::read();
    if(o.decision_study() && capture_source)
        reuse_pending_source["previous_updates"]=Statistics::instance().at_current_frame()["newton"].back()["pcg"].value("recurrence_updates",-1);
    if(capture_source&&(o.phase=="D0"||o.phase=="D3"||o.phase=="DA"||o.phase=="DT"||o.phase=="DB"||o.decision_study())) {
        const auto started=std::chrono::steady_clock::now();reuse_source_ax.resize(x.size());spmv(x,reuse_source_ax.view());
        reuse_pending_b.resize(b.size());reuse_pending_au.resize(x.size());
        CUDA_SAFE_CALL(cudaMemcpy(reuse_pending_b.data(),b.data(),b.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(reuse_pending_au.data(),reuse_source_ax.data(),x.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        reuse_pending_source["diagnostic_source_capture_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["reuse_history_stage_ms"]=reuse_pending_ms;
    info["reuse_history_pending_bytes"]=x.size()*sizeof(Float);
}
void PCGSolver::reuse_accept() {
    if(!reuse_pending)return;
    const auto o=PcgReuseOptions::read();
    const size_t capacity=o.mode=="g4" || !o.directory.empty()?4:1;
    const auto start=std::chrono::steady_clock::now();
    if(reuse_history.size()>=capacity) {
        reuse_history.erase(reuse_history.begin());reuse_history_sources.erase(reuse_history_sources.begin());
    }
    reuse_history.push_back(std::move(reuse_pending));
    reuse_history_sources.push_back(reuse_pending_source);
    reuse_source_b=std::move(reuse_pending_b);reuse_source_au=std::move(reuse_pending_au);
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["reuse_history_committed"]=true;info["reuse_history_columns_after"]=reuse_history.size();
    const double commit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    info["reuse_history_commit_ms"]=commit_ms;reuse_history_sources.back()["commit_ms"]=commit_ms;
}

SizeT PCGSolver::pcg_reuse(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,SizeT max_iter,const std::string& requested_mode,bool audited,double frozen_rho_b,double reference_gamma)
{
    CostScope scope("pcg.reuse.solve");
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const int n=static_cast<int>(b.size()),blocks=(n+255)/256;
    const SizeT original_budget=max_iter>0?max_iter-1:0;
    const bool curve=!reuse_curve_prefix.empty(),metric=reuse_metric_active;
    const bool signal=curve&&PcgReuseOptions::read().phase=="DT";
    const SizeT budget=curve&&reuse_curve_budget>=0?std::min(original_budget,SizeT(reuse_curve_budget)):original_budget;
    size_t spmvs=0,preconditions=0,dots=0,updates=0;int replacements=0;bool fallback=false;
    size_t metric_applies=0,curve_spmvs=0;double metric_ms=0;
    double audit_ms=0;
    reuse_partials.resize(std::max(1,blocks));
    auto cv=[](auto& v){return cudatool::CDenseVectorView<Float>{v.data(),static_cast<int>(v.size())};};
    auto dot=[&](const double* a,const double* c) {
        ++dots;return My_PCG_General_v_v_Reduction_Algorithm(reuse_partials.data(),a,c,reduction_result.data(),n);
    };
    auto mv=[&](cudatool::CDenseVectorView<Float> a,cudatool::DenseVectorView<Float> c){++spmvs;spmv(a,c);};
    auto pre=[&](){++preconditions;apply_preconditioner(z.view(),r.cview());};
    auto dnorm=[&](const double* value){
        const auto t=std::chrono::steady_clock::now();++metric_applies;reuse_metric_z.resize(n);
        reuse_diagonal_apply<<<(n/3+255)/256,256>>>(reuse_diagonal_inverse.data(),value,reuse_metric_z.data(),n/3);
        const double norm=dot(value,reuse_metric_z.data());
        metric_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();return norm;
    };
    CUDA_SAFE_CALL(cudaDeviceSynchronize());const auto start=std::chrono::steady_clock::now();
    x.buffer_view().fill(0);r.buffer_view().copy_from(b.buffer_view());pre();
    const double measured_rho_b=dot(r.data(),z.data()),bb=dot(b.data(),b.data());
    const double rho_b=frozen_rho_b>=0?frozen_rho_b:measured_rho_b;
    const double target=m_config.global_tol_rate*rho_b;
    const double db=metric?dnorm(b.data()):0,dt=metric?m_config.global_tol_rate*reuse_metric_b:0;
    std::string mode=requested_mode;
    const bool decision=mode=="probe_off"||mode=="probe_on"||mode=="probe_policy"||mode=="cheap_policy";
    bool probe=false;
    int previous=-1;
    double threshold=0;
    if(decision) {
        const bool cheap=mode=="cheap_policy";
        const bool policy=cheap||mode=="probe_policy";
        previous=reuse_history_sources.empty()?-1:reuse_history_sources.back().value("previous_updates",-1);
        mode=reuse_history.empty()?"b0":reuse_history.size()==1?"g1":"w1";
        if(previous>=0) {
            if(policy) {
                const char* raw=std::getenv(cheap?"GIPC_DG_CHEAP_RULE":"GIPC_DG_PROBE_RULE");
                if(!raw)throw std::runtime_error("Missing locked decision rule");
                const std::string rule(raw);
                if(rule=="batch")mode="g4_batch";
                else if(rule!="simple"){
                    threshold=std::stod(rule);
                    if(!std::isfinite(threshold))throw std::runtime_error("Invalid locked threshold");
                    if(cheap){if(previous>=threshold)mode="g4_batch";}else probe=true;
                }
            } else probe=true;
        }
        info["decision_previous"]=previous;
        info["decision_probe"]=probe;
    }
    info["execution"]="host";info["reuse_mode"]=mode;info["reuse_audit_mode"]=audited?"A":"L";
    info["rho_initial"]=rho_b;info["rho_b"]=rho_b;info["rho_target"]=target;
    info["reference_gamma_target"]=reference_gamma;
    info["stopping_metric"]=metric?"independent_block_diagonal":"native_preconditioned_rho";
    info["diagnostic_curve_only"]=curve;
    if(metric) {
        info["metric_b"]=db;info["metric_b_frozen"]=reuse_metric_b;info["metric_target"]=dt;
        const double gap=std::abs(db-reuse_metric_b)/std::max(std::abs(reuse_metric_b),1e-300);
        info["metric_b_relative_gap"]=gap;
        if(!std::isfinite(db)||(bb>0&&db<=0)||!std::isfinite(gap)||gap>1e-10)
            throw IpcPcgDirectionFailure("Independent D norm changed or invalid");
    }
    info["iterations"]=0;info["recurrence_updates"]=0;info["iteration_limit"]=false;
    info["rho_b_measured"]=measured_rho_b;
    const double rho_gap=frozen_rho_b>=0?std::abs(measured_rho_b-frozen_rho_b)/
        std::max(std::abs(frozen_rho_b),1e-300):0;
    info["rho_b_relative_gap"]=rho_gap;
    info["reuse_spmv_count"]=spmvs;info["reuse_preconditioner_count"]=preconditions;
    info["reuse_reduction_readbacks"]=dots;info["reuse_iteration_budget"]=budget;
    info["reuse_solve_ordinal"]=reuse_solve_ordinal;
    info["reuse_sources"]=reuse_history_sources;
    if(!std::isfinite(rho_gap)||rho_gap>1e-10) {
        info["reuse_failure"]="frozen_rho_changed";
        info["reuse_failed_attempt_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        throw IpcPcgDirectionFailure("Frozen preconditioner RHS norm changed");
    }
    if(!std::isfinite(bb)||!std::isfinite(rho_b)||bb<0||(bb>0&&rho_b<=0))
        throw IpcPcgDirectionFailure("Reuse invalid original RHS/preconditioner norm");
    bool batched=mode=="g4_batch";
    bool augmented=mode=="g4_aug"||batched;
    size_t m=mode=="b0"?0:std::min(reuse_history.size(),size_t(mode=="g4"||augmented?4:1));
    info["reuse_history_columns"]=m;info["reuse_rank"]=0;
    std::vector<std::unique_ptr<DeviceDenseVector>> au;
    Eigen::VectorXd scales(m);Eigen::MatrixXd augmentation_inverse;
    size_t offset=reuse_history.size()-m;
    bool cached_ap=false;
    double first_curvature=0;
    std::unique_ptr<DeviceDenseVector> last_au;
    size_t augmentation_projections=0;double augmentation_ms=0;
    cudatool::DeviceBuffer<double> batch_workspace;
    ReuseProjectionColumns batch_au{},batch_history{};
    // Algorithm 1 of Carlberg et al. (arXiv:1512.05820): keep z and rho
    // unchanged, and subtract U E^+ U^T A z only from the search direction.
    auto orthogonalize=[&](){
        if(!augmented||fallback||augmentation_inverse.rows()==0)return;
        const auto t=std::chrono::steady_clock::now();Eigen::VectorXd h(m);
        if(batched) {
            double* raw=batch_workspace.data()+blocks*m;
            reuse_projection_partials<<<dim3(blocks,static_cast<unsigned>(m)),256>>>(batch_au,z.data(),batch_workspace.data(),n);
            reuse_projection_finish<<<static_cast<unsigned>(m),256>>>(batch_workspace.data(),raw,blocks);
            CUDA_SAFE_CALL(cudaMemcpy(h.data(),raw,m*sizeof(double),cudaMemcpyDeviceToHost));++dots;
            h.array()/=scales.array();
        } else for(size_t j=0;j<m;++j)h[j]=dot(au[j]->data(),z.data())/scales[j];
        const Eigen::VectorXd c=augmentation_inverse*h;
        if(!c.allFinite())throw IpcPcgDirectionFailure("Nonfinite augmented projection");
        if(batched) {
            ReuseProjectionWeights weights{};
            for(size_t j=0;j<m;++j)weights.value[j]=-c[j]/scales[j];
            reuse_projection_update<<<blocks,256>>>(p.data(),batch_history,weights,static_cast<int>(m),n);
        } else for(size_t j=0;j<m;++j)reuse_axpy<<<blocks,256>>>(p.data(),reuse_history[offset+j]->data(),-c[j]/scales[j],n);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());++augmentation_projections;
        augmentation_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    };
    double rho=measured_rho_b;
    auto initialize_seed=[&](){
        CostScope initialization("pcg.reuse.projection");
        Eigen::VectorXd h(m),coef(m);Eigen::MatrixXd gram(m,m);
        bool valid=true;
        for(size_t j=0;j<m;++j) {
            if(last_au && j+1==m)au.push_back(std::move(last_au));
            else {
                au.emplace_back(std::make_unique<DeviceDenseVector>());au.back()->resize(n);
                mv(reuse_history[offset+j]->cview(),au.back()->view());
            }
            const double q=mode=="w1"?1:dot(reuse_history[offset+j]->data(),au.back()->data());
            if(!std::isfinite(q)||q<=0)valid=false;
            scales[j]=q>0?std::sqrt(q):1;
        }
        if(mode=="w1") {
            x.buffer_view().copy_from(reuse_history.back()->buffer_view());
            audit_residual<<<blocks,256>>>(r.data(),b.data(),au.back()->data(),n);
            info["reuse_rank"]=1;
        } else if(valid) {
            for(size_t i=0;i<m;++i) {
                h[i]=dot(reuse_history[offset+i]->data(),b.data())/scales[i];
                for(size_t j=0;j<m;++j)
                    gram(i,j)=dot(reuse_history[offset+i]->data(),au[j]->data())/(scales[i]*scales[j]);
            }
            const double asym=(gram-gram.transpose()).cwiseAbs().rowwise().sum().maxCoeff()/
                std::max(gram.cwiseAbs().rowwise().sum().maxCoeff(),1e-300);
            info["reuse_gram_asymmetry"]=asym;
            valid=gram.allFinite()&&h.allFinite()&&asym<=1e-10;
            if(valid) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen((gram+gram.transpose())*.5);
                valid=eigen.info()==Eigen::Success;
                if(valid) {
                    const auto vals=eigen.eigenvalues();const double hi=vals[m-1];
                    valid=std::isfinite(hi)&&hi>0&&vals[0]>=-1e-10*hi;
                    if(valid) {
                        coef.setZero();int rank=0;
                        if(augmented||signal)augmentation_inverse=Eigen::MatrixXd::Zero(m,m);
                        for(size_t j=0;j<m;++j)if(vals[j]>1e-10*hi) {
                            auto q=eigen.eigenvectors().col(j);coef+=q*(q.dot(h)/vals[j]);++rank;
                            if(augmented||signal)augmentation_inverse+=q*q.transpose()/vals[j];
                        }
                        valid=coef.allFinite();info["reuse_rank"]=rank;
                        if(valid)for(size_t j=0;j<m;++j)
                            reuse_axpy<<<blocks,256>>>(x.data(),reuse_history[offset+j]->data(),coef[j]/scales[j],n);
                        info["reuse_projection_energy_gain"]=valid?.5*h.dot(coef):0;
                        if(m==1)info["reuse_coefficient"]=valid?coef[0]/scales[0]:0;
                    }
                }
            }
            if(valid) {
                r.buffer_view().copy_from(b.buffer_view());
                for(size_t j=0;j<m;++j)
                    reuse_axpy<<<blocks,256>>>(r.data(),au[j]->data(),-coef[j]/scales[j],n);
            }
        }
        if(!valid && mode!="w1") {
            if(augmented)throw IpcPcgDirectionFailure("Augmented basis failed validation");
            x.buffer_view().fill(0);r.buffer_view().copy_from(b.buffer_view());
            info["reuse_projection_rejected"]=true;info["reuse_rank"]=0;
        }
        pre();rho=dot(r.data(),z.data());
    };
    if(m && bb>0)initialize_seed();
    if(probe && m && bb>0) {
        const auto& u=*reuse_history.back();
        const double energy=mode=="g1"?scales[0]*scales[0]:dot(u.data(),au[0]->data());
        const bool at_stop=rho==0 || (metric && dnorm(r.data())<=dt);
        double score=-1;
        if(!at_stop){
            mv(z.cview(),Ap.view());
            first_curvature=dot(z.data(),Ap.data());cached_ap=true;
            const double cross=dot(au[0]->data(),z.data());
            if(energy>0 && first_curvature>0 && std::isfinite(energy) && std::isfinite(first_curvature)){
                const double cosine=(cross/std::sqrt(energy))/std::sqrt(first_curvature);
                const double chi=cosine*cosine;
                if(!std::isfinite(chi)||chi>1+1e-10)throw IpcPcgDirectionFailure("Invalid decision A-angle");
                score=previous*std::min(1.,chi);info["decision_chi"]=std::min(1.,chi);
            }
        }
        info["decision_score"]=score;info["decision_at_stop"]=at_stop;
        if(curve){
            const std::string file=reuse_curve_prefix+"_probe_z.bin";
            std::vector<double> data(n);CUDA_SAFE_CALL(cudaMemcpy(data.data(),z.data(),n*sizeof(double),cudaMemcpyDeviceToHost));
            std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),n*sizeof(double));
            if(!out)throw std::runtime_error("Probe vector export failed");info["decision_z_file"]=file;
        }
        if(score>=0 && (requested_mode=="probe_on" || (requested_mode=="probe_policy" && score>=threshold))) {
            batched=augmented=true;cached_ap=false;mode="g4_batch";
            if(reuse_history.size()==1 && info.value("reuse_rank",0)==1){
                augmentation_inverse=Eigen::MatrixXd::Ones(1,1);
            } else {
                last_au=std::move(au.back());au.clear();m=std::min(size_t(4),reuse_history.size());offset=reuse_history.size()-m;
                scales.resize(m);x.buffer_view().fill(0);initialize_seed();
            }
        }
    }
    if(decision){info["decision_mode"]=mode;info["reuse_mode"]=mode;info["reuse_history_columns"]=m;}
    if(batched&&m&&augmentation_inverse.rows()) {
        batch_workspace.resize((blocks+1)*m);
        for(size_t j=0;j<m;++j) {
            batch_au.column[j]=au[j]->data();
            batch_history.column[j]=reuse_history[offset+j]->data();
        }
    }
    const double rho_seed=rho;info["rho_seed"]=rho_seed;
    if(signal&&m&&augmentation_inverse.rows()) {
        const auto t=std::chrono::steady_clock::now();Eigen::VectorXd h(m);
        for(size_t j=0;j<m;++j)h[j]=dot(au[j]->data(),z.data())/scales[j];
        const Eigen::VectorXd c=augmentation_inverse*h;const double ell=h.dot(c);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
        info["initial_coupling"]={{"ell",ell},{"rho",rho},{"ell_over_rho",rho>0?ell/rho:0},
            {"observation_ms",elapsed},{"decision_boundary","after G4 Gram and seed M application; setup is already paid"}};
        const std::string file=reuse_curve_prefix+"_z0.bin";std::vector<double> values(n);
        CUDA_SAFE_CALL(cudaMemcpy(values.data(),z.data(),n*sizeof(double),cudaMemcpyDeviceToHost));
        std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),n*sizeof(double));
        if(!f)throw std::runtime_error("Initial coupling z export failed");info["initial_coupling"]["z_file"]=file;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    info["reuse_initialization_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    p.copy_from(z);orthogonalize();
    if(curve&&augmented&&PcgReuseOptions::read().phase=="DB") {
        for(const auto& item : {std::make_pair("z",z.data()),std::make_pair("p",p.data())}) {
            const std::string file=reuse_curve_prefix+"_initial_"+item.first+".bin";
            std::vector<double> values(n);
            CUDA_SAFE_CALL(cudaMemcpy(values.data(),item.second,n*sizeof(double),cudaMemcpyDeviceToHost));
            std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),n*sizeof(double));
            if(!out)throw std::runtime_error("Projection validation export failed");
            info[std::string("initial_")+item.first+"_file"]=file;
        }
    }
    int last_sample=-1;
    auto sample=[&](bool force){
        if(!curve||last_sample==int(updates))return;
        const bool gate_candidate=metric&&!info.contains("first_D_pass_updates")&&dnorm(r.data())<=dt;
        if(!force&&!gate_candidate&&updates>1&&(updates&(updates-1)))return;
        last_sample=int(updates);reuse_trace_ax.resize(n);reuse_trace_r.resize(n);
        spmv(cv(x),reuse_trace_ax.view());++curve_spmvs;
        audit_residual<<<blocks,256>>>(reuse_trace_r.data(),b.data(),reuse_trace_ax.data(),n);
        const double rr=dot(reuse_trace_r.data(),reuse_trace_r.data());
        const double q=metric?dnorm(reuse_trace_r.data())/std::max(reuse_metric_b,1e-300):-1;
        if(metric&&!info.contains("first_D_pass_updates")&&std::isfinite(q)&&q<=m_config.global_tol_rate*(1+1e-10)
            &&dot(b.data(),x.data())>0)info["first_D_pass_updates"]=updates;
        std::vector<double> values(n);CUDA_SAFE_CALL(cudaMemcpy(values.data(),x.data(),n*sizeof(double),cudaMemcpyDeviceToHost));
        const std::string file=reuse_curve_prefix+"_k"+std::to_string(updates)+".bin";
        std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),n*sizeof(double));
        if(!out)throw std::runtime_error("D curve direction export failed");
        info["curve"].push_back({{"updates",updates},{"q_D",q},{"gamma2",std::sqrt(rr/std::max(bb,1e-300))},
            {"direction_file",file},{"algorithm_spmv",spmvs},{"preconditioner",preconditions}});
    };
    sample(true);
    auto audit=[&]() {
        const auto t=std::chrono::steady_clock::now();CostScope check("pcg.reuse.true_residual_audit");
        mv(cv(x),Ap.view());audit_residual<<<blocks,256>>>(r.data(),b.data(),Ap.data(),n);
        pre();rho=dot(r.data(),z.data());
        const double rr=dot(r.data(),r.data()),bx=dot(b.data(),x.data());
        info["rho_true_exit"]=rho;info["true_relative_residual"]=bb>0?std::sqrt(rr/bb):std::sqrt(rr);
        info["rhs_dot_solution"]=bx;
        const double dd=metric?dnorm(r.data()):0;
        if(metric)info["q_D_true_exit"]=dd/std::max(reuse_metric_b,1e-300);
        const bool passed=std::isfinite(rho)&&rho>=0&&std::isfinite(rr)&&std::isfinite(bx)&&
            (bb==0?rr==0:bx>0)&&(metric?(std::isfinite(dd)&&dd>=0&&dd<=dt*(1+1e-10)):rho<=target*(1+1e-10))&&
            (reference_gamma<0 || (bb>0?std::sqrt(rr/bb):std::sqrt(rr))<=reference_gamma);
        info["reuse_true_audit_passed"]=passed;
        audit_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
        return passed;
    };
    bool finished=false;std::string failure;
    while(!finished) {
        bool legacy_stopped=false;
        if(!std::isfinite(rho)||rho<0)failure="invalid_rho";
        else if(rho==0 || (!curve&&audited&&reference_gamma<0&&(metric?dnorm(r.data())<=dt:rho<=target))) {
            if(!audited){finished=true;break;}
        } else {
            while(updates<budget) {
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>(reference_gamma>=0?120:60)) {
                    failure="registered_linear_timeout";break;
                }
                cost_trace_iteration(static_cast<int>(updates+1));
                double curvature=first_curvature;
                if(!cached_ap){mv(p.cview(),Ap.view());curvature=dot(p.data(),Ap.data());}
                cached_ap=false;
                if(!std::isfinite(curvature)||curvature<=0){failure="nonpositive_curvature";break;}
                const double alpha=rho/curvature;
                if(!std::isfinite(alpha)){failure="nonfinite_alpha";break;}
                update_vector_dx_r<<<blocks,256>>>(x.data(),r.data(),p.data(),Ap.data(),alpha,n);
                ++updates;info["rho_stop"]=rho;
                sample(false);
                // Preserve the production previous-rho stopping order.
                if(!curve&&(metric?dnorm(r.data())<=dt:rho<=target) && (reference_gamma<0 ||
                    std::sqrt(dot(r.data(),r.data())/std::max(bb,1e-300))<=reference_gamma)) {legacy_stopped=true;break;}
                pre();const double next=dot(r.data(),z.data());
                if(!std::isfinite(next)||next<0){failure="invalid_rho";break;}
                if(next==0){rho=0;legacy_stopped=true;break;}
                const double beta=next/rho;
                if(!std::isfinite(beta)){failure="nonfinite_beta";break;}
                update_vector_c<<<blocks,256>>>(p.data(),z.data(),beta,n);orthogonalize();rho=next;
            }
            if(!audited && failure.empty() && legacy_stopped){finished=true;break;}
        }
        if(failure=="registered_linear_timeout")break;
        if(curve&&updates>=budget) {audit();info["diagnostic_budget_reached"]=true;finished=true;break;}
        cached_ap=false;
        if(audited && audit()){finished=true;break;}
        if(updates>=budget){failure="iteration_limit";break;}
        if(audited&&failure.empty()&&std::isfinite(rho)&&rho>0&&replacements<2) {
            ++replacements;p.copy_from(z);orthogonalize();continue;
        }
        if(!fallback&&mode!="b0") {
            fallback=true;failure.clear();x.buffer_view().fill(0);
            r.buffer_view().copy_from(b.buffer_view());pre();rho=dot(r.data(),z.data());p.copy_from(z);continue;
        }
        if(failure.empty())failure="true_residual_failed";break;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());cost_trace_iteration(-1);
    sample(true);
    info["iterations"]=updates;info["recurrence_updates"]=updates;
    info["reuse_total_linear_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    info["reuse_audit_ms"]=audit_ms;info["reuse_spmv_count"]=spmvs;
    info["reuse_preconditioner_count"]=preconditions;info["reuse_reduction_readbacks"]=dots;
    info["reuse_replacements"]=replacements;info["reuse_fallback"]=fallback;
    info["reuse_iteration_budget"]=budget;info["iteration_limit"]=!finished&&updates>=budget;
    info["original_iteration_budget"]=original_budget;info["D_apply_count"]=metric_applies;
    info["D_apply_ms"]=metric_ms;info["curve_spmv_count"]=curve_spmvs;
    info["augmentation_projections"]=augmentation_projections;info["augmentation_ms"]=augmentation_ms;
    if(!finished) {info["reuse_failure"]=failure;throw IpcPcgDirectionFailure("Reuse PCG: "+failure);}
    // L mode explicitly keeps this audit out of its reported service timing.
    if(!audited){audit();info["reuse_audit_outside_timing"]=true;}
    return static_cast<SizeT>(updates);
}

void PCGSolver::reuse_study(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,SizeT max_iter)
{
    const auto o=PcgReuseOptions::read();if(reuse_solve_ordinal!=o.target_solve)return;
    if(o.phase=="D0"||o.phase=="D3"||o.phase=="DA"||o.phase=="DT"||o.phase=="DB"||o.decision_study()) {reuse_explain_study(x,b,max_iter);return;}
    namespace fs=std::filesystem;fs::create_directories(o.directory);
    const std::string prefix=o.directory+"/f"+std::to_string(total_Frames+1)+"_s"+std::to_string(reuse_solve_ordinal);
    if(fs::exists(prefix+".json"))throw std::runtime_error("Refusing to overwrite reuse study");
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const Json saved=info;const auto config=m_config;
    const int saved_restriction=research_restriction_override();
    const double frozen_rho_b=saved.at("rho_b").get<double>();
    const auto before=snapshot_system(prefix+"_system");
    auto restore_scratch=checkpoint_preconditioner_scratch();
    auto get=[](const auto& v){std::vector<Float> a(v.size());if(!a.empty())
        CUDA_SAFE_CALL(cudaMemcpy(a.data(),v.data(),a.size()*sizeof(Float),cudaMemcpyDeviceToHost));return a;};
    const auto primary=get(x),zr=get(z),rr=get(r),pr=get(p),ar=get(Ap);
    auto put=[](auto& v,const auto& a){if(!a.empty())CUDA_SAFE_CALL(cudaMemcpy(v.data(),a.data(),a.size()*sizeof(Float),cudaMemcpyHostToDevice));};
    Json study={{"schema","pcg-reuse-frozen-1"},{"frame",total_Frames+1},{"solve",reuse_solve_ordinal},
        {"history_columns",reuse_history.size()},{"system",before},{"runs",Json::array()},
        {"reference_runs",Json::array()},{"diagnostic_only",true},{"history_sources",reuse_history_sources},
        {"phase",o.phase}};
    attach_solve_context(study);
    auto restore=[&](){put(x,primary);put(z,zr);put(r,rr);put(p,pr);put(Ap,ar);restore_scratch();info=saved;m_config=config;
        research_restriction_override()=saved_restriction;};
    try {
        if(o.phase=="P1M") {
            // Diagnostic only: never change the production preconditioner or the
            // registered 1e-10 agreement gate to make a candidate pass.
            const auto rhs=get(b);study["production_rho_b"]=frozen_rho_b;
            study["preconditioner_probes"]=Json::array();
            const int n=static_cast<int>(b.size());reuse_partials.resize(std::max(1,(n+255)/256));
            for(int variant=0;variant<4;++variant) {
                const char* labels[]={"native_restored","native_carry_scratch","ordered_fp32","ordered_fp64"};
                const int restriction=variant<2?0:variant-1;
                restore();std::vector<std::vector<Float>> vectors;
                Json group={{"label",labels[variant]},{"samples",Json::array()}};
                for(int rep=0;rep<10;++rep) {
                    if(variant!=1||rep==0)restore();
                    research_restriction_override()=restriction;
                    r.buffer_view().copy_from(b.buffer_view());
                    apply_preconditioner(z.view(),r.cview());
                    const double gpu=My_PCG_General_v_v_Reduction_Algorithm(reuse_partials.data(),r.data(),z.data(),reduction_result.data(),n);
                    auto value=get(z);long double cpu=0,norm=0,diff=0;
                    for(int i=0;i<n;++i) {cpu+=(long double)rhs[i]*value[i];norm+=(long double)value[i]*value[i];
                        if(!vectors.empty()){const long double delta=value[i]-vectors.front()[i];diff+=delta*delta;}}
                    const std::string file=prefix+"_M"+std::to_string(variant)+"_r"+std::to_string(rep+1)+".bin";
                    std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(value.data()),value.size()*sizeof(Float));
                    if(!out)throw std::runtime_error("Preconditioner probe export failed");
                    group["samples"].push_back({{"repeat",rep+1},{"rho_gpu",gpu},{"rho_cpu",double(cpu)},
                        {"relative_to_production",std::abs(gpu-frozen_rho_b)/std::max(std::abs(frozen_rho_b),1e-300)},
                        {"relative_z_to_first",std::sqrt(double(diff/std::max(norm,1e-300L)))},{"file",file}});
                    vectors.push_back(std::move(value));
                }
                double pair=0;for(size_t a=0;a<vectors.size();++a)for(size_t c=0;c<a;++c) {
                    long double difference=0,den=0;for(int i=0;i<n;++i){const long double d=vectors[a][i]-vectors[c][i];
                        difference+=d*d;den+=(long double)vectors[c][i]*vectors[c][i];}
                    pair=std::max(pair,std::sqrt(double(difference/std::max(den,1e-300L))));
                }
                group["max_pair_relative_z"]=pair;study["preconditioner_probes"].push_back(group);
            }
        } else {
        for(size_t j=0;j<reuse_history.size();++j) {
            auto values=get(*reuse_history[j]);const std::string file=prefix+"_u"+std::to_string(j)+".bin";
            std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Float));
            if(!out)throw std::runtime_error("History export failed");study["history_files"].push_back(file);
        }
        const bool cold=reuse_solve_ordinal==1;
        std::vector<std::string> arms=cold||o.phase=="P1L"?std::vector<std::string>{"b0","w1","g1"}:
            (o.phase=="P1A"?std::vector<std::string>{"b0","w1","g1","b0L"}:
                std::vector<std::string>{"b0","b0_placebo","w1","g1","g4"});
        std::vector<std::vector<int>> order;
        if(!cold&&o.phase=="P1L") {
            order={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
            const std::string scene=std::getenv("GIPC_SCENE")?std::getenv("GIPC_SCENE"):"";
            const int omit=scene=="boxes1920_cloth_l"?0:scene=="cloth_hang_l"?3:4;
            order.erase(order.begin()+omit);
        } else {
            for(int shift=0;shift<int(arms.size());++shift) {
                std::vector<int> row;for(int j=0;j<int(arms.size());++j)row.push_back((j+shift)%arms.size());
                order.push_back(row);
            }
            if(o.phase=="P3") {
                const auto first=order;for(auto row:first){std::reverse(row.begin(),row.end());order.push_back(row);}
            }
        }
        std::mt19937 random(20261116u+unsigned(total_Frames+1));std::shuffle(order.begin(),order.end(),random);
        const int repetitions=cold?1:o.repeats;
        for(int rep=cold?0:-1;rep<repetitions;++rep)for(size_t j=0;j<arms.size();++j) {
            const auto& arm=arms[rep<0?j:order[rep%order.size()][j]];restore();
            info=Json::object();attach_solve_context(info);
            Json row={{"arm",arm},{"repeat",rep+1},{"warmup",rep<0}};
            row["order_position"]=j;
            try {pcg_reuse(x,b,max_iter,arm=="b0L"||arm=="b0_placebo"?"b0":arm,
                    arm!="b0L"&&o.audit=="A",frozen_rho_b);row["status"]="completed";
                if(rep>=0) {
                    const auto direction=get(x);const std::string file=prefix+"_"+arm+"_r"+std::to_string(rep+1)+".bin";
                    std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(direction.data()),direction.size()*sizeof(Float));
                    if(!out)throw std::runtime_error("Replay direction export failed");row["direction_file"]=file;
                }
            }
            catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
            row["pcg"]=info;study["runs"].push_back(row);
        }
        if(!cold)for(double tolerance:{1e-10,1e-12})for(int rep=0;rep<2;++rep) {
            restore();m_config.global_tol_rate=tolerance;
            info=Json::object();attach_solve_context(info);
            Json row={{"rho_tolerance",tolerance},{"repeat",rep+1}};
            const char* explicit_l2=std::getenv("GIPC_PCG_REUSE_REFERENCE_L2");
            const double gamma=explicit_l2&&std::string(explicit_l2)=="1"?(tolerance==1e-10?1e-8:1e-9):-1;
            row["gamma_target"]=gamma;
            try {pcg_reuse(x,b,max_iter,"b0",true,frozen_rho_b,gamma);row["status"]="completed";
                const auto direction=get(x);const std::string file=prefix+"_ref"+std::to_string(study["reference_runs"].size())+".bin";
                std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(direction.data()),direction.size()*sizeof(Float));
                if(!out)throw std::runtime_error("Reference export failed");row["direction_file"]=file;
            }catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
            row["pcg"]=info;study["reference_runs"].push_back(row);
        }
        }
        restore();const auto after=snapshot_system("");
        study["operator_preserved"]=snapshot_operator_identity(before)==snapshot_operator_identity(after);
        study["scratch_restored"]=before==after;
        study["solution_restored"]=primary==get(x);study["info_restored"]=info==saved;
        std::ofstream out(prefix+".json");out<<study.dump(2);out.close();
        if(!out||!study["operator_preserved"].get<bool>()||!study["scratch_restored"].get<bool>()||
           !study["solution_restored"].get<bool>()||!study["info_restored"].get<bool>())
            throw std::runtime_error("Reuse study failed to preserve production state");
        info["reuse_study_file"]=prefix+".json";
    }catch(...){restore();throw;}
}
} // namespace gipc
