// Independent D-based frozen diagnostics. Never used by the physical carrier.
#include <Eigen/Cholesky>
#include <map>
namespace gipc {
void PCGSolver::reuse_explain_study(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,SizeT max_iter)
{
    const auto options=PcgReuseOptions::read();namespace fs=std::filesystem;
    const bool service_study=options.phase=="DT"||options.phase=="DB"||options.decision_study();
    if(service_study&&reuse_pending)throw std::runtime_error("Service study requires an empty pending history slot");
    fs::create_directories(options.directory);
    const std::string prefix=options.directory+"/f"+std::to_string(total_Frames+1)+"_s"+std::to_string(reuse_solve_ordinal);
    if(fs::exists(prefix+".json"))throw std::runtime_error("Refusing to overwrite D study");
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const Json saved=info;const auto config=m_config;const int saved_restriction=research_restriction_override();
    const auto before=snapshot_system(prefix+"_system");auto restore_scratch=checkpoint_preconditioner_scratch();
    auto get=[](const auto& v){std::vector<Float> a(v.size());if(!a.empty())
        CUDA_SAFE_CALL(cudaMemcpy(a.data(),v.data(),a.size()*sizeof(Float),cudaMemcpyDeviceToHost));return a;};
    auto put=[](auto& v,const auto& a){if(!a.empty())CUDA_SAFE_CALL(cudaMemcpy(v.data(),a.data(),a.size()*sizeof(Float),cudaMemcpyHostToDevice));};
    auto output=[](const std::string& file,const auto& a){std::ofstream f(file,std::ios::binary);
        f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(a[0]));if(!f)throw std::runtime_error("D diagnostic write failed");};
    const auto primary=get(x),rhs=get(b),zr=get(z),rr=get(r),pr=get(p),ar=get(Ap);
    auto restore=[&](){if(service_study)reuse_discard();put(x,primary);put(z,zr);put(r,rr);put(p,pr);put(Ap,ar);restore_scratch();
        info=saved;m_config=config;reuse_metric_active=false;reuse_curve_prefix.clear();reuse_curve_budget=-1;
        research_restriction_override()=saved_restriction;};
    Json study={{"schema","pcg-reuse-D-frozen-1"},{"frame",total_Frames+1},{"solve",reuse_solve_ordinal},
        {"history_columns",reuse_history.size()},{"history_sources",reuse_history_sources},{"system",before},
        {"runs",Json::array()},{"reference_runs",Json::array()},{"curve_runs",Json::array()},
        {"diagnostic_only",true},{"phase",options.phase},{"carrier_stop","original B0-A native rho"}};
    attach_solve_context(study);
    auto finish=[&](){restore();const auto after=snapshot_system("");
        study["operator_preserved"]=snapshot_operator_identity(before)==snapshot_operator_identity(after);
        study["scratch_restored"]=before==after;study["solution_restored"]=primary==get(x);study["info_restored"]=info==saved;
        std::ofstream f(prefix+".json");f<<study.dump(2);f.close();
        if(!f||!study["operator_preserved"].get<bool>()||!study["scratch_restored"].get<bool>()||
            !study["solution_restored"].get<bool>()||!study["info_restored"].get<bool>())throw std::runtime_error("D study preservation failed");
        info["reuse_study_file"]=prefix+".json";};
    try {
        const auto started=std::chrono::steady_clock::now();const int n=int(b.size()),nb=n/3;
        if(n%3)throw std::runtime_error("D metric requires full 3x3 blocks");
        const size_t entries=before["blocks"].get<size_t>();
        std::vector<int> rows(entries),cols(entries);std::vector<Matrix3x3> values(entries);
        auto input=[](const std::string& file,auto& a){std::ifstream f(file,std::ios::binary);
            f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(a[0]));if(!f)throw std::runtime_error("D snapshot read failed");};
        input(prefix+"_system_rows.bin",rows);input(prefix+"_system_cols.bin",cols);input(prefix+"_system_values.bin",values);
        std::vector<double> inverse(9*nb);std::vector<int> seen(nb,0);double min_eigen=std::numeric_limits<double>::infinity();
        for(size_t k=0;k<entries;++k)if(rows[k]==cols[k]) {
            const int i=rows[k];if(i<0||i>=nb||seen[i]++)throw std::runtime_error("Duplicate or invalid diagonal block");
            const Matrix3x3 H=values[k];const double norm=H.norm();
            if(!H.allFinite()||norm==0||(H-H.transpose()).norm()>1e-12*norm)throw std::runtime_error("D block nonsymmetric/nonfinite");
            const Matrix3x3 S=(H+H.transpose())*.5;
            Eigen::SelfAdjointEigenSolver<Matrix3x3> eig(S);
            if(eig.info()!=Eigen::Success||eig.eigenvalues()[0]<=0)throw std::runtime_error("D block not positive definite");
            min_eigen=std::min(min_eigen,eig.eigenvalues()[0]);
            Eigen::LLT<Matrix3x3> factor(S);if(factor.info()!=Eigen::Success)throw std::runtime_error("D Cholesky failed");
            const Matrix3x3 inv=factor.solve(Matrix3x3::Identity());
            if(!inv.allFinite())throw std::runtime_error("D inverse nonfinite");
            for(int a=0;a<3;++a)for(int c=0;c<3;++c)inverse[9*i+3*a+c]=inv(a,c);
        }
        if(std::find(seen.begin(),seen.end(),0)!=seen.end())throw std::runtime_error("Missing diagonal block");
        reuse_diagonal_inverse.resize(inverse.size());
        CUDA_SAFE_CALL(cudaMemcpy(reuse_diagonal_inverse.data(),inverse.data(),inverse.size()*sizeof(double),cudaMemcpyHostToDevice));
        reuse_metric_z.resize(n);reuse_partials.resize(std::max(1,(n+255)/256));
        auto norm=[&](){reuse_diagonal_apply<<<(nb+255)/256,256>>>(reuse_diagonal_inverse.data(),b.data(),reuse_metric_z.data(),nb);
            return My_PCG_General_v_v_Reduction_Algorithm(reuse_partials.data(),b.data(),reuse_metric_z.data(),reduction_result.data(),n);};
        reuse_metric_b=norm();long double cpu=0;
        for(int i=0;i<nb;++i)for(int a=0;a<3;++a)for(int c=0;c<3;++c)
            cpu+=(long double)rhs[3*i+a]*inverse[9*i+3*a+c]*rhs[3*i+c];
        Json probes=Json::array();for(int rep=0;rep<10;++rep)probes.push_back(norm());
        double gap=0;for(const auto& value:probes)gap=std::max(gap,std::abs(value.get<double>()-reuse_metric_b)/std::max(std::abs(reuse_metric_b),1e-300));
        const double cpu_gap=std::abs(double(cpu)-reuse_metric_b)/std::max(std::abs(reuse_metric_b),1e-300);
        if(!std::isfinite(reuse_metric_b)||reuse_metric_b<=0||gap>1e-10||cpu_gap>1e-10)throw std::runtime_error("D norm repeat/CPU check failed");
        study["D_metric"]={{"blocks",nb},{"min_block_eigenvalue",min_eigen},{"b_norm",reuse_metric_b},
            {"b_norm_cpu",double(cpu)},{"cpu_relative_gap",cpu_gap},{"repeat_relative_gap",gap},{"probes",probes},
            {"preparation_and_validation_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()},
            {"preparation_scope","CPU snapshot extraction/checking plus upload; not a production build-time measurement"}};
        output(prefix+"_D_inverse.bin",inverse);
        for(size_t j=0;j<reuse_history.size();++j){const auto a=get(*reuse_history[j]);const std::string f=prefix+"_u"+std::to_string(j)+".bin";
            output(f,a);study["history_files"].push_back(f);}
        if(!reuse_history.empty()) {
            if(reuse_source_b.size()!=rhs.size()||reuse_source_au.size()!=rhs.size())throw std::runtime_error("Missing accepted source b/Au");
            output(prefix+"_source_b.bin",reuse_source_b);output(prefix+"_source_Au.bin",reuse_source_au);
            study["source_b_file"]=prefix+"_source_b.bin";study["source_Au_file"]=prefix+"_source_Au.bin";
        }
        if(options.phase=="DA") {
            if(reuse_history.empty())throw std::runtime_error("DA requires accepted history");
            std::map<int,int> limits;
            // The carrier is still native. Both interventions share this frozen
            // A/b/U; ordered restriction and native restriction are separate strata.
            for(int restriction:{2,0}) {
                std::vector<double> first;double gap=0;bool identical=true;
                for(int rep=0;rep<10;++rep) {
                    restore();research_restriction_override()=restriction;
                    r.buffer_view().copy_from(b.buffer_view());apply_preconditioner(z.view(),r.cview());
                    const auto v=get(z);if(rep==0)first=v;
                    long double d=0,s=0;for(size_t i=0;i<v.size();++i){const long double e=v[i]-first[i];d+=e*e;s+=(long double)first[i]*first[i];}
                    if(!std::isfinite(double(d))||!std::isfinite(double(s))||s<=0)throw std::runtime_error("DA M probe invalid");
                    gap=std::max(gap,std::sqrt(double(d/s)));identical=identical&&v==first;
                }
                study["M_probes"].push_back({{"restriction",restriction},{"repeats",10},{"bitwise_identical",identical},{"relative_to_first_max",gap}});
                if(restriction==2&&!identical)throw std::runtime_error("DA ordered M failed repeatability gate");
                restore();research_restriction_override()=restriction;reuse_metric_active=true;
                info=Json::object();attach_solve_context(info);Json row={{"restriction",restriction},{"arm","b0"}};
                try {pcg_reuse(x,b,max_iter,"b0",true);row["status"]="completed";limits[restriction]=info["recurrence_updates"].get<int>();
                    const std::string file=prefix+"_baseline_M"+std::to_string(restriction)+".bin";output(file,get(x));row["direction_file"]=file;
                }catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
                row["pcg"]=info;study["calibration_runs"].push_back(row);
            }
            for(double tolerance:{1e-10,1e-12})for(int rep=0;rep<2;++rep) {
                restore();research_restriction_override()=2;reuse_metric_active=true;m_config.global_tol_rate=tolerance;
                info=Json::object();attach_solve_context(info);const double gamma=tolerance==1e-10?1e-8:1e-9;
                Json row={{"D_tolerance",tolerance},{"gamma_target",gamma},{"repeat",rep+1},{"restriction",2}};
                try {pcg_reuse(x,b,max_iter,"b0",true,-1,gamma);row["status"]="completed";
                    const std::string file=prefix+"_ref"+std::to_string(study["reference_runs"].size())+".bin";output(file,get(x));row["direction_file"]=file;
                }catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
                row["pcg"]=info;study["reference_runs"].push_back(row);
            }
            for(int restriction:{2,0})for(int rep=0;rep<2;++rep)for(int slot=0;slot<2;++slot) {
                const std::string arm=(slot+rep)%2?"g4_aug":"g4";
                Json row={{"arm",arm},{"restriction",restriction},{"repeat",rep+1},{"warmup",false},{"diagnostic_curve_only",true}};
                if(!limits.count(restriction)){row["status"]="not_executed";row["error"]="B0 calibration failed";study["runs"].push_back(row);continue;}
                restore();research_restriction_override()=restriction;reuse_metric_active=true;reuse_curve_budget=limits[restriction];
                reuse_curve_prefix=prefix+"_M"+std::to_string(restriction)+"_"+arm+"_r"+std::to_string(rep+1);
                row["limit"]=reuse_curve_budget;info=Json::object();attach_solve_context(info);
                try {pcg_reuse(x,b,max_iter,arm,true);row["status"]="completed";}
                catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
                row["pcg"]=info;study["runs"].push_back(row);
            }
            finish();return;
        }
        // Every call starts from the same resident operator/scratch. The old-rho
        // bridge uses its own observed rho; it is not certified as a shared-rho arm.
        const std::vector<std::string> arms=options.decision_study()?std::vector<std::string>{"b0","b0_placebo","w1","g1","g4","g4_batch",
            options.phase=="DG1"?"probe_off":"cheap_policy",options.phase=="DG1"?"probe_on":"probe_policy"}:
            options.phase=="D0"?std::vector<std::string>{"b0_old","b0"}:
            options.phase=="DB"?std::vector<std::string>{"b0","b0_placebo","w1","g1","g4","g4_aug","g4_batch"}:
            options.phase=="DT"?std::vector<std::string>{"b0","b0_placebo","w1","g1","g4","g4_aug"}:
            std::vector<std::string>{"b0","b0_placebo","w1","g1","g4"};
        std::vector<std::vector<int>> order;
        for(int shift=0;shift<int(arms.size());++shift){std::vector<int> v;
            for(int j=0;j<int(arms.size());++j)v.push_back((j+shift)%arms.size());order.push_back(v);}
        if(arms.size()>=5){const auto forward=order;for(auto v:forward){std::reverse(v.begin(),v.end());order.push_back(v);}}
        std::mt19937 rng(options.decision_study()?20261118u:20261117u+unsigned(total_Frames+1)*97u+unsigned(reuse_solve_ordinal));std::shuffle(order.begin(),order.end(),rng);
        int curve_limit=-1;
        for(int rep=-1;rep<options.repeats;++rep)for(size_t pos=0;pos<arms.size();++pos) {
            const auto& arm=arms[rep<0?pos:order[rep%order.size()][pos]];restore();reuse_metric_active=arm!="b0_old";
            info=Json::object();attach_solve_context(info);Json row={{"arm",arm},{"repeat",rep+1},{"warmup",rep<0},{"order_position",pos}};
            try {
                CUDA_SAFE_CALL(cudaDeviceSynchronize());const auto service_started=std::chrono::steady_clock::now();
                pcg_reuse(x,b,max_iter,(arm=="b0_old"||arm=="b0_placebo")?"b0":arm,true);
                if(service_study) {
                    const auto stage_started=std::chrono::steady_clock::now();
                    if(arm!="b0"&&arm!="b0_placebo")reuse_stage(cudatool::CDenseVectorView<Float>{x.data(),int(x.size())},b,false);
                    CUDA_SAFE_CALL(cudaDeviceSynchronize());
                    row["history_stage_service_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage_started).count();
                    row["service_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-service_started).count();
                    row["service_scope"]="resident accepted history; Gram/projection/PCG/audit/fallback plus pending-output staging; excludes D/M builds, commit and diagnostic I/O";
                    reuse_discard();
                }
                row["status"]="completed";
                if(rep<0&&arm=="b0")curve_limit=info["recurrence_updates"].get<int>();
                if(rep>=0){const std::string f=prefix+"_"+arm+"_r"+std::to_string(rep+1)+".bin";
                    output(f,get(x));row["direction_file"]=f;}
            }catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
            row["pcg"]=info;study["runs"].push_back(row);
        }
        for(double tolerance:{1e-10,1e-12})for(int rep=0;rep<2;++rep) {
            restore();reuse_metric_active=true;m_config.global_tol_rate=tolerance;info=Json::object();attach_solve_context(info);
            const double gamma=tolerance==1e-10?1e-8:1e-9;
            Json row={{"D_tolerance",tolerance},{"gamma_target",gamma},{"repeat",rep+1}};
            try {pcg_reuse(x,b,max_iter,"b0",true,-1,gamma);row["status"]="completed";
                const std::string f=prefix+"_ref"+std::to_string(study["reference_runs"].size())+".bin";output(f,get(x));row["direction_file"]=f;
            }catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
            row["pcg"]=info;study["reference_runs"].push_back(row);
        }
        const std::vector<std::string> curve_arms=options.phase=="DG1"?std::vector<std::string>{"probe_off"}:
            options.phase=="DG2"?std::vector<std::string>{}:
            options.phase=="DB"?std::vector<std::string>{"g4_aug","g4_batch"}:
            options.phase=="DT"?std::vector<std::string>{"b0","g4","g4_aug"}:std::vector<std::string>{"b0","g1","g4"};
        if(options.phase!="D0"&&curve_limit>=0)for(const std::string& arm:curve_arms) {
            restore();reuse_metric_active=true;reuse_curve_budget=options.decision_study()?0:curve_limit;reuse_curve_prefix=prefix+"_curve_"+arm;
            info=Json::object();attach_solve_context(info);Json row={{"arm",arm},{"limit",curve_limit}};
            try {pcg_reuse(x,b,max_iter,arm,true);row["status"]="completed";}
            catch(const std::exception& e){row["status"]="failed";row["error"]=e.what();}
            row["pcg"]=info;study["curve_runs"].push_back(row);
        }
        finish();
    }catch(const std::exception& e){study["setup_or_export_error"]=e.what();finish();throw;}
}
} // namespace gipc
