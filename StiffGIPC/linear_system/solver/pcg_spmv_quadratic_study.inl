// Diagnostic-only and excluded from timing claims. The acceptance bounds are
// defined in spmv_quadratic_reference.h before GPU experiments are performed.
#include <linear_system/utils/spmv_quadratic_reference.h>
#include <linear_system/utils/spmv.h>
namespace gipc
{
void PCGSolver::spmv_quadratic_fixed_study(cudatool::DenseVectorView<Float> x,
    cudatool::CDenseVectorView<Float> b,const std::string& prefix)
{
    if(!spmv_quadratic_supported || spmv_quadratic_partials_count<=0)
        throw std::runtime_error("SpMV quadratic study unavailable: "+spmv_quadratic_reason);
    static_assert(sizeof(Float)==sizeof(double),"SpMV study requires FP64");
    CostSampleGuard sample("frozen_operator_probe");
    CostScope scope("diagnostic.spmv_quadratic_including_audit");
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const auto saved_info=info;
    const auto before=snapshot_system(prefix);
    const auto signature=graph_signature(),saved_key=captured_key;
    const auto saved_graph=graph;const auto saved_exec=graph_exec;
    const auto saved_reduce_bytes=spmv_quadratic_reduce_bytes;
    const auto saved_max_iter=captured_max_iter;const auto saved_tol=captured_tol;
    const auto saved_captures=captures,saved_hits=cache_hits,saved_invalidations=invalidations;
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
    save(spmv_quadratic_partials.data(),spmv_quadratic_partials.size()*sizeof(Float));
    save(spmv_quadratic_reduce_storage.data(),spmv_quadratic_reduce_storage.size());
    mas_dot_scratch(save);
    auto restore=[&](){for(const auto& entry:saved)
        CUDA_SAFE_CALL(cudaMemcpy(entry.address,entry.bytes.data(),entry.bytes.size(),cudaMemcpyHostToDevice));};
    std::vector<Float> direction(x.size()),rhs(b.size());
    CUDA_SAFE_CALL(cudaMemcpy(direction.data(),x.data(),direction.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),b.data(),rhs.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    const size_t stored_blocks=before["blocks"].get<size_t>();
    std::vector<int> rows(stored_blocks),cols(stored_blocks);
    std::vector<double> matrix(stored_blocks*9);
    auto read=[&](const char* suffix,void* destination,size_t bytes){
        std::ifstream file(prefix+suffix,std::ios::binary);
        if(bytes)file.read(static_cast<char*>(destination),bytes);
        if(!file || file.peek()!=std::char_traits<char>::eof())
            throw std::runtime_error("SpMV study snapshot size/read failure");
    };
    read("_rows.bin",rows.data(),rows.size()*sizeof(int));
    read("_cols.bin",cols.data(),cols.size()*sizeof(int));
    read("_values.bin",matrix.data(),matrix.size()*sizeof(double));
    // CPU ownership checks include empty and both 32/256-thread boundaries.
    Json ownership=Json::array();
    bool ownership_passed=true;
    for(size_t count:{size_t(0),size_t(1),size_t(31),size_t(32),size_t(33),
                      size_t(255),size_t(256),size_t(257),stored_blocks})
    {
        const size_t partials=count?(count-1)/256+1:1;
        std::vector<unsigned char> writers(count,0);
        for(size_t block=0;block<partials;++block)for(size_t lane=0;lane<256;++lane)
        {const size_t k=block*256+lane;if(k<count)++writers[k];}
        const bool valid=std::all_of(writers.begin(),writers.end(),[](unsigned char n){return n==1;});
        ownership_passed=ownership_passed && valid;
        ownership.push_back({{"stored_blocks",count},{"partials",partials},{"unique_writer_per_block",valid}});
    }
    Json study={{"scope","Frozen stored-block SRBK Ap and quadratic; host and component Graph"},
        {"full_PCG_speed_or_quality_certified",false},{"system",before},
        {"cpu_long_double_digits",std::numeric_limits<long double>::digits},
        {"bound_policy","gamma_n per-coordinate absolute expanded terms; scalar absolute expanded terms; CPU reference included"},
        {"bitwise_Ap_is_diagnostic_only",true},{"partial_count",spmv_quadratic_partials_count},
        {"full_PCG_zero_rhs_reexecuted",false},{"production_graph_growth_invalidations_tested",false},
        {"zero_case_scope","Zero input exercises SpMV and pAp, not a second complete PCG solve; initial rho is unchanged by this component"},
        {"tail_scope","Actual frozen matrix runs on GPU; synthetic 0/1/31/32/33/255/256/257 ownership is CPU-only"},
        {"cpu_partial_ownership",ownership},{"cases",Json::array()},{"passed",ownership_passed}};
    attach_solve_context(study);
    const int n=static_cast<int>(x.size()),dot_blocks=(n-1)/256+1;
    cudatool::DeviceBuffer<Float> input(n),output(n),scalar(1),old_dot_partials(dot_blocks);
    size_t old_reduce_bytes=0;
    CUDA_SAFE_CALL(cub::DeviceReduce::Sum(nullptr,old_reduce_bytes,old_dot_partials.data(),
        scalar.data(),dot_blocks,cudaStreamPerThread));
    cudatool::DeviceBuffer<unsigned char> old_reduce_storage(old_reduce_bytes);
    const cudatool::CDenseVectorView<Float> v{input.data(),n};
    const cudatool::DenseVectorView<Float> y{output.data(),n};
    cudaGraph_t probe_graph=nullptr;cudaGraphExec_t probe_exec=nullptr;
    bool capturing=false;
    auto cleanup=[&](){
        if(probe_exec){cudaGraphExecDestroy(probe_exec);probe_exec=nullptr;}
        if(probe_graph){cudaGraphDestroy(probe_graph);probe_graph=nullptr;}
    };
    try
    {
        for(int test=0;test<3;++test)
        {
            auto values=direction;
            if(test==1)std::fill(values.begin(),values.end(),0.0);
            if(test==2)for(size_t i=0;i<values.size();++i)values[i]=(static_cast<int>(i%17)-8)*0.125;
            const auto ref=spmv_quadratic_reference(rows,cols,matrix,values);
            CUDA_SAFE_CALL(cudaMemcpy(input.data(),values.data(),values.size()*sizeof(Float),cudaMemcpyHostToDevice));
            spmv(v,y);
            PCG_vdv_Reduction<<<dot_blocks,256>>>(old_dot_partials.data(),input.data(),output.data(),n);
            CUDA_SAFE_CALL(cub::DeviceReduce::Sum(old_reduce_storage.data(),old_reduce_bytes,
                old_dot_partials.data(),scalar.data(),dot_blocks,cudaStreamPerThread));
            double old_q=0;
            std::vector<Float> old_ap(n);
            CUDA_SAFE_CALL(cudaMemcpy(&old_q,scalar.data(),sizeof(old_q),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(old_ap.data(),output.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
            auto check_ap=[&](const std::vector<Float>& actual){
                bool passed=true;long double max_error=0,max_ratio=0,max_bound=0;
                for(size_t i=0;i<actual.size();++i)
                {
                    const auto error=std::abs(static_cast<long double>(actual[i])-ref.ap[i]);
                    passed=passed && std::isfinite(actual[i]) && error<=ref.ap_bound[i];
                    max_error=std::max(max_error,error);max_bound=std::max(max_bound,ref.ap_bound[i]);
                    if(ref.ap_bound[i]>0)max_ratio=std::max(max_ratio,error/ref.ap_bound[i]);
                }
                return Json{{"passed",passed},{"max_absolute_error",static_cast<double>(max_error)},
                    {"max_per_coordinate_bound",static_cast<double>(max_bound)},
                    {"max_error_over_own_bound",static_cast<double>(max_ratio)}};
            };
            const auto old_check=check_ap(old_ap);
            const auto old_q_error=std::abs(static_cast<long double>(old_q)-ref.quadratic);
            const bool old_valid=old_check["passed"].get<bool>() && std::isfinite(old_q) && old_q_error<=ref.old_scalar_bound;
            Json item={{"input",test==0?"actual_solved_Newton_direction":test==1?"zero":"signed_tail_pattern"},
                {"scalar_count",n},{"stored_blocks",stored_blocks},{"lower_blocks",ref.lower_blocks},
                {"upper_blocks",ref.upper_blocks},{"diagonal_blocks",ref.diagonal_blocks},
                {"expanded_scalar_terms",ref.expanded_terms},{"cpu_quadratic",static_cast<double>(ref.quadratic)},
                {"quadratic_absolute_terms",static_cast<double>(ref.quadratic_absolute_terms)},
                {"fused_scalar_roundoff_bound",static_cast<double>(ref.fused_scalar_bound)},
                {"old_scalar_roundoff_bound",static_cast<double>(ref.old_scalar_bound)},
                {"old_Ap_reference",old_check},{"old_dot",old_q},
                {"old_dot_cpu_error",static_cast<double>(old_q_error)},{"old_passed",old_valid},
                {"runs",Json::array()}};
            if(!old_valid)study["passed"]=false;
            for(int mode=0;mode<3;++mode)
            {
                // A stale/no-op replay must not pass using the preceding
                // launch's result. All live outputs/partials start as NaNs.
                CUDA_SAFE_CALL(cudaMemsetAsync(output.data(),0xff,n*sizeof(Float),cudaStreamPerThread));
                CUDA_SAFE_CALL(cudaMemsetAsync(scalar.data(),0xff,sizeof(Float),cudaStreamPerThread));
                CUDA_SAFE_CALL(cudaMemsetAsync(spmv_quadratic_partials.data(),0xff,
                    spmv_quadratic_partials_count*sizeof(Float),cudaStreamPerThread));
                if(mode==0)apply_spmv_quadratic(v,y,scalar.data());
                else
                {
                    if(!probe_exec)
                    {
                        CostCaptureGuard guard;
                        CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                        capturing=true;
                        apply_spmv_quadratic(v,y,scalar.data());
                        CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&probe_graph));
                        capturing=false;
                        CUDA_SAFE_CALL(cudaGraphInstantiate(&probe_exec,probe_graph,nullptr,nullptr,0));
                    }
                    CUDA_SAFE_CALL(cudaGraphLaunch(probe_exec,cudaStreamPerThread));
                }
                double q=0;std::vector<Float> actual(n);
                CUDA_SAFE_CALL(cudaMemcpy(&q,scalar.data(),sizeof(q),cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(actual.data(),output.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
                const auto checked=check_ap(actual);
                const bool exact=std::memcmp(old_ap.data(),actual.data(),n*sizeof(Float))==0;
                bool pair_ap_passed=true;long double pair_max=0;
                for(int i=0;i<n;++i)
                {
                    const auto error=std::abs(static_cast<long double>(actual[i])-old_ap[i]);
                    pair_ap_passed=pair_ap_passed && std::isfinite(actual[i]) && error<=2*ref.ap_bound[i];
                    pair_max=std::max(pair_max,error);
                }
                const auto error=std::abs(static_cast<long double>(q)-ref.quadratic);
                const auto pair_error=std::abs(static_cast<long double>(q)-old_q);
                const bool valid=old_valid && checked["passed"].get<bool>() && pair_ap_passed
                    && std::isfinite(q) && error<=ref.fused_scalar_bound
                    && pair_error<=ref.fused_scalar_bound+ref.old_scalar_bound;
                item["runs"].push_back({{"execution",mode==0?"host_launch":mode==1?"graph_first":"graph_replay"},
                    {"Ap_bitwise_equal",exact},{"Ap_cpu_reference",checked},{"Ap_pair_within_bounds",pair_ap_passed},
                    {"Ap_pair_max_absolute_error",static_cast<double>(pair_max)},
                    {"quadratic",q},{"quadratic_cpu_error",static_cast<double>(error)},
                    {"quadratic_old_dot_error",static_cast<double>(pair_error)},{"passed",valid}});
                if(!valid)study["passed"]=false;
            }
            study["cases"].push_back(item);
        }
        cleanup();
        // A truly empty private matrix (not just zero input for nonzero A).
        // It neither replaces the production matrix nor uses its nnz count.
        study["empty_matrix_fixture"]=Json::array();
        Spmv empty_operator;
        for(int mode=0;mode<3;++mode)
        {
            CUDA_SAFE_CALL(cudaMemsetAsync(output.data(),0xff,n*sizeof(Float),cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(scalar.data(),0xff,sizeof(Float),cudaStreamPerThread));
            auto run_empty=[&](){empty_operator.warp_reduce_sym_spmv_quadratic(
                nullptr,nullptr,nullptr,0,v,y,scalar.data());};
            if(mode==0)run_empty();
            else
            {
                if(!probe_exec)
                {
                    CostCaptureGuard guard;
                    CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                    capturing=true;run_empty();
                    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&probe_graph));
                    capturing=false;
                    CUDA_SAFE_CALL(cudaGraphInstantiate(&probe_exec,probe_graph,nullptr,nullptr,0));
                }
                CUDA_SAFE_CALL(cudaGraphLaunch(probe_exec,cudaStreamPerThread));
            }
            double q=0;std::vector<Float> empty_ap(n);
            CUDA_SAFE_CALL(cudaMemcpy(&q,scalar.data(),sizeof(q),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(empty_ap.data(),output.data(),n*sizeof(Float),cudaMemcpyDeviceToHost));
            const bool valid=std::isfinite(q) && q==0 && std::all_of(empty_ap.begin(),empty_ap.end(),
                [](double value){return std::isfinite(value) && value==0;});
            study["empty_matrix_fixture"].push_back({{"execution",mode==0?"host_launch":mode==1?"graph_first":"graph_replay"},
                {"scope","Private empty Spmv wrapper with direct single partial; not a complete zero-RHS PCG solve"},
                {"stored_blocks",0},{"quadratic",q},{"Ap_and_quadratic_finite_zero",valid},{"passed",valid}});
            if(!valid)study["passed"]=false;
        }
        cleanup();restore();
    }
    catch(...)
    {
        if(capturing){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(cudaStreamPerThread,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}
        cleanup();restore();info=saved_info;throw;
    }
    bool restored=true;
    for(const auto& entry:saved)
    {
        std::vector<unsigned char> actual(entry.bytes.size());
        CUDA_SAFE_CALL(cudaMemcpy(actual.data(),entry.address,actual.size(),cudaMemcpyDeviceToHost));
        restored=restored && actual==entry.bytes;
    }
    std::vector<Float> rhs_after(rhs.size());
    CUDA_SAFE_CALL(cudaMemcpy(rhs_after.data(),b.data(),rhs_after.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    const auto after=snapshot_system("");
    const bool rhs_equal=std::memcmp(rhs.data(),rhs_after.data(),rhs.size()*sizeof(Float))==0;
    const bool graph_equal=signature==graph_signature() && saved_key==captured_key && saved_graph==graph
        && saved_exec==graph_exec && saved_max_iter==captured_max_iter && saved_tol==captured_tol
        && captures==saved_captures && cache_hits==saved_hits && invalidations==saved_invalidations
        && saved_reduce_bytes==spmv_quadratic_reduce_bytes;
    study["all_work_buffers_restored_bitwise"]=restored;
    study["rhs_unchanged_bitwise"]=rhs_equal;
    study["system_and_MAS_scratch_unchanged"]=before==after;
    study["production_graph_unchanged"]=graph_equal;
    study["passed"]=study["passed"].get<bool>() && restored && rhs_equal && before==after && graph_equal;
    info=saved_info;info["spmv_quadratic_study"]=study;
    info["fixed_study_file"]=prefix+"_spmv_quadratic_study.json";
    std::ofstream file(prefix+"_spmv_quadratic_study.json");file<<study.dump(2);file.close();
    if(!file || !study["passed"].get<bool>())throw std::runtime_error("Protected SpMV quadratic study failed");
}
}
