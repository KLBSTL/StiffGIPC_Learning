// Diagnostic only: two graph schedules, one assembled A/b/M, no checkpoint reload.
// Included by pcg_solver.cu after the ordinary solver implementations.
#include <array>
#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

namespace gipc
{
void PCGSolver::graph_chunk_study(cudatool::DenseVectorView<Float> x,
                                 cudatool::CDenseVectorView<Float> b,
                                 SizeT max_iter, const std::string& prefix)
{
    static_assert(std::is_nothrow_move_constructible<DeviceDenseVector>::value
                  && std::is_nothrow_move_assignable<DeviceDenseVector>::value,
                  "Study must move buffer ownership without copying/freeing production data");
    using Clock = std::chrono::steady_clock;
    auto milliseconds = [](Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    };
    auto check = [](cudaError_t result) {
        if(result!=cudaSuccess)throw std::runtime_error(
            std::string("Graph chunk study CUDA: ")+cudaGetErrorString(result));
    };
    if(b.size()==0 || b.size()>std::numeric_limits<int>::max()-255 || x.size()!=b.size()
       || max_iter<=1 || max_iter>std::numeric_limits<int>::max())
        throw std::runtime_error("Graph chunk study requires a nonempty graph-supported system");
    if(diagnostic_fixed_iterations!=0 || m_config.global_tol_rate!=Float(1e-4))
        throw std::runtime_error("Graph chunk study requires the unchanged default rho stopping rule");

    check(cudaStreamSynchronize(cudaStreamPerThread));
    const auto setup_start=Clock::now();
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    const Json saved_info=info;
    const auto saved_config=m_config;
    const int saved_fixed=diagnostic_fixed_iterations, saved_fused=diagnostic_fused_override;
    const bool saved_chunk_timing=diagnostic_chunk_timing;
    const auto saved_context=solve_context();
    Json saved_context_json;attach_solve_context(saved_context_json);
    auto& trace=cost_trace_state();
    const int saved_depth=trace.capture_depth, saved_iteration=trace.iteration;
    const char* saved_sample=trace.sample_kind;
    const auto saved_trace_next=trace.next_id;
    const auto saved_trace_stack=trace.stack;
    const auto saved_trace_pending=trace.pending.size();
    const auto saved_signature=graph_signature();
    auto read = [&](const Float* data,size_t count) {
        std::vector<Float> result(count);
        if(count)check(cudaMemcpy(result.data(),data,count*sizeof(Float),cudaMemcpyDeviceToHost));
        return result;
    };
    auto digest = [&](const void* data,size_t bytes) {
        std::vector<unsigned char> host(bytes);
        if(bytes)check(cudaMemcpy(host.data(),data,bytes,cudaMemcpyDeviceToHost));
        std::uint64_t hash=14695981039346656037ull;
        for(unsigned char value:host){hash^=value;hash*=1099511628211ull;}
        return Json{{"bytes",bytes},{"fnv1a64",hash}};
    };
    auto workspace_identity = [&]() {
        Json result;
        auto add=[&](const char* name,const auto& buffer) {
            using Value=typename std::decay_t<decltype(buffer)>::value_type;
            result[name]=digest(buffer.data(),buffer.size()*sizeof(Value));
            result[name]["address"]=reinterpret_cast<std::uintptr_t>(buffer.data());
            result[name]["size"]=buffer.size();result[name]["capacity"]=buffer.capacity();
        };
        add("r",r);add("z",z);add("p",p);add("Ap",Ap);
        add("reduction_result",reduction_result);add("graph_scalars",graph_scalars);
        add("graph_reduce_storage",graph_reduce_storage);
        return result;
    };
    const auto primary=read(x.data(),x.size());
    const auto rhs=read(b.data(),b.size());
    long double rhs_norm2=0;
    for(Float value:rhs)
    {
        if(!std::isfinite(value))throw std::runtime_error("Graph chunk study nonfinite RHS");
        rhs_norm2+=static_cast<long double>(value)*value;
    }
    if(!(rhs_norm2>0) || !std::isfinite(rhs_norm2))
        throw std::runtime_error("Graph chunk timing requires a finite nonzero RHS; use guard fixture for zero RHS");
    const auto before=snapshot_system(prefix);
    if(!before.value("preconditioner_export_complete",false))
        throw std::runtime_error("Graph chunk study requires GIPC_MAS_SNAPSHOT=1 and complete M export");
    bool has_legacy_mas=false;
    for(const auto& local:before.at("local_preconditioners"))
        if(local.value("kind",std::string{})=="MAS_full_owned_buffers")
        {
            if(local.value("wide_apply",true) || local.value("inverse64",true)
               || local.value("cholesky",true))
                throw std::runtime_error("Graph chunk study is restricted to frozen legacy MAS");
            has_legacy_mas=true;
        }
    if(!has_legacy_mas)throw std::runtime_error("Graph chunk study requires a legacy MAS subsystem");
    const auto production_workspace=workspace_identity();
    auto restore_m=checkpoint_preconditioner_scratch();

    struct Workspace
    {
        DeviceDenseVector r,z,p,Ap;
        cudatool::DeviceBuffer<Float> reduction,scalars;
        cudatool::DeviceBuffer<unsigned char> reduce_storage;
    } private_workspace;
    const int n=static_cast<int>(b.size()),blocks=(n+255)/256;
    private_workspace.r.resize(n);private_workspace.z.resize(n);
    private_workspace.p.resize(n);private_workspace.Ap.resize(n);
    private_workspace.reduction.resize(1);
    // K1 has 10 scalars; K4 has 13. Preallocate the maximum capacity before
    // either capture. pcg_graph may resize the logical range, not its address.
    private_workspace.scalars.resize(std::max<size_t>(13,graph_scalars.size()));
    check(cudaMemset(private_workspace.scalars.data(),0,private_workspace.scalars.size()*sizeof(Float)));
    size_t reduce_bytes=0;
    check(cub::DeviceReduce::Sum(nullptr,reduce_bytes,private_workspace.p.data(),
                                private_workspace.scalars.data(),blocks,cudaStreamPerThread));
    private_workspace.reduce_storage.resize(reduce_bytes);
    DeviceDenseVector work_x(n),residual_ax(n);
    struct GraphState
    {
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        std::vector<std::uintptr_t> key;
        SizeT max_iter=0;Float tol=0;
        std::uint64_t captures=0,hits=0,invalidations=0;
    } production_graph,arms[2];
    auto exchange_graph=[&](GraphState& state) {
        using std::swap;
        swap(graph,state.graph);swap(graph_exec,state.exec);captured_key.swap(state.key);
        swap(captured_max_iter,state.max_iter);swap(captured_tol,state.tol);
        swap(captures,state.captures);swap(cache_hits,state.hits);swap(invalidations,state.invalidations);
    };
    auto exchange_workspace=[&]() {
        using std::swap;
        swap(r,private_workspace.r);swap(z,private_workspace.z);
        swap(p,private_workspace.p);swap(Ap,private_workspace.Ap);
        swap(reduction_result,private_workspace.reduction);
        swap(graph_scalars,private_workspace.scalars);
        swap(graph_reduce_storage,private_workspace.reduce_storage);
    };
    auto graph_identity=[&]() {
        return Json{{"graph",reinterpret_cast<std::uintptr_t>(graph)},
            {"exec",reinterpret_cast<std::uintptr_t>(graph_exec)},
            {"key",captured_key},{"max_iter",captured_max_iter},{"tol",captured_tol},
            {"captures",captures},{"hits",cache_hits},{"invalidations",invalidations}};
    };
    const auto production_graph_identity=graph_identity();
    struct Events
    {
        cudaEvent_t begin=nullptr,end=nullptr;
        ~Events(){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
    } events;
    check(cudaEventCreate(&events.begin));check(cudaEventCreate(&events.end));
    Json study={{"schema","fixed_graph_chunk_v1"},{"system",before},
        {"direction",Statistics::instance().at_current_frame()["newton"].size()},
        {"zero_initial_guess",true},{"mas","legacy"},{"rho_tolerance",saved_config.global_tol_rate},
        {"max_iter",max_iter},{"warmup_pairs",2},{"measured_pairs",7},
        {"runs",Json::array()},{"passed",false},{"production_workspace_before",production_workspace},
        {"pass_scope","bounded solves, cache reuse, finite outputs and production restoration; numerical equivalence and speed require separate analysis"},
        {"production_graph_before",production_graph_identity},
        {"residual_method","GPU full A*x; host norm (long double digits recorded), not independent CPU reference"},
        {"host_long_double_digits",std::numeric_limits<long double>::digits},
        {"event_scope","whole stream solve, including host submission gaps; not sum of GPU kernels"},
        {"preconditioner_preparation","frozen before study; not remeasured or claimed zero"}};
    attach_solve_context(study);
    const std::string report_path=prefix+"_graph_chunk_study.json";
    auto write_report=[&]() {
        std::ofstream output(report_path);output<<study.dump(2);output.close();
        if(!output)throw std::runtime_error("Cannot write graph chunk study report");
    };
    bool installed=false,restored=false;
    std::string restore_error;
    auto restore=[&]() noexcept {
        if(restored)return;
        auto attempt=[&](const std::function<void()>& operation) {
            try{operation();}catch(const std::exception& e){if(restore_error.empty())restore_error=e.what();}
            catch(...){if(restore_error.empty())restore_error="unknown restoration failure";}
        };
        attempt([&]{check(cudaStreamSynchronize(cudaStreamPerThread));});
        if(installed)
        {
            // An exception may leave one private arm installed in the solver.
            attempt([&]{if(graph_exec)check(cudaGraphExecDestroy(graph_exec));});graph_exec=nullptr;
            attempt([&]{if(graph)check(cudaGraphDestroy(graph));});graph=nullptr;
            for(auto& arm:arms)
            {
                attempt([&]{if(arm.exec)check(cudaGraphExecDestroy(arm.exec));});arm.exec=nullptr;
                attempt([&]{if(arm.graph)check(cudaGraphDestroy(arm.graph));});arm.graph=nullptr;
            }
            // Clear private metadata before swapping the exact production state.
            captured_key.clear();captured_max_iter=0;captured_tol=0;captures=cache_hits=invalidations=0;
            exchange_graph(production_graph);exchange_workspace();installed=false;
        }
        attempt([&]{restore_m();});
        attempt([&]{check(cudaMemcpy(x.data(),primary.data(),primary.size()*sizeof(Float),cudaMemcpyHostToDevice));});
        m_config=saved_config;diagnostic_fixed_iterations=saved_fixed;diagnostic_fused_override=saved_fused;
        diagnostic_chunk_timing=saved_chunk_timing;
        solve_context()=saved_context;trace.capture_depth=saved_depth;
        trace.iteration=saved_iteration;trace.sample_kind=saved_sample;
        attempt([&]{info=saved_info;});
        restored=true;
    };
    struct RestoreGuard
    {
        std::function<void()> restore;
        ~RestoreGuard(){restore();}
    } restore_guard{restore};
    auto record_restoration=[&]() {
        study["restoration_error"]=restore_error;
        const auto after=snapshot_system("");
        study["system_after"]=after;study["full_system_restored"]=before==after;
        const auto returned=read(x.data(),x.size());
        study["primary_restored_bitwise"]=std::memcmp(primary.data(),returned.data(),primary.size()*sizeof(Float))==0;
        study["production_workspace_after"]=workspace_identity();
        study["workspace_restored"]=study["production_workspace_after"]==production_workspace;
        study["production_graph_after"]=graph_identity();
        study["production_graph_restored"]=study["production_graph_after"]==production_graph_identity;
        study["operator_signature_restored"]=graph_signature()==saved_signature;
        study["production_info_restored"]=info==saved_info;
        Json context;attach_solve_context(context);
        study["solve_context_restored"]=context==saved_context_json;
        study["config_restored"]=m_config.max_iter_ratio==saved_config.max_iter_ratio
            &&m_config.global_tol_rate==saved_config.global_tol_rate&&m_config.use_bsr==saved_config.use_bsr
            &&m_config.conditional_graph==saved_config.conditional_graph
            &&m_config.graph_chunk_iterations==saved_config.graph_chunk_iterations
            &&diagnostic_fixed_iterations==saved_fixed&&diagnostic_fused_override==saved_fused
            &&diagnostic_chunk_timing==saved_chunk_timing;
        study["cost_trace_restored"]=trace.capture_depth==saved_depth&&trace.iteration==saved_iteration
            &&trace.sample_kind==saved_sample&&trace.next_id==saved_trace_next
            &&trace.stack==saved_trace_stack&&trace.pending.size()==saved_trace_pending;
    };
    try
    {
        exchange_workspace();exchange_graph(production_graph);installed=true;
        // Suppress ordinary cost scopes: the private event pair measures this
        // study without appending diagnostic intervals to production telemetry.
        trace.capture_depth=saved_depth+1;trace.sample_kind="fixed_graph_chunk";
        diagnostic_fixed_iterations=0;diagnostic_fused_override=0;
        diagnostic_chunk_timing=true;
        m_config.conditional_graph=true;
        study["private_setup_host_ms"]=milliseconds(setup_start);
        std::array<int,2> uses{{0,0}};
        for(int pair=-2;pair<7;++pair)
        {
            const bool warmup=pair<0;
            const int sequence=warmup?pair+2:pair;
            for(int order=0;order<2;++order)
            {
                const int arm=(sequence%2==0)?order:1-order;
                const int chunk=arm==0?1:4;
                const auto reset_start=Clock::now();
                restore_m();check(cudaMemsetAsync(work_x.data(),0,n*sizeof(Float),cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));
                const double reset_ms=milliseconds(reset_start);
                m_config.graph_chunk_iterations=chunk;
                info=Json::object();exchange_graph(arms[arm]);
                check(cudaEventRecord(events.begin,cudaStreamPerThread));
                const auto start=Clock::now();
                const auto iterations=pcg_graph(work_x.view(),b,max_iter);
                const double host_ms=milliseconds(start);
                check(cudaEventRecord(events.end,cudaStreamPerThread));check(cudaEventSynchronize(events.end));
                float event_ms=0;check(cudaEventElapsedTime(&event_ms,events.begin,events.end));
                Json record={{"warmup",warmup},{"pair",warmup?pair+3:pair+1},{"order",order+1},
                    {"chunk",chunk},{"arm_use",uses[arm]+1},{"cold",uses[arm]==0},
                    {"solve_host_ms",host_ms},{"solve_event_ms",event_ms},{"reset_host_ms",reset_ms},
                    {"iterations",iterations},{"pcg",info},{"passed",false}};
                record["cache_hit"]=info.value("graph_cache_hit",false);
                record["capture_instantiate_host_ms"]=info.value("graph_capture_instantiate_host_ms",0.0);
                record["final_graph_scalars"]=read(graph_scalars.data(),graph_scalars.size());
                record["graph_scalar_count"]=graph_scalars.size();
                for(const char* field:{"graph_initialization_event_ms","graph_replay_event_ms",
                                      "graph_initial_readback_host_ms","graph_final_readback_host_ms"})
                    record[field]=info.contains(field)?info[field]:Json(nullptr);
                const bool cache_ok=uses[arm]==0?!record["cache_hit"].get<bool>():record["cache_hit"].get<bool>();
                record["cache_contract_passed"]=cache_ok;
                ++uses[arm];exchange_graph(arms[arm]);
                const auto export_start=Clock::now();
                const auto solution=read(work_x.data(),work_x.size());
                spmv(work_x.cview(),residual_ax.view());
                const auto ax=read(residual_ax.data(),residual_ax.size());
                long double residual2=0;bool finite=true;
                for(size_t i=0;i<solution.size();++i)
                {
                    finite=finite&&std::isfinite(solution[i])&&std::isfinite(ax[i]);
                    const long double error=static_cast<long double>(rhs[i])-ax[i];residual2+=error*error;
                }
                const double residual=std::sqrt(static_cast<double>(residual2/rhs_norm2));
                finite=finite&&std::isfinite(residual)&&std::isfinite(host_ms)&&std::isfinite(event_ms);
                record["true_relative_residual"]=std::isfinite(residual)?Json(residual):Json(nullptr);
                record["finite"]=finite;
                record["solution_identity"]=digest(work_x.data(),work_x.size()*sizeof(Float));
                if(!warmup)
                {
                    const std::string path=prefix+"_k"+std::to_string(chunk)+"_pair"+std::to_string(pair+1)+"_x.bin";
                    std::ofstream output(path,std::ios::binary);
                    output.write(reinterpret_cast<const char*>(solution.data()),solution.size()*sizeof(Float));output.close();
                    if(!output)throw std::runtime_error("Cannot write graph chunk solution");
                    record["solution_file"]=path;
                }
                record["export_and_residual_host_ms"]=milliseconds(export_start);
                const bool passed=finite&&cache_ok&&iterations<max_iter
                    &&!info.value("iteration_limit",false)&&!info.contains("breakdown");
                record["passed"]=passed;study["runs"].push_back(std::move(record));
                if(!passed)throw std::runtime_error("Graph chunk solve/cache/finite contract failed");
            }
        }
        const auto restore_start=Clock::now();restore();
        study["restore_host_ms"]=milliseconds(restore_start);
        record_restoration();
        study["passed"]=restore_error.empty()&&study["runs"].size()==18
            &&study["full_system_restored"].get<bool>()&&study["primary_restored_bitwise"].get<bool>()
            &&study["workspace_restored"].get<bool>()&&study["production_graph_restored"].get<bool>()
            &&study["operator_signature_restored"].get<bool>()&&study["production_info_restored"].get<bool>()
            &&study["solve_context_restored"].get<bool>()&&study["config_restored"].get<bool>()
            &&study["cost_trace_restored"].get<bool>();
        write_report();
        if(!study["passed"].get<bool>())throw std::runtime_error("Graph chunk study restoration failed");
        info["graph_chunk_study_file"]=report_path;
    }
    catch(...)
    {
        const auto failure=std::current_exception();
        study["failed_run_info"]=info;restore();study["passed"]=false;study["restoration_error"]=restore_error;
        try{record_restoration();}catch(const std::exception& e){study["restoration_proof_error"]=e.what();}
        catch(...){study["restoration_proof_error"]="unknown restoration proof failure";}
        try{std::rethrow_exception(failure);}catch(const std::exception& e){study["error"]=e.what();}
        catch(...){study["error"]="unknown graph chunk study failure";}
        try{write_report();}catch(...){} // Preserve the original solver error.
        std::rethrow_exception(failure);
    }
}
} // namespace gipc
