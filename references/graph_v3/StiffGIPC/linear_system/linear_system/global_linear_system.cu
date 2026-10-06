#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/solver/pcg_solver.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/timer.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

extern bool experimental_pcg_conditional_mas;
extern std::string benchmark_output_dir;

namespace gipc
{
namespace
{
bool linear_stage_audit_enabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GIPC_LINEAR_STAGE_AUDIT");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled && !benchmark_output_dir.empty();
}

double build_rhs_ms = 0;
double build_convert_ms = 0;
double build_preconditioner_ms = 0;
}  // namespace

bool GlobalLinearSystem::build_linear_system()
{
    const bool audit = linear_stage_audit_enabled();
    const auto build_start = std::chrono::steady_clock::now();
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    if(audit) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto convert_start = std::chrono::steady_clock::now();
    convert_new();
    if(audit) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto preconditioner_start = std::chrono::steady_clock::now();

    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);

    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }

    if(audit)
    {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        const auto build_end = std::chrono::steady_clock::now();
        const auto ms = [](auto end, auto begin) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        build_rhs_ms = ms(convert_start, build_start);
        build_convert_ms = ms(preconditioner_start, convert_start);
        build_preconditioner_ms = ms(build_end, preconditioner_start);
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

void GlobalLinearSystem::release_spmv_graph()
{
    if(m_spmv_graph_exec)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaGraphExecDestroy(m_spmv_graph_exec));
        m_spmv_graph_exec = nullptr;
    }
    if(m_spmv_graph)
    {
        CUDA_SAFE_CALL(cudaGraphDestroy(m_spmv_graph));
        m_spmv_graph = nullptr;
    }
    m_spmv_graph_key = {};
}

GlobalLinearSystem::PcgGraphStats GlobalLinearSystem::pcg_graph_stats() const
{
    const auto* pcg = dynamic_cast<const PCGSolver*>(m_solver.get());
    if(!pcg) return {};
    const auto stats = pcg->graph_direction_stats();
    return {stats.launches, stats.captures, stats.fallbacks,
            stats.cache_hits, stats.invalidations, stats.updates,
            stats.rebuilds, stats.update_rejects,
            stats.instantiate_wall_ms, stats.update_wall_ms};
}

GlobalLinearSystem::PcgGraphStats GlobalLinearSystem::pcg_tail_stats() const
{
    const auto* pcg = dynamic_cast<const PCGSolver*>(m_solver.get());
    if(!pcg) return {};
    const auto stats = pcg->graph_tail_stats();
    return {stats.launches, stats.captures, stats.fallbacks,
            stats.cache_hits, stats.invalidations, stats.updates,
            stats.rebuilds, stats.update_rejects,
            stats.instantiate_wall_ms, stats.update_wall_ms};
}

GlobalLinearSystem::PcgGraphStats GlobalLinearSystem::pcg_conditional_stats() const
{
    const auto* pcg = dynamic_cast<const PCGSolver*>(m_solver.get());
    if(!pcg) return {};
    const auto stats = pcg->graph_conditional_stats();
    return {stats.launches, stats.captures, stats.fallbacks,
            stats.cache_hits, stats.invalidations, stats.updates,
            stats.rebuilds, stats.update_rejects,
            stats.instantiate_wall_ms, stats.update_wall_ms,
            stats.device_iterations};
}

void GlobalLinearSystem::set_pcg_relative_rho_tolerance(Float rate)
{
    auto* pcg = dynamic_cast<PCGSolver*>(m_solver.get());
    if(!pcg || !std::isfinite(rate) || rate <= 0 || rate > 1)
        throw std::runtime_error("Invalid adaptive PCG relative rho tolerance");
    pcg->set_global_tol_rate(rate);
}

GlobalLinearSystem::PcgSolveReport GlobalLinearSystem::pcg_solve_report() const
{
    const auto* pcg = dynamic_cast<const PCGSolver*>(m_solver.get());
    if(!pcg) return {};
    const auto r = pcg->solve_report();
    return {r.requested_rate, r.initial_rho, r.true_rho,
            r.initial_iterations, r.fallback_iterations,
            r.baseline_fallback, r.verified, r.initial_exit, r.fallback_exit};
}

GlobalLinearSystem::~GlobalLinearSystem()
{
    release_spmv_graph();
}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    CT_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    const bool record_stage = linear_stage_audit_enabled();
    const auto stage_start = std::chrono::steady_clock::now();
    bool success = build_linear_system();
    if(!success)
        return 0;
    if(record_stage) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto solve_start = std::chrono::steady_clock::now();
    CT_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    auto iter = m_solver->solve(m_x, m_b);
    if(record_stage) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto distribute_start = std::chrono::steady_clock::now();
    distribute_solution();
    if(record_stage)
    {
        const auto stage_end = std::chrono::steady_clock::now();
        static uint64_t solve_index = 0;
        std::ofstream csv(benchmark_output_dir + "/linear_solver_stage_audit.csv",
                          solve_index == 0 ? std::ios::trunc : std::ios::app);
        if(solve_index == 0)
            csv << "solve_index,build_ms,rhs_ms,convert_ms,preconditioner_ms,pcg_ms,distribute_ms,pcg_iterations\n";
        const auto ms = [](auto end, auto begin) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        csv << ++solve_index << ',' << ms(solve_start, stage_start)
            << ',' << build_rhs_ms << ',' << build_convert_ms
            << ',' << build_preconditioner_ms
            << ',' << ms(distribute_start, solve_start)
            << ',' << ms(stage_end, distribute_start) << ',' << iter << '\n';
        if(!csv) throw std::runtime_error("Cannot write linear solver stage audit");
    }
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(cudatool::DenseVectorView<Float>  z,
                                              cudatool::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else if(experimental_pcg_conditional_mas)
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            z.data(), r.data(), std::min(z.size(), r.size()) * sizeof(Float),
            cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



void GlobalLinearSystem::convert_new()
{
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
//#ifndef SymGH
//    m_converter.ge2sym(*gipc_global_triplet);
//#endif
}



void GlobalLinearSystem::spmv(Float                         a,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y)
{
    auto* values = gipc_global_triplet->block_values();
    auto* rows = gipc_global_triplet->block_row_indices();
    auto* cols = gipc_global_triplet->block_col_indices();
    const int count = gipc_global_triplet->h_unique_key_number;
    auto ordinary_spmv = [&]()
    {
        m_spmv.warp_reduce_sym_spmv(a, values, rows, cols, count, x, b, y);
    };
    if(!m_graph_spmv_enabled || a != 1.0 || b != 0.0 || count <= 0 || y.size() == 0)
    {
        ordinary_spmv();
        return;
    }

    const GraphSpmvKey key{values, rows, cols, x.data(), y.data(), count, static_cast<SizeT>(y.size())};
    if(!m_spmv_graph_exec || !(key == m_spmv_graph_key))
    {
        release_spmv_graph();
        // CUDA kernels use the per-thread default stream in this build.
        auto capture = cudaStreamBeginCapture(cudaStreamPerThread,
                                              cudaStreamCaptureModeThreadLocal);
        if(capture != cudaSuccess)
        {
            cudaGetLastError();
            ++m_graph_spmv_stats.fallbacks;
            ordinary_spmv();
            return;
        }
        ordinary_spmv();
        capture = cudaStreamEndCapture(cudaStreamPerThread, &m_spmv_graph);
        if(capture != cudaSuccess || !m_spmv_graph)
        {
            cudaGetLastError();
            if(m_spmv_graph)
                cudaGraphDestroy(m_spmv_graph);
            m_spmv_graph = nullptr;
            ++m_graph_spmv_stats.fallbacks;
            ordinary_spmv();
            return;
        }
        capture = cudaGraphInstantiate(&m_spmv_graph_exec, m_spmv_graph, nullptr, nullptr, 0);
        if(capture != cudaSuccess)
        {
            cudaGetLastError();
            release_spmv_graph();
            ++m_graph_spmv_stats.fallbacks;
            ordinary_spmv();
            return;
        }
        m_spmv_graph_key = key;
        ++m_graph_spmv_stats.recaptures;
    }
    CUDA_SAFE_CALL(cudaGraphLaunch(m_spmv_graph_exec, cudaStreamPerThread));
    ++m_graph_spmv_stats.launches;
}
}  // namespace gipc
