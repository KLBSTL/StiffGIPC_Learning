#pragma once
#include <list>
#include <linear_system/utils/spmv.h>
#include <linear_system/utils/converter.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <cuda_tools/cuda_all.h>
#include <gipc/utils/json.h>


namespace gipc
{
class GlobalLinearSystem
{
    template <typename T>
    using U = std::unique_ptr<T>;
    friend class IterativeSolver;
    friend class IPreconditioner;
    friend class ILinearSubsystem;
    friend class LocalPreconditioner;

  public:
    GlobalLinearSystem() {}

    ~GlobalLinearSystem();

    static constexpr int BlockSize = 3;

    template <typename T, typename... Args>
    T& create(Args&&... args)
    {
        if constexpr(std::is_base_of_v<ILinearSubsystem, T>)
        {
            return static_cast<T&>(
                _create_subsystem(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IterativeSolver, T>)
        {
            return static_cast<T&>(
                _create_solver(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IPreconditioner, T>)
        {
            return static_cast<T&>(_create_preconditioner(
                std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else
        {
            CT_ASSERT(false, "Unknown type");
        }
    }

    /**
     * \brief solve the global linear system, using the specified solver
     * 
     * \details `solve_linear_system()` will:
     * - build the global linear system from the subsystems
     * - distribute the assembly assignments to the subsystems
     * - solve the linear system using the specified solver
     * - distribute the solution to the subsystems
     */
    gipc::SizeT solve_linear_system();

    struct GraphSpmvStats
    {
        uint64_t launches = 0;
        uint64_t recaptures = 0;
        uint64_t fallbacks = 0;
    };
    void enable_spmv_graph(bool enabled) { m_graph_spmv_enabled = enabled; }
    GraphSpmvStats graph_spmv_stats() const { return m_graph_spmv_stats; }
    struct PcgGraphStats
    {
        uint64_t launches = 0;
        uint64_t captures = 0;
        uint64_t fallbacks = 0;
        uint64_t cache_hits = 0;
        uint64_t invalidations = 0;
        uint64_t updates = 0;
        uint64_t rebuilds = 0;
        uint64_t update_rejects = 0;
        double instantiate_wall_ms = 0;
        double update_wall_ms = 0;
        uint64_t device_iterations = 0;
    };
    PcgGraphStats pcg_graph_stats() const;
    PcgGraphStats pcg_tail_stats() const;
    PcgGraphStats pcg_conditional_stats() const;
    struct PcgSolveReport
    {
        Float requested_rate = 0;
        Float initial_rho = 0;
        Float true_rho = 0;
        SizeT initial_iterations = 0;
        SizeT fallback_iterations = 0;
        bool baseline_fallback = false;
        bool verified = false;
        const char* initial_exit = "unavailable";
        const char* fallback_exit = "not_run";
    };
    void set_pcg_relative_rho_tolerance(Float rate);
    PcgSolveReport pcg_solve_report() const;

    Json               as_json() const;
    GIPCTripletMatrix* gipc_global_triplet = nullptr;

  private:
    std::vector<U<ILinearSubsystem>> m_subsystems;
    std::vector<DiagonalSubsystem*>  m_inner_subsystems;

    std::vector<U<LocalPreconditioner>> m_local_preconditioners;
    U<GlobalPreconditioner>             m_global_preconditioner;
    U<IterativeSolver>                  m_solver;

    cudatool::LinearSystemContext      m_context;
    cudatool::DeviceDenseVector<Float> m_x;
    cudatool::DeviceDenseVector<Float> m_b;

    std::vector<SizeT> m_rhs_count_per_subsystem;
    std::vector<SizeT> m_rhs_offset_per_subsystem;
    std::vector<Float> m_accuracy_statisfied_per_subsystem;

    size_t                         reserved_triplet_count = 0;
    Spmv                           m_spmv;
    struct GraphSpmvKey
    {
        const void* values = nullptr;
        const void* rows = nullptr;
        const void* cols = nullptr;
        const void* x = nullptr;
        const void* y = nullptr;
        int triplet_count = 0;
        SizeT vector_size = 0;
        bool operator==(const GraphSpmvKey& other) const
        {
            return values == other.values && rows == other.rows && cols == other.cols
                   && x == other.x && y == other.y
                   && triplet_count == other.triplet_count
                   && vector_size == other.vector_size;
        }
    };
    bool m_graph_spmv_enabled = false;
    GraphSpmvStats m_graph_spmv_stats;
    GraphSpmvKey m_spmv_graph_key;
    GraphSpmvKey m_graph_spmv_key;
    cudaGraph_t m_spmv_graph = nullptr;
    cudaGraphExec_t m_spmv_graph_exec = nullptr;
    void release_spmv_graph();
    Converter                      m_converter;
    cudatool::DeviceDenseVector<Float> fake_y;


    bool build_linear_system();
    void distribute_solution();
    void apply_preconditioner(cudatool::DenseVectorView<Float>  z,
                              cudatool::CDenseVectorView<Float> r);

    void convert_new();

    void spmv(Float a, cudatool::CDenseVectorView<Float> x, Float b, cudatool::DenseVectorView<Float> y);

    DiagonalSubsystem& _create_subsystem(U<DiagonalSubsystem>&& subsystem);

    IterativeSolver& _create_solver(U<IterativeSolver>&& solver);
    LocalPreconditioner& _create_preconditioner(U<LocalPreconditioner>&& preconditioner);
    GlobalPreconditioner& _create_preconditioner(U<GlobalPreconditioner>&& preconditioner);
};
}  // namespace gipc
