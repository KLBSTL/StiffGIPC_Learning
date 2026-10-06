#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>

namespace gipc
{
class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
    bool  device_scalar_nograph = false;
    bool  graph_direction_segment = false;
    bool  graph_direction_cache = false;
    bool  graph_direction_update = false;
    bool  graph_tail_cache = false;
    bool  graph_conditional_while = false;
    bool  graph_conditional_cache = false;
    bool  verify_true_residual = false;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = cudatool::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver();
    struct GraphDirectionStats
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
    GraphDirectionStats graph_direction_stats() const { return m_graph_direction_stats; }
    GraphDirectionStats graph_tail_stats() const { return m_graph_tail_stats; }
    GraphDirectionStats graph_conditional_stats() const { return m_graph_conditional_stats; }
    struct SolveReport
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
    SolveReport solve_report() const { return m_last_solve_report; }
    void set_global_tol_rate(Float rate) { m_config.global_tol_rate = rate; }
    Float baseline_tol_rate() const { return m_baseline_tol_rate; }

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    DeviceDenseVector reduction_scratch;  // ping-pong dot workspace
    DeviceDenseVector scalar_workspace;  // rho, rho_new, dot, alpha, beta on GPU
    PCGSolverConfig   m_config;
    Float m_baseline_tol_rate = 0;
    Float m_initial_rho = 0;
    SolveReport m_last_solve_report;
    const char* m_last_iteration_exit = "unavailable";
    GraphDirectionStats m_graph_direction_stats;
    GraphDirectionStats m_graph_tail_stats;
    GraphDirectionStats m_graph_conditional_stats;
    struct DirectionGraphKey
    {
        GraphMatrixSignature matrix;
        const void* x = nullptr;
        const void* r = nullptr;
        const void* z = nullptr;
        const void* p = nullptr;
        const void* Ap = nullptr;
        const void* scratch = nullptr;
        const void* scalar = nullptr;
        SizeT dof = 0;
        int device = -1;
        bool operator==(const DirectionGraphKey& other) const
        {
            return matrix.values == other.matrix.values
                   && matrix.rows == other.matrix.rows
                   && matrix.cols == other.matrix.cols
                   && matrix.preconditioner == other.matrix.preconditioner
                   && matrix.triplet_count == other.matrix.triplet_count
                   && matrix.local_preconditioner_count == other.matrix.local_preconditioner_count
                   && matrix.local_preconditioner_shape == other.matrix.local_preconditioner_shape
                   && x == other.x && r == other.r && z == other.z
                   && p == other.p && Ap == other.Ap
                   && scratch == other.scratch && scalar == other.scalar
                   && dof == other.dof && device == other.device;
        }
    };
    DirectionGraphKey m_direction_graph_key;
    SizeT m_conditional_graph_max_iter = 0;
    Float m_conditional_graph_rate = 0;
    cudaGraph_t m_direction_graph = nullptr;
    cudaGraphExec_t m_direction_exec = nullptr;
    size_t m_direction_graph_node_count = 0;
    bool m_graph_topology_probe_pending = false;
    int m_graph_audit_frame = 0;
    bool m_graph_audit_done = false;
    struct TailGraphKey
    {
        const void* r = nullptr;
        const void* z = nullptr;
        const void* p = nullptr;
        const void* Ap = nullptr;
        const void* scratch = nullptr;
        const void* scalar = nullptr;
        SizeT dof = 0;
        int device = -1;
        bool operator==(const TailGraphKey& other) const
        {
            return r == other.r && z == other.z && p == other.p
                   && Ap == other.Ap && scratch == other.scratch
                   && scalar == other.scalar && dof == other.dof
                   && device == other.device;
        }
    };
    TailGraphKey m_tail_graph_key;
    cudaGraph_t m_tail_graph = nullptr;
    cudaGraphExec_t m_tail_exec = nullptr;
    // Verified PCG must inspect rho after the dot product and before beta/p.
    // Cache the latter as a second graph so the host check keeps its order.
    cudaGraph_t m_verify_update_graph = nullptr;
    cudaGraphExec_t m_verify_update_exec = nullptr;
    void release_direction_graph();
    void release_tail_graph();

  protected:
    SizeT solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter);
    SizeT pcg_device_scalar(cudatool::DenseVectorView<Float> x,
                            cudatool::CDenseVectorView<Float> b, SizeT max_iter);
    SizeT pcg_conditional_while(cudatool::DenseVectorView<Float> x,
                                cudatool::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
