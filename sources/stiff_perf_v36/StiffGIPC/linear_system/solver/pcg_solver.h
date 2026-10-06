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
    bool conditional_graph = false;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = cudatool::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver();

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    cudatool::DeviceBuffer<Float> reduction_result;
    PCGSolverConfig   m_config;
    cudatool::DeviceBuffer<unsigned char> graph_reduce_storage;
    cudatool::DeviceBuffer<Float> graph_scalars;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::vector<std::uintptr_t> captured_key;
    SizeT captured_max_iter = 0;
    Float captured_tol = 0;
    std::uint64_t captures = 0, cache_hits = 0, invalidations = 0;
    void release_graph();
    int diagnostic_fixed_iterations = 0;
    int diagnostic_fused_override = -1;
    void fixed_system_study(cudatool::DenseVectorView<Float> x,
                            cudatool::CDenseVectorView<Float> b, SizeT max_iter, SizeT primary_iter);
    SizeT pcg_graph(cudatool::DenseVectorView<Float> x,
                    cudatool::CDenseVectorView<Float> b, SizeT max_iter);

  protected:
    SizeT solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
