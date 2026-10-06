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
    static int guard_fixture(const char* output);

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
    bool mas_dot_requested=false,mas_dot_effective=false,mas_dot_supported=false;
    std::string mas_dot_reason;
    SizeT mas_dot_partial_count=0;
    cudatool::DeviceBuffer<Float> mas_dot_partials;
    cudatool::DeviceBuffer<unsigned char> mas_dot_reduce_storage;
    size_t mas_dot_reduce_bytes=0;
    void prepare_mas_dot(SizeT count);
    void apply_mas_dot(cudatool::DenseVectorView<Float> z,
                       cudatool::CDenseVectorView<Float> r,Float* result,bool prepared_only=false);
    Float apply_mas_dot_host(cudatool::DenseVectorView<Float> z,
                             cudatool::CDenseVectorView<Float> r);
    void mas_dot_fixed_study(cudatool::DenseVectorView<Float> x,
                             cudatool::CDenseVectorView<Float> b,const std::string& prefix);
    void release_graph();
    bool spmv_quadratic_requested=false,spmv_quadratic_effective=false,spmv_quadratic_supported=false;
    std::string spmv_quadratic_reason;
    SizeT spmv_quadratic_partials_count=0;
    cudatool::DeviceBuffer<Float> spmv_quadratic_partials;
    cudatool::DeviceBuffer<unsigned char> spmv_quadratic_reduce_storage;
    size_t spmv_quadratic_reduce_bytes=0;
    void prepare_spmv_quadratic(SizeT count);
    void apply_spmv_quadratic(cudatool::CDenseVectorView<Float> input,
        cudatool::DenseVectorView<Float> output,Float* result);
    Float apply_spmv_quadratic_host(cudatool::CDenseVectorView<Float> input,
        cudatool::DenseVectorView<Float> output);
    void spmv_quadratic_fixed_study(cudatool::DenseVectorView<Float> x,
        cudatool::CDenseVectorView<Float> b,const std::string& prefix);
    int diagnostic_fixed_iterations = 0;
    int diagnostic_fused_override = -1;
    void fail_pcg(int code,Float rho,Float curvature);
    bool check_host_rho(Float rho);
    void operator_audit(cudatool::DenseVectorView<Float> x,
                        cudatool::CDenseVectorView<Float> b,const std::string& prefix);
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
