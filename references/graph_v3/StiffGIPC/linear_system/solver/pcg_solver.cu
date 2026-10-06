#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <utility>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern int total_Frames;
extern std::string benchmark_output_dir;
extern bool experimental_pcg_legacy_stop;
extern bool experimental_pcg_fused_dot_tail;
extern bool experimental_pcg_fused_continue;



// All 256 lanes participate, including tail lanes carrying zero. The previous
// early return before __syncthreads and full-mask shuffle used inactive lanes.
__global__ void PCG_vdv_Reduction(double* out, const double* a, const double* b, int numbers)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ double warp_sums[8];
    double value = idx < numbers ? a[idx] * b[idx] : 0.0;
    for(int offset = 1; offset < 32; offset <<= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if(lane == 0)
        warp_sums[warp] = value;
    __syncthreads();
    if(warp == 0)
    {
        value = lane < 8 ? warp_sums[lane] : 0.0;
        for(int offset = 1; offset < 32; offset <<= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if(lane == 0)
            out[blockIdx.x] = value;
    }
}

// Input and output never overlap: in-place compaction races across blocks when
// a reduction level contains more than one block (e.g. 70,266 PCG DOFs).
__global__ void add_reduction(double* out, const double* in, int numbers)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ double warp_sums[8];
    double value = idx < numbers ? in[idx] : 0.0;
    for(int offset = 1; offset < 32; offset <<= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if(lane == 0)
        warp_sums[warp] = value;
    __syncthreads();
    if(warp == 0)
    {
        value = lane < 8 ? warp_sums[lane] : 0.0;
        for(int offset = 1; offset < 32; offset <<= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if(lane == 0)
            out[blockIdx.x] = value;
    }
}


__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void pcg_true_residual_kernel(int count, double* residual,
                                         const double* rhs, const double* Ax)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) residual[i] = rhs[i] - Ax[i];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(
    double* temp, double* scratch, const double* A, const double* B, int vertexNum)
{
    if(vertexNum < 1)
        return 0.0;
    constexpr int threads = 256;
    int count = (vertexNum + threads - 1) / threads;
    PCG_vdv_Reduction<<<count, threads>>>(temp, A, B, vertexNum);
    double* in = temp;
    double* out = scratch;
    while(count > 1)
    {
        const int blocks = (count + threads - 1) / threads;
        add_reduction<<<blocks, threads>>>(out, in, count);
        count = blocks;
        std::swap(in, out);
    }
    double result;
    CUDA_SAFE_CALL(cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

// A1 ablation: keep all PCG arithmetic scalars on the device. These kernels
// use the same reduction tree and the same legacy convergence check position.
__global__ void pcg_device_alpha(double* scalar, bool verify)
{
    if(verify)
    {
        // An invalid direction must not modify x/r. The host reads this flag
        // together with the already required current-rho readback.
        scalar[5] = !isfinite(scalar[1]) ? 2.0
                    : scalar[1] <= 0.0 ? 1.0
                    : !isfinite(scalar[0]) || scalar[0] < 0.0 ? 3.0 : 0.0;
        if(scalar[5] != 0.0)
        {
            scalar[3] = 0.0;
            return;
        }
    }
    scalar[3] = __ddiv_rn(scalar[0], scalar[1]);
}

__global__ void pcg_device_beta(double* scalar)
{
    scalar[4] = __ddiv_rn(scalar[2], scalar[0]);
    scalar[0] = scalar[2];
}

__global__ void pcg_device_update_dx_r(double* dx, double* r,
                                        const double* p, const double* Ap,
                                        const double* alpha, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    dx[i] += alpha[0] * p[i];
    r[i] -= alpha[0] * Ap[i];
}

__global__ void pcg_device_update_p(double* p, const double* z,
                                     const double* beta, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    p[i] = z[i] + beta[0] * p[i];
}

__global__ void pcg_device_update_p_continue(
    double* p, const double* z, double* scalar,
    cudaGraphConditionalHandle handle, int max_iter, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count)
        p[i] = z[i] + scalar[4] * p[i];
    // The graph observes the handle only after this entire kernel completes.
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int completed = static_cast<int>(scalar[6]) + 1;
        scalar[6] = static_cast<double>(completed);
        cudaGraphSetConditional(handle,
                                scalar[8] == 0.0 && completed < max_iter - 1);
    }
}

// The conditional WHILE graph keeps the legacy check of the previous rho
// immediately after updating x/r. The final preconditioner/tail may still
// execute, but it cannot change x and is discarded when the solve returns.
__global__ void pcg_conditional_init(double* scalar)
{
    scalar[6] = 0.0;  // executed direction iterations
    scalar[7] = scalar[0];  // initial rho
    scalar[8] = 0.0;  // old-rho stop flag
}

__global__ void pcg_conditional_check_old_rho(double* scalar, double rate)
{
    scalar[8] = fabs(scalar[0]) <= rate * scalar[7] ? 1.0 : 0.0;
}

__global__ void pcg_conditional_continue(double* scalar,
                                         cudaGraphConditionalHandle handle,
                                         int max_iter)
{
    const int completed = static_cast<int>(scalar[6]) + 1;
    scalar[6] = static_cast<double>(completed);
    cudaGraphSetConditional(handle,
                            scalar[8] == 0.0 && completed < max_iter - 1);
}

// An optional, one-shot graph topology probe. It changes no simulation data.
__global__ void pcg_graph_topology_probe_node() {}

static void pcg_dot_device(double* out, double* temp, double* scratch,
                           const double* a, const double* b, int count,
                           bool fused_tail = false)
{
    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;
    PCG_vdv_Reduction<<<blocks, threads>>>(
        fused_tail && blocks == 1 ? out : temp, a, b, count);
    if(fused_tail && blocks == 1)
        return;
    double* in = temp;
    double* next = scratch;
    while(blocks > 1)
    {
        const int reduced = (blocks + threads - 1) / threads;
        double* target = fused_tail && reduced == 1 ? out : next;
        add_reduction<<<reduced, threads>>>(target, in, blocks);
        blocks = reduced;
        if(target == out)
            return;
        in = target;
        next = target == scratch ? temp : scratch;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(out, in, sizeof(double),
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
    , m_baseline_tol_rate(cfg.global_tol_rate)
{
    const char* probe = std::getenv("GIPC_PCG_GRAPH_TOPOLOGY_PROBE");
    m_graph_topology_probe_pending = cfg.graph_direction_update && probe
                                     && probe[0] == '1' && probe[1] == '\0';
    const char* audit_frame = std::getenv("GIPC_PCG_GRAPH_AUDIT_FRAME");
    if(audit_frame && cfg.device_scalar_nograph)
    {
        const long frame = std::strtol(audit_frame, nullptr, 10);
        if(frame > 0 && frame <= 1000000) m_graph_audit_frame = static_cast<int>(frame);
    }
}
PCGSolver::~PCGSolver()
{
    release_direction_graph();
    release_tail_graph();
}
void PCGSolver::release_direction_graph()
{
    if(m_direction_exec)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaGraphExecDestroy(m_direction_exec));
        m_direction_exec = nullptr;
    }
    if(m_direction_graph)
    {
        CUDA_SAFE_CALL(cudaGraphDestroy(m_direction_graph));
        m_direction_graph = nullptr;
    }
    m_direction_graph_key = {};
    m_direction_graph_node_count = 0;
    m_conditional_graph_max_iter = 0;
    m_conditional_graph_rate = 0;
}
void PCGSolver::release_tail_graph()
{
    if(m_tail_exec || m_verify_update_exec)
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
    if(m_tail_exec)
    {
        CUDA_SAFE_CALL(cudaGraphExecDestroy(m_tail_exec));
        m_tail_exec = nullptr;
    }
    if(m_verify_update_exec)
    {
        CUDA_SAFE_CALL(cudaGraphExecDestroy(m_verify_update_exec));
        m_verify_update_exec = nullptr;
    }
    if(m_tail_graph)
    {
        CUDA_SAFE_CALL(cudaGraphDestroy(m_tail_graph));
        m_tail_graph = nullptr;
    }
    if(m_verify_update_graph)
    {
        CUDA_SAFE_CALL(cudaGraphDestroy(m_verify_update_graph));
        m_verify_update_graph = nullptr;
    }
    m_tail_graph_key = {};
}
SizeT PCGSolver::solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    reduction_scratch.resize((b.size() + 255) / 256);
    if(!m_config.graph_direction_cache && !m_config.graph_conditional_cache
       && m_direction_exec) release_direction_graph();
    if(!m_config.graph_tail_cache && (m_tail_exec || m_verify_update_exec))
        release_tail_graph();
    if(m_config.device_scalar_nograph)
        scalar_workspace.resize(m_config.graph_conditional_while ? 9 : 6);
    const Float requested_rate = m_config.global_tol_rate;
    m_last_solve_report = {};
    m_last_solve_report.requested_rate = requested_rate;
    auto iter = m_config.graph_conditional_while
        ? pcg_conditional_while(x, b, m_config.max_iter_ratio * b.size())
        : m_config.device_scalar_nograph
            ? pcg_device_scalar(x, b, m_config.max_iter_ratio * b.size())
            : pcg(x, b, m_config.max_iter_ratio * b.size());
    m_last_solve_report.initial_iterations = iter;
    m_last_solve_report.initial_exit = m_last_iteration_exit;
    m_last_solve_report.initial_rho = m_initial_rho;
    auto true_residual_rho = [&]()
    {
        spmv(cudatool::CDenseVectorView<Float>(x), Ap.view());
        const int n = static_cast<int>(b.size());
        pcg_true_residual_kernel<<<(n + 255) / 256, 256>>>(
            n, r.buffer_view().data(), b.buffer_view().data(),
            Ap.buffer_view().data());
        apply_preconditioner(z, r);
        return My_PCG_General_v_v_Reduction_Algorithm(
            p.buffer_view().data(), reduction_scratch.buffer_view().data(),
            r.buffer_view().data(), z.buffer_view().data(), n);
    };
    if(m_config.verify_true_residual)
    {
        Float true_rho = true_residual_rho();
        // The legacy PCG checks the previous rho after updating x. Its true
        // residual can exceed the requested rate even for the stock solver.
        const Float verification_factor = experimental_pcg_legacy_stop ? 2.0 : 1.0;
        const Float tolerance = verification_factor * requested_rate
                                * m_last_solve_report.initial_rho;
        if((!std::isfinite(true_rho) || true_rho < 0
            || true_rho > tolerance * (1.0 + 1e-6))
           && requested_rate > m_baseline_tol_rate)
        {
            m_last_solve_report.baseline_fallback = true;
            m_config.global_tol_rate = m_baseline_tol_rate;
            x.buffer_view().fill(0);
            m_last_solve_report.fallback_iterations =
                m_config.device_scalar_nograph && !experimental_pcg_legacy_stop
                    ? pcg_device_scalar(x, b, m_config.max_iter_ratio * b.size())
                    : pcg(x, b, m_config.max_iter_ratio * b.size());
            m_last_solve_report.fallback_exit = m_last_iteration_exit;
            m_config.global_tol_rate = requested_rate;
            true_rho = true_residual_rho();
            iter += m_last_solve_report.fallback_iterations;
        }
        m_last_solve_report.true_rho = true_rho;
        const Float effective_rate = m_last_solve_report.baseline_fallback
                                         ? m_baseline_tol_rate : requested_rate;
        m_last_solve_report.verified =
            std::isfinite(true_rho) && true_rho >= 0
            && true_rho <= verification_factor * effective_rate
                                   * m_last_solve_report.initial_rho
                                   * (1.0 + 1e-6);
        if(!m_last_solve_report.verified)
            throw std::runtime_error("Adaptive PCG true residual failed verification");
    }
    else
    {
        const char* audit = std::getenv("GIPC_PCG_LEGACY_TRUE_RESIDUAL_AUDIT");
        if(audit && audit[0] == '1' && audit[1] == '\0'
           && !benchmark_output_dir.empty())
        {
            static uint64_t audit_count = 0;
            const Float true_rho = true_residual_rho();
            const auto path = benchmark_output_dir + "/pcg_legacy_true_residual.csv";
            std::ofstream csv(path, audit_count == 0
                                       ? std::ios::trunc : std::ios::app);
            if(audit_count++ == 0)
                csv << "frame,solve_index,requested_rate,initial_rho,true_rho,"
                       "iterations,exit\n";
            csv << std::setprecision(17) << total_Frames << ','
                << audit_count << ',' << requested_rate
                << ',' << m_last_solve_report.initial_rho << ',' << true_rho
                << ',' << iter << ',' << m_last_solve_report.initial_exit
                << '\n';
            if(!csv) throw std::runtime_error("Cannot write legacy PCG residual audit");
        }
    }
    if(m_graph_audit_frame == total_Frames + 1 && !m_graph_audit_done
       && !benchmark_output_dir.empty() && !m_config.verify_true_residual)
    {
        // Re-run eager PCG on the same assembled system after preserving the
        // actual direction. Audit runs are excluded from performance results.
        m_graph_audit_done = true;
        const SizeT n = b.size();
        const auto bytes = static_cast<size_t>(n) * sizeof(Float);
        std::vector<Float> actual(n), reference(n), rhs(n), Ax(n);
        CUDA_SAFE_CALL(cudaMemcpy(actual.data(), x.buffer_view().data(), bytes,
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(rhs.data(), b.buffer_view().data(), bytes,
                                  cudaMemcpyDeviceToHost));
        auto residual_rate = [&](cudatool::CDenseVectorView<Float> direction)
        {
            spmv(direction, Ap.view());
            CUDA_SAFE_CALL(cudaMemcpy(Ax.data(), Ap.buffer_view().data(), bytes,
                                      cudaMemcpyDeviceToHost));
            double numerator = 0, denominator = 0;
            for(SizeT i = 0; i < n; ++i)
            {
                const double delta = rhs[i] - Ax[i];
                numerator += delta * delta;
                denominator += rhs[i] * rhs[i];
            }
            return std::sqrt(numerator / denominator);
        };
        const double actual_residual = residual_rate(
            cudatool::CDenseVectorView<Float>(x));
        DeviceDenseVector eager_x;
        eager_x.resize(n);
        eager_x.buffer_view().fill(0);
        const auto saved_exit = m_last_iteration_exit;
        const auto saved_rho = m_initial_rho;
        const auto eager_iterations = pcg(eager_x.view(), b,
                                          m_config.max_iter_ratio * n);
        const auto eager_exit = m_last_iteration_exit;
        const auto eager_rho = m_initial_rho;
        CUDA_SAFE_CALL(cudaMemcpy(reference.data(), eager_x.buffer_view().data(),
                                  bytes, cudaMemcpyDeviceToHost));
        const double eager_residual = residual_rate(eager_x.cview());
        m_last_iteration_exit = saved_exit;
        m_initial_rho = saved_rho;
        double squared_delta = 0, squared_reference = 0, max_delta = 0;
        for(SizeT i = 0; i < n; ++i)
        {
            const double delta = actual[i] - reference[i];
            squared_delta += delta * delta;
            squared_reference += reference[i] * reference[i];
            max_delta = std::max(max_delta, std::abs(delta));
        }
        const auto prefix = benchmark_output_dir + "/pcg_same_system_audit";
        auto write_vector = [&](const std::string& name, const std::vector<Float>& value)
        {
            std::ofstream file(prefix + name, std::ios::binary);
            file.write(reinterpret_cast<const char*>(value.data()), bytes);
            if(!file) throw std::runtime_error("Cannot write PCG same-system audit vector");
        };
        write_vector("_actual.f64", actual);
        write_vector("_eager.f64", reference);
        std::ofstream csv(prefix + ".csv");
        csv << "frame,dof,actual_iterations,eager_iterations,actual_exit,eager_exit,actual_initial_rho,eager_initial_rho,actual_relative_true_residual,eager_relative_true_residual,direction_rms_difference,direction_relative_l2_difference,direction_max_abs_difference\n";
        csv << std::setprecision(17) << (total_Frames + 1) << ',' << n << ','
            << iter << ',' << eager_iterations << ','
            << m_last_solve_report.initial_exit << ',' << eager_exit << ','
            << m_last_solve_report.initial_rho << ',' << eager_rho << ','
            << actual_residual << ',' << eager_residual << ','
            << std::sqrt(squared_delta / n) << ','
            << std::sqrt(squared_delta / squared_reference) << ','
            << max_delta << '\n';
        if(!csv) throw std::runtime_error("Cannot write PCG same-system audit");
    }
    return iter;
}


SizeT PCGSolver::pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;
    m_last_iteration_exit = "iteration_limit";

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    reduction_scratch.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
    }

    p   = z;
    rz0 = rz;
    m_initial_rho = rz0;
    if(m_config.verify_true_residual && rz0 == 0)
    {
        m_last_iteration_exit = "zero_rhs";
        return 0;
    }
    if(m_config.verify_true_residual && (!std::isfinite(rz0) || rz0 < 0))
    {
        m_last_iteration_exit = "invalid_initial_rho";
        return 0;
    }

    for(k = 1; k < max_iter; ++k)
    {
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                    reduction_scratch.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       z.size());

            if(m_config.verify_true_residual
               && (!std::isfinite(dot_res) || dot_res <= 0))
            {
                m_last_iteration_exit = std::isfinite(dot_res)
                                            ? "non_positive_curvature" : "nonfinite_curvature";
                return k - 1;
            }
            alpha = rz / dot_res;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        if((!m_config.verify_true_residual || experimental_pcg_legacy_stop)
           && std::abs(rz) <= m_config.global_tol_rate * rz0)
        {
            m_last_iteration_exit = "recursive_residual";
            break;
        }

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                    reduction_scratch.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
        }

        // The legacy check uses the previous rho after updating x.
        // Verified runs check the current rho before accepting the direction.
        if(m_config.verify_true_residual
           && (!std::isfinite(rz_new) || rz_new < 0))
        {
            m_last_iteration_exit = "invalid_recursive_rho";
            return k;
        }
        if(m_config.verify_true_residual && !experimental_pcg_legacy_stop
           && std::abs(rz_new) <= m_config.global_tol_rate * rz0)
        {
            rz = rz_new;
            m_last_iteration_exit = "recursive_residual";
            break;
        }

        beta = rz_new / rz;

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;
    }

    return k;
}

SizeT PCGSolver::pcg_device_scalar(cudatool::DenseVectorView<Float> x,
                                      cudatool::CDenseVectorView<Float> b,
                                      SizeT max_iter)
{
    m_last_iteration_exit = "iteration_limit";
    r.buffer_view().copy_from(b.buffer_view());
    apply_preconditioner(z, r);
    double* const scalar = scalar_workspace.buffer_view().data();
    const int count = static_cast<int>(z.size());
    pcg_dot_device(scalar, p.buffer_view().data(),
                   reduction_scratch.buffer_view().data(),
                   r.buffer_view().data(), z.buffer_view().data(), count);
    Float rho = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&rho, scalar, sizeof(Float), cudaMemcpyDeviceToHost));
    const Float rho_initial = rho;
    m_initial_rho = rho_initial;
    if(m_config.verify_true_residual && rho_initial == 0)
    {
        m_last_iteration_exit = "zero_rhs";
        return 0;
    }
    if(m_config.verify_true_residual && (!std::isfinite(rho_initial) || rho_initial < 0))
    {
        m_last_iteration_exit = "invalid_initial_rho";
        return 0;
    }
    p = z;
    SizeT k = 0;
    const int blocks = (count + 255) / 256;
    cudaGraph_t direction_graph = nullptr;
    cudaGraphExec_t direction_exec = nullptr;
    cudaGraph_t tail_graph = nullptr;
    cudaGraphExec_t tail_exec = nullptr;
    cudaGraph_t verify_update_graph = nullptr;
    cudaGraphExec_t verify_update_exec = nullptr;
    bool capture_failed = false;
    bool tail_capture_failed = false;
    bool verify_update_capture_failed = false;
    bool needs_update = false;
    DirectionGraphKey key;
    TailGraphKey tail_key;
    if(m_config.graph_direction_cache || m_config.graph_tail_cache)
    {
        key.matrix = graph_matrix_signature();
        key.x = x.buffer_view().data();
        key.r = r.buffer_view().data();
        key.z = z.buffer_view().data();
        key.p = p.buffer_view().data();
        key.Ap = Ap.buffer_view().data();
        key.scratch = reduction_scratch.buffer_view().data();
        key.scalar = scalar;
        key.dof = static_cast<SizeT>(count);
        CUDA_SAFE_CALL(cudaGetDevice(&key.device));
        tail_key = {key.r, key.z, key.p, key.Ap, key.scratch,
                    key.scalar, key.dof, key.device};
        if(m_config.graph_direction_cache && m_direction_exec
           && !(key == m_direction_graph_key))
        {
            ++m_graph_direction_stats.invalidations;
            if(m_config.graph_direction_update)
                needs_update = true;
            else
                release_direction_graph();
        }
        if(m_config.graph_direction_cache && m_direction_exec && !needs_update)
        {
            direction_exec = m_direction_exec;
            ++m_graph_direction_stats.cache_hits;
        }
        if(m_config.graph_tail_cache)
        {
            if((m_tail_exec || m_verify_update_exec)
               && !(tail_key == m_tail_graph_key))
            {
                release_tail_graph();
                ++m_graph_tail_stats.invalidations;
            }
            if(m_tail_exec)
            {
                tail_exec = m_tail_exec;
                ++m_graph_tail_stats.cache_hits;
            }
            if(m_config.verify_true_residual && m_verify_update_exec)
            {
                verify_update_exec = m_verify_update_exec;
                ++m_graph_tail_stats.cache_hits;
            }
        }
    }
    auto ordinary_direction = [&]()
    {
        spmv(p.cview(), Ap.view());
        pcg_dot_device(scalar + 1, z.buffer_view().data(),
                       reduction_scratch.buffer_view().data(),
                       p.buffer_view().data(), Ap.buffer_view().data(), count);
        pcg_device_alpha<<<1, 1>>>(scalar, m_config.verify_true_residual);
        pcg_device_update_dx_r<<<blocks, 256>>>(
            x.buffer_view().data(), r.buffer_view().data(),
            p.buffer_view().data(), Ap.buffer_view().data(), scalar + 3, count);
    };
    auto ordinary_tail = [&]()
    {
        pcg_dot_device(scalar + 2, Ap.buffer_view().data(),
                       reduction_scratch.buffer_view().data(),
                       r.buffer_view().data(), z.buffer_view().data(), count);
        if(!m_config.verify_true_residual)
        {
            pcg_device_beta<<<1, 1>>>(scalar);
            pcg_device_update_p<<<blocks, 256>>>(p.buffer_view().data(),
                                                 z.buffer_view().data(), scalar + 4, count);
        }
    };
    auto ordinary_verify_update = [&]()
    {
        pcg_device_beta<<<1, 1>>>(scalar);
        pcg_device_update_p<<<blocks, 256>>>(p.buffer_view().data(),
                                             z.buffer_view().data(), scalar + 4, count);
    };
    for(k = 1; k < max_iter; ++k)
    {
        if((m_config.graph_direction_segment || m_config.graph_direction_cache)
           && !direction_exec && !capture_failed)
        {
            // Recapture on a changed key; update the executable only when
            // CUDA accepts the new definition and its topology is compatible.
            const auto capture_start = std::chrono::steady_clock::now();
            auto status = cudaStreamBeginCapture(cudaStreamPerThread,
                                                 cudaStreamCaptureModeThreadLocal);
            if(status == cudaSuccess)
            {
                ordinary_direction();
                if(needs_update && m_graph_topology_probe_pending)
                {
                    pcg_graph_topology_probe_node<<<1, 1>>>();
                    m_graph_topology_probe_pending = false;
                }
                status = cudaStreamEndCapture(cudaStreamPerThread, &direction_graph);
                if(status == cudaSuccess && direction_graph && needs_update)
                {
                    size_t node_count = 0;
                    status = cudaGraphGetNodes(direction_graph, nullptr, &node_count);
                    if(status == cudaSuccess)
                    {
                        // The old graph may still be queued on this stream.
                        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
                        if(node_count == m_direction_graph_node_count)
                        {
                            cudaGraphExecUpdateResultInfo info{};
                            const auto update_start = std::chrono::steady_clock::now();
                            const auto update_status = cudaGraphExecUpdate(
                                m_direction_exec, direction_graph, &info);
                            m_graph_direction_stats.update_wall_ms +=
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - update_start).count();
                            if(update_status == cudaSuccess
                               && info.result == cudaGraphExecUpdateSuccess)
                            {
                                ++m_graph_direction_stats.updates;
                                CUDA_SAFE_CALL(cudaGraphDestroy(m_direction_graph));
                                m_direction_graph = direction_graph;
                                m_direction_graph_key = key;
                                direction_exec = m_direction_exec;
                                direction_graph = nullptr;
                            }
                            else if(update_status == cudaErrorGraphExecUpdateFailure)
                            {
                                ++m_graph_direction_stats.update_rejects;
                                cudaGetLastError();
                            }
                            else
                            {
                                CUDA_SAFE_CALL(update_status == cudaSuccess
                                                   ? cudaErrorUnknown : update_status);
                            }
                        }
                        if(!direction_exec)
                        {
                            release_direction_graph();
                            ++m_graph_direction_stats.rebuilds;
                        }
                    }
                }
                if(status == cudaSuccess && direction_graph)
                    status = cudaGraphInstantiate(&direction_exec, direction_graph,
                                                   nullptr, nullptr, 0);
            }
            if(status != cudaSuccess || !direction_exec)
            {
                cudaGetLastError();
                if(direction_exec) cudaGraphExecDestroy(direction_exec);
                if(direction_graph) cudaGraphDestroy(direction_graph);
                if(needs_update) release_direction_graph();
                direction_exec = nullptr;
                direction_graph = nullptr;
                capture_failed = true;
                ++m_graph_direction_stats.fallbacks;
            }
            else
            {
                ++m_graph_direction_stats.captures;
                m_graph_direction_stats.instantiate_wall_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - capture_start).count();
                if(m_config.graph_direction_cache && direction_graph)
                {
                    m_direction_graph = direction_graph;
                    m_direction_exec = direction_exec;
                    m_direction_graph_key = key;
                    CUDA_SAFE_CALL(cudaGraphGetNodes(direction_graph, nullptr,
                                                     &m_direction_graph_node_count));
                }
            }
        }
        if(direction_exec)
        {
            // A launch error may have partially updated x/r; fail the solve
            // instead of replaying the ordinary path on an unknown state.
            CUDA_SAFE_CALL(cudaGraphLaunch(direction_exec, cudaStreamPerThread));
            ++m_graph_direction_stats.launches;
        }
        else
            ordinary_direction();
        if(!m_config.verify_true_residual
           && std::abs(rho) <= m_config.global_tol_rate * rho_initial)
        {
            m_last_iteration_exit = "recursive_residual";
            break;
        }
        apply_preconditioner(z, r);
        if(m_config.graph_tail_cache)
        {
            if(!tail_exec && !tail_capture_failed)
            {
                const auto capture_start = std::chrono::steady_clock::now();
                auto status = cudaStreamBeginCapture(
                    cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                if(status == cudaSuccess)
                {
                    ordinary_tail();
                    status = cudaStreamEndCapture(cudaStreamPerThread, &tail_graph);
                    if(status == cudaSuccess && tail_graph)
                        status = cudaGraphInstantiate(&tail_exec, tail_graph,
                                                       nullptr, nullptr, 0);
                }
                if(status != cudaSuccess || !tail_exec)
                {
                    cudaGetLastError();
                    if(tail_exec) cudaGraphExecDestroy(tail_exec);
                    if(tail_graph) cudaGraphDestroy(tail_graph);
                    tail_exec = nullptr;
                    tail_graph = nullptr;
                    tail_capture_failed = true;
                    ++m_graph_tail_stats.fallbacks;
                }
                else
                {
                    ++m_graph_tail_stats.captures;
                    m_graph_tail_stats.instantiate_wall_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - capture_start).count();
                    m_tail_graph = tail_graph;
                    m_tail_exec = tail_exec;
                    m_tail_graph_key = tail_key;
                }
            }
            if(tail_exec)
            {
                CUDA_SAFE_CALL(cudaGraphLaunch(tail_exec, cudaStreamPerThread));
                ++m_graph_tail_stats.launches;
            }
            else
                ordinary_tail();
        }
        else
            pcg_dot_device(scalar + 2, Ap.buffer_view().data(),
                           reduction_scratch.buffer_view().data(),
                           r.buffer_view().data(), z.buffer_view().data(), count);
        if(m_config.verify_true_residual)
        {
            Float current_scalars[6] = {};
            CUDA_SAFE_CALL(cudaMemcpy(current_scalars, scalar,
                                      sizeof(current_scalars), cudaMemcpyDeviceToHost));
            const Float current_rho = current_scalars[2];
            if(current_scalars[5] != 0.0)
            {
                m_last_iteration_exit = current_scalars[5] == 1.0
                                            ? "non_positive_curvature"
                                            : current_scalars[5] == 2.0
                                                  ? "nonfinite_curvature"
                                                  : "invalid_recursive_rho";
                break;
            }
            if(!std::isfinite(current_rho) || current_rho < 0)
            {
                m_last_iteration_exit = "invalid_recursive_rho";
                break;
            }
            if(std::abs(current_rho)
               <= m_config.global_tol_rate * rho_initial)
            {
                rho = current_rho;
                m_last_iteration_exit = "recursive_residual";
                break;
            }
        }
        if(m_config.verify_true_residual && m_config.graph_tail_cache)
        {
            if(!verify_update_exec && !verify_update_capture_failed)
            {
                const auto capture_start = std::chrono::steady_clock::now();
                auto status = cudaStreamBeginCapture(
                    cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                if(status == cudaSuccess)
                {
                    ordinary_verify_update();
                    status = cudaStreamEndCapture(cudaStreamPerThread,
                                                  &verify_update_graph);
                    if(status == cudaSuccess && verify_update_graph)
                        status = cudaGraphInstantiate(&verify_update_exec,
                                                       verify_update_graph,
                                                       nullptr, nullptr, 0);
                }
                if(status != cudaSuccess || !verify_update_exec)
                {
                    cudaGetLastError();
                    if(verify_update_exec) cudaGraphExecDestroy(verify_update_exec);
                    if(verify_update_graph) cudaGraphDestroy(verify_update_graph);
                    verify_update_exec = nullptr;
                    verify_update_graph = nullptr;
                    verify_update_capture_failed = true;
                    ++m_graph_tail_stats.fallbacks;
                }
                else
                {
                    ++m_graph_tail_stats.captures;
                    m_graph_tail_stats.instantiate_wall_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - capture_start).count();
                    m_verify_update_graph = verify_update_graph;
                    m_verify_update_exec = verify_update_exec;
                    m_tail_graph_key = tail_key;
                }
            }
            if(verify_update_exec)
            {
                CUDA_SAFE_CALL(cudaGraphLaunch(verify_update_exec,
                                               cudaStreamPerThread));
                ++m_graph_tail_stats.launches;
            }
            else
                ordinary_verify_update();
        }
        else if(!m_config.graph_tail_cache || m_config.verify_true_residual)
        {
            ordinary_verify_update();
        }
        CUDA_SAFE_CALL(cudaMemcpy(&rho, scalar, sizeof(Float), cudaMemcpyDeviceToHost));
        if(m_config.verify_true_residual && (!std::isfinite(rho) || rho < 0))
        {
            m_last_iteration_exit = "invalid_recursive_rho";
            break;
        }
    }
    if(direction_exec && !m_config.graph_direction_cache)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaGraphExecDestroy(direction_exec));
    }
    if(direction_graph && !m_config.graph_direction_cache)
        CUDA_SAFE_CALL(cudaGraphDestroy(direction_graph));
    return k;
}

SizeT PCGSolver::pcg_conditional_while(cudatool::DenseVectorView<Float> x,
                                       cudatool::CDenseVectorView<Float> b,
                                       SizeT max_iter)
{
    // The optional cache keeps a graph only while its captured pointers,
    // matrix shape, iteration bound, and convergence rate are unchanged.
    if((m_config.verify_true_residual && !experimental_pcg_legacy_stop)
       || max_iter <= 1
       || max_iter > static_cast<SizeT>(std::numeric_limits<int>::max()))
    {
        ++m_graph_conditional_stats.fallbacks;
        return pcg_device_scalar(x, b, max_iter);
    }
    m_last_iteration_exit = "iteration_limit";
    r.buffer_view().copy_from(b.buffer_view());
    apply_preconditioner(z, r);
    double* const scalar = scalar_workspace.buffer_view().data();
    const int count = static_cast<int>(z.size());
    const int blocks = (count + 255) / 256;
    pcg_dot_device(scalar, p.buffer_view().data(),
                   reduction_scratch.buffer_view().data(),
                   r.buffer_view().data(), z.buffer_view().data(), count,
                   experimental_pcg_fused_dot_tail);
    Float rho_initial = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&rho_initial, scalar, sizeof(Float),
                              cudaMemcpyDeviceToHost));
    m_initial_rho = rho_initial;
    p = z;
    pcg_conditional_init<<<1, 1>>>(scalar);

    DirectionGraphKey key;
    if(m_config.graph_conditional_cache)
    {
        key.matrix = graph_matrix_signature();
        key.x = x.buffer_view().data();
        key.r = r.buffer_view().data();
        key.z = z.buffer_view().data();
        key.p = p.buffer_view().data();
        key.Ap = Ap.buffer_view().data();
        key.scratch = reduction_scratch.buffer_view().data();
        key.scalar = scalar;
        key.dof = static_cast<SizeT>(count);
        CUDA_SAFE_CALL(cudaGetDevice(&key.device));
        if(m_direction_exec
           && (!(key == m_direction_graph_key)
               || m_conditional_graph_max_iter != max_iter
               || m_conditional_graph_rate != m_config.global_tol_rate))
        {
            ++m_graph_conditional_stats.invalidations;
            release_direction_graph();
        }
        if(m_direction_exec)
            ++m_graph_conditional_stats.cache_hits;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = m_config.graph_conditional_cache
        ? m_direction_exec : nullptr;
    cudaGraph_t captured = nullptr;
    cudaGraphConditionalHandle handle{};
    cudaGraphNode_t while_node = nullptr;
    cudaError_t status = cudaSuccess;
    if(!graph_exec) status = cudaGraphCreate(&graph, 0);
    if(status == cudaSuccess && !graph_exec)
        status = cudaGraphConditionalHandleCreate(
            &handle, graph, 1, cudaGraphCondAssignDefault);
    cudaGraphNodeParams params{};
    if(status == cudaSuccess && !graph_exec)
    {
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = handle;
        params.conditional.type = cudaGraphCondTypeWhile;
        params.conditional.size = 1;
#if CUDART_VERSION >= 13000
        status = cudaGraphAddNode(&while_node, graph, nullptr, nullptr, 0, &params);
#else
        status = cudaGraphAddNode(&while_node, graph, nullptr, 0, &params);
#endif
    }
    if(status == cudaSuccess && !graph_exec)
    {
        cudaGraph_t body = params.conditional.phGraph_out[0];
        status = cudaStreamBeginCaptureToGraph(
            cudaStreamPerThread, body, nullptr, nullptr, 0,
            cudaStreamCaptureModeThreadLocal);
        if(status == cudaSuccess)
        {
            const char* failed_stage = nullptr;
            auto capture_ok = [&](const char* stage)
            {
                cudaStreamCaptureStatus capture_state;
                const auto query = cudaStreamIsCapturing(
                    cudaStreamPerThread, &capture_state);
                if(query != cudaSuccess || capture_state != cudaStreamCaptureStatusActive)
                {
                    status = query == cudaSuccess
                                 ? cudaErrorStreamCaptureInvalidated : query;
                    failed_stage = stage;
                }
            };
            spmv(p.cview(), Ap.view());
            capture_ok("spmv");
            if(status == cudaSuccess)
            {
                pcg_dot_device(scalar + 1, z.buffer_view().data(),
                               reduction_scratch.buffer_view().data(),
                               p.buffer_view().data(), Ap.buffer_view().data(),
                               count, experimental_pcg_fused_dot_tail);
                capture_ok("direction_dot");
            }
            if(status == cudaSuccess)
            {
                pcg_device_alpha<<<1, 1>>>(scalar, false);
                pcg_device_update_dx_r<<<blocks, 256>>>(
                    x.buffer_view().data(), r.buffer_view().data(),
                    p.buffer_view().data(), Ap.buffer_view().data(), scalar + 3,
                    count);
                pcg_conditional_check_old_rho<<<1, 1>>>(
                    scalar, m_config.global_tol_rate);
                capture_ok("direction_update");
            }
            if(status == cudaSuccess)
            {
                apply_preconditioner(z, r);
                capture_ok("preconditioner");
            }
            if(status == cudaSuccess)
            {
                pcg_dot_device(scalar + 2, Ap.buffer_view().data(),
                               reduction_scratch.buffer_view().data(),
                               r.buffer_view().data(), z.buffer_view().data(),
                               count, experimental_pcg_fused_dot_tail);
                pcg_device_beta<<<1, 1>>>(scalar);
                if(experimental_pcg_fused_continue)
                    pcg_device_update_p_continue<<<blocks, 256>>>(
                        p.buffer_view().data(), z.buffer_view().data(),
                        scalar, handle, static_cast<int>(max_iter), count);
                else
                {
                    pcg_device_update_p<<<blocks, 256>>>(
                        p.buffer_view().data(), z.buffer_view().data(),
                        scalar + 4, count);
                    pcg_conditional_continue<<<1, 1>>>(
                        scalar, handle, static_cast<int>(max_iter));
                }
                capture_ok("tail");
            }
            const auto end_status = cudaStreamEndCapture(
                cudaStreamPerThread, &captured);
            if(failed_stage)
                std::cerr << "Conditional PCG capture failed in "
                          << failed_stage << ": " << cudaGetErrorString(status)
                          << '\n';
            if(status == cudaSuccess) status = end_status;
        }
    }
    const auto instantiate_start = std::chrono::steady_clock::now();
    if(status == cudaSuccess && !graph_exec)
        status = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if(status != cudaSuccess || !graph_exec)
    {
        cudaGetLastError();
        if(graph_exec) cudaGraphExecDestroy(graph_exec);
        ++m_graph_conditional_stats.fallbacks;
        // A stream capture error can invalidate the CUDA stream. Do not run
        // another solve on that state; end this diagnostic process cleanly.
        throw std::runtime_error(std::string("Conditional PCG capture failed: ")
                                 + cudaGetErrorString(status));
    }
    if(graph)
    {
        ++m_graph_conditional_stats.captures;
        m_graph_conditional_stats.instantiate_wall_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - instantiate_start).count();
        if(m_config.graph_conditional_cache)
        {
            m_direction_graph = graph;
            m_direction_exec = graph_exec;
            m_direction_graph_key = key;
            m_conditional_graph_max_iter = max_iter;
            m_conditional_graph_rate = m_config.global_tol_rate;
        }
    }
    CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec, cudaStreamPerThread));
    ++m_graph_conditional_stats.launches;
    Float report[3] = {};
    CUDA_SAFE_CALL(cudaMemcpy(report, scalar + 6, sizeof(report),
                              cudaMemcpyDeviceToHost));
    if(!m_config.graph_conditional_cache)
    {
        CUDA_SAFE_CALL(cudaGraphExecDestroy(graph_exec));
        CUDA_SAFE_CALL(cudaGraphDestroy(graph));
    }
    const SizeT completed = static_cast<SizeT>(report[0]);
    if(completed < 1 || completed >= max_iter)
        throw std::runtime_error("Conditional PCG returned invalid iteration count");
    m_graph_conditional_stats.device_iterations += completed;
    if(report[2] != 0.0)
    {
        m_last_iteration_exit = "recursive_residual";
        return completed;
    }
    return max_iter;
}

}  // namespace gipc
