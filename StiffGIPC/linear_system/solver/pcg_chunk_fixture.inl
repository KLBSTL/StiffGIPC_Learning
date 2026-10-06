// Included once after pcg_graph_impl.inl. This fixture calls the production
// guard/vector kernels and a real conditional WHILE graph. Scripted operator
// outputs deliberately isolate control semantics; this is NOT an A/b/M solve.
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int chunk_fixture_n = 513; // Three blocks, including a one-element tail.

__global__ void chunk_fixture_feed(double* s, const double* pap, const double* next,
                                   unsigned* dispatched, int length, int chunk,
                                   bool begin)
{
    const unsigned step = begin ? (*dispatched)++ : *dispatched - 1;
    const double poison = __longlong_as_double(0x7ff8000000000001LL);
    // Never inspect active or reproduce a stopping formula here. Every captured
    // substep receives input, including deliberately poisoned inactive tails.
    s[begin ? (chunk == 4 ? 11 : 1) : (chunk == 4 ? 12 : 2)] =
        step < static_cast<unsigned>(length) ? (begin ? pap[step] : next[step]) : poison;
}

__global__ void chunk_fixture_work(double* ap, double* z, const unsigned* dispatched,
                                   int length, double ap_value, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const bool scripted = *dispatched <= static_cast<unsigned>(length);
    const double poison = __longlong_as_double(0x7ff8000000000001LL);
    ap[i] = scripted ? ap_value : poison;
    z[i] = scripted ? 1.0 : poison;
}

__global__ void chunk_fixture_freeze(const double* x, const double* r, const double* p,
                                     const double* s, double* saved, unsigned* audit, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 3 * n + 10 || s[10] != 0) return;
    const double value = i < n ? x[i] : i < 2 * n ? r[i - n] :
                         i < 3 * n ? p[i - 2 * n] : s[i - 3 * n];
    // audit[0] is immutable throughout this kernel. A separate one-thread node
    // publishes the snapshot only after every block has completed its writes.
    if(audit[0])
    {
        if(__double_as_longlong(value) != __double_as_longlong(saved[i]))
            atomicAdd(audit + 2, 1u);
    }
    else saved[i] = value;
}

__global__ void chunk_fixture_mark(const double* s, unsigned* audit)
{
    if(s[10] == 0)
    {
        if(audit[0]) ++audit[1]; // Number of checked inactive substeps.
        else audit[0] = 1;
    }
}

struct ChunkFixtureGraph
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~ChunkFixtureGraph()
    {
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
    }
};

struct ChunkFixtureCase
{
    std::string name;
    int iterations, fixed = 0, max_iter = 40, error = 0;
    bool converged = false;
    double initial = 1.0, tolerance = 1e-4, ap = 0.0;
    std::vector<double> pap, next;
    ChunkFixtureCase(const std::string& label, int steps)
        : name(label), iterations(steps), pap(steps, 1.0), next(steps, 1.0) {}
};

struct ChunkFixtureResult
{
    std::array<double, 13> s{};
    std::array<unsigned, 3> audit{};
    std::vector<double> state;
    unsigned dispatched = 0;
};

ChunkFixtureResult chunk_fixture_run(const ChunkFixtureCase& c, int chunk)
{
    constexpr int n = chunk_fixture_n, blocks = (n + 255) / 256;
    cudatool::DeviceBuffer<double> s(13), x(n), r(n), p(n), ap(n), z(n);
    cudatool::DeviceBuffer<double> pap(c.pap.size()), next(c.next.size()), saved(3 * n + 10);
    cudatool::DeviceBuffer<unsigned> dispatched(1), audit(3);
    std::array<double, 13> initial{}; // K1 init intentionally leaves s1..s4 alone.
    initial[0] = c.initial;
    std::vector<double> host_x(n, 0.0), host_r(n, 1.0), host_p(n);
    for(int i = 0; i < n; ++i) host_p[i] = 1.0 + double(i) / 1024.0;
    CUDA_SAFE_CALL(cudaMemcpy(s.data(), initial.data(), sizeof(initial), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(x.data(), host_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(r.data(), host_r.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(p.data(), host_p.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pap.data(), c.pap.data(), c.pap.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(next.data(), c.next.data(), c.next.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(dispatched.data(), 0, sizeof(unsigned)));
    CUDA_SAFE_CALL(cudaMemset(audit.data(), 0, 3 * sizeof(unsigned)));
    graph_init<<<1, 1>>>(s.data());
    graph_check_zero_rho<<<blocks, 256>>>(r.data(), n, s.data());
    if(chunk == 4) chunk_init<<<1, 1>>>(s.data());
    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

    ChunkFixtureGraph owned; // Destroy graph before any referenced buffer.
    CUDA_SAFE_CALL(cudaGraphCreate(&owned.graph, 0));
    cudaGraphConditionalHandle handle{};
    CUDA_SAFE_CALL(cudaGraphConditionalHandleCreate(&handle, owned.graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNode_t node{};
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
#if CUDART_VERSION >= 13000
    CUDA_SAFE_CALL(cudaGraphAddNode(&node, owned.graph, nullptr, nullptr, 0, &params));
#else
    CUDA_SAFE_CALL(cudaGraphAddNode(&node, owned.graph, nullptr, 0, &params));
#endif
    CUDA_SAFE_CALL(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,
        params.conditional.phGraph_out[0], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    for(int substep = 0; substep < chunk; ++substep)
    {
        chunk_fixture_feed<<<1, 1>>>(s.data(), pap.data(), next.data(), dispatched.data(),
                                     c.iterations, chunk, true);
        chunk_fixture_work<<<blocks, 256>>>(ap.data(), z.data(), dispatched.data(), c.iterations, c.ap, n);
        if(chunk == 4)
        {
            chunk_alpha<<<1, 1>>>(s.data());
            chunk_dx_r<<<blocks, 256>>>(x.data(), r.data(), p.data(), ap.data(), s.data(), n);
        }
        else
        {
            graph_alpha<<<1, 1>>>(s.data());
            graph_dx_r<<<blocks, 256>>>(x.data(), r.data(), p.data(), ap.data(), s.data(), n);
        }
        chunk_fixture_feed<<<1, 1>>>(s.data(), pap.data(), next.data(), dispatched.data(),
                                     c.iterations, chunk, false);
        if(chunk == 4)
        {
            chunk_beta<<<1, 1>>>(s.data(), c.tolerance, c.fixed);
            chunk_zero_check<<<blocks, 256>>>(r.data(), n, s.data());
            chunk_p_update<<<blocks, 256>>>(p.data(), z.data(), s.data(), n);
            chunk_finalize<<<1, 1>>>(s.data(), c.max_iter, c.tolerance, c.fixed);
            chunk_fixture_freeze<<<(3 * n + 10 + 255) / 256, 256>>>(
                x.data(), r.data(), p.data(), s.data(), saved.data(), audit.data(), n);
            chunk_fixture_mark<<<1, 1>>>(s.data(), audit.data());
        }
        else
        {
            graph_beta<<<1, 1>>>(s.data(), c.tolerance, c.fixed);
            graph_check_zero_rho<<<blocks, 256>>>(r.data(), n, s.data());
            graph_p_continue<<<blocks, 256>>>(p.data(), z.data(), s.data(), n,
                c.max_iter, c.tolerance, c.fixed, handle);
        }
    }
    if(chunk == 4) chunk_continue<<<1, 1>>>(s.data(), handle);
    cudaGraph_t body = nullptr;
    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread, &body));
    CUDA_SAFE_CALL(cudaGraphInstantiate(&owned.exec, owned.graph, nullptr, nullptr, 0));
    CUDA_SAFE_CALL(cudaGraphLaunch(owned.exec, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
    ChunkFixtureResult result;
    result.state.resize(3 * n);
    CUDA_SAFE_CALL(cudaMemcpy(result.s.data(), s.data(), sizeof(result.s), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(result.audit.data(), audit.data(), sizeof(result.audit), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&result.dispatched, dispatched.data(), sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(result.state.data(), x.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(result.state.data() + n, r.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(result.state.data() + 2 * n, p.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
} // namespace

int gipc::PCGSolver::chunk_guard_fixture(const char* output)
{
    std::vector<ChunkFixtureCase> cases;
    for(int position = 1; position <= 4; ++position)
    {
        ChunkFixtureCase c("previous_rho_stop_position_" + std::to_string(position), position);
        c.converged = true;
        if(position == 1) c.tolerance = 1.0;
        else c.next[position - 2] = 1e-8;
        c.next.back() = -1; // Must be ignored by previous-rho stop after x/r update.
        cases.push_back(c);
    }
    for(int steps = 1; steps <= 5; ++steps)
    {
        ChunkFixtureCase c("fixed_iterations_" + std::to_string(steps), steps);
        c.fixed = steps; c.converged = true; cases.push_back(c);
    }
    for(int limit : {2, 4, 5, 6})
    {
        ChunkFixtureCase c("max_iter_" + std::to_string(limit), limit - 1);
        c.max_iter = limit; cases.push_back(c);
    }
    for(int position = 1; position <= 4; ++position)
    {
        ChunkFixtureCase curvature("curvature_error_position_" + std::to_string(position), position);
        curvature.pap.back() = -1; curvature.error = 5; cases.push_back(curvature);
        ChunkFixtureCase rho("rho_error_position_" + std::to_string(position), position);
        rho.next.back() = -1; rho.error = 2; cases.push_back(rho);
    }
    {
        ChunkFixtureCase c("zero_updated_rho_nonzero_residual", 1);
        c.next[0] = 0; c.error = 3; cases.push_back(c);
        c.name = "zero_updated_rho_zero_residual";
        c.ap = 1; c.error = 0; c.converged = true; cases.push_back(c);
    }
    {
        ChunkFixtureCase c("nonfinite_curvature", 2);
        c.pap.back() = std::numeric_limits<double>::quiet_NaN(); c.error = 4; cases.push_back(c);
        c = ChunkFixtureCase("nonfinite_updated_rho", 3);
        c.next.back() = std::numeric_limits<double>::infinity(); c.error = 1; cases.push_back(c);
        c = ChunkFixtureCase("alpha_overflow", 1);
        c.initial = 1e300; c.pap[0] = 1e-300; c.error = 6; cases.push_back(c);
        c = ChunkFixtureCase("beta_overflow", 1);
        c.initial = 1e-300; c.next[0] = 1e300; c.error = 7; cases.push_back(c);
        c = ChunkFixtureCase("curvature_error_before_previous_rho_stop", 1);
        c.pap[0] = 0; c.tolerance = 1; c.error = 5; c.converged = true; cases.push_back(c);
    }

    Json report = {{"schema", "gipc.pcg_chunk_guard_fixture.v1"}, {"passed", true},
        {"scope", "Production K1/K4 kernels and real conditional graphs with scripted operator outputs"},
        {"n", chunk_fixture_n}, {"vector_blocks", 3}, {"candidate_k", 4},
        {"full_A_b_M_pcg_validated", false}, {"performance_certified", false},
        {"host_fallback_max_iter_le_one_covered", false}, {"initial_zero_rhs_entry_covered", false},
        {"allowed_tail_scratch", {"s11", "s12", "Ap", "z"}},
        {"frozen_tail_state", {"x", "r", "p", "s0_through_s9"}}, {"cases", Json::array()}};
    for(const auto& c : cases)
    {
        const auto one = chunk_fixture_run(c, 1), four = chunk_fixture_run(c, 4);
        const bool equal_scalars = std::memcmp(one.s.data(), four.s.data(), 10 * sizeof(double)) == 0;
        const bool equal_vectors = std::memcmp(one.state.data(), four.state.data(),
                                              one.state.size() * sizeof(double)) == 0;
        const unsigned expected_dispatches = unsigned((c.iterations + 3) / 4 * 4);
        const unsigned expected_tail = expected_dispatches - unsigned(c.iterations);
        const bool expected_state = one.s[6] == c.iterations && four.s[6] == c.iterations
            && one.s[9] == c.error && four.s[9] == c.error
            && one.s[5] == int(c.converged) && four.s[5] == int(c.converged)
            && four.s[10] == 0 && one.dispatched == unsigned(c.iterations)
            && four.dispatched == expected_dispatches;
        const bool frozen = four.audit[0] == 1 && four.audit[1] == expected_tail && four.audit[2] == 0;
        const bool pass = equal_scalars && equal_vectors && expected_state && frozen;
        Json row = {{"name", c.name}, {"passed", pass}, {"expected_iterations", c.iterations},
            {"max_iter", c.max_iter}, {"fixed_iterations", c.fixed},
            {"rho_tolerance", c.tolerance}, {"initial_rho", c.initial},
            {"expected_error", c.error}, {"expected_converged_flag", c.converged},
            {"k1_iterations", one.s[6]}, {"k4_iterations", four.s[6]},
            {"k1_error", one.s[9]}, {"k4_error", four.s[9]},
            {"logical_scalars_bitwise_equal", equal_scalars}, {"x_r_p_bitwise_equal", equal_vectors},
            {"expected_state_passed", expected_state}, {"k1_dispatched_substeps", one.dispatched},
            {"k4_dispatched_substeps", four.dispatched}, {"expected_inactive_tail", expected_tail},
            {"checked_inactive_tail", four.audit[1]}, {"tail_bit_mismatches", four.audit[2]},
            {"tail_main_state_frozen", frozen}};
        // Bit representations preserve NaN/Inf diagnostics without invalid JSON.
        row["k1_scalar_bits"] = Json::array(); row["k4_scalar_bits"] = Json::array();
        for(int i = 0; i < 10; ++i)
        {
            std::uint64_t a = 0, b = 0;
            std::memcpy(&a, one.s.data() + i, sizeof(a));
            std::memcpy(&b, four.s.data() + i, sizeof(b));
            row["k1_scalar_bits"].push_back(a); row["k4_scalar_bits"].push_back(b);
        }
        report["cases"].push_back(row);
        if(!pass) report["passed"] = false;
    }
    report["case_count"] = cases.size();
    std::ofstream file(output);
    file << report.dump(2); file.close();
    if(!file) throw std::runtime_error("Cannot write PCG chunk guard fixture report");
    return report["passed"].get<bool>() ? 0 : 1;
}
