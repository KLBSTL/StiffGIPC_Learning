// Conditional PCG port from the frozen v3 implementation. Unlike v3, dot
// products retain the official base's CUB reduction, and MAS buffers remain
// dynamically sized. The base checks the previous rho AFTER updating x/r.
namespace
{
__global__ void graph_init(double* s)
{
    s[5]=0;s[6] = 0; s[7] = s[0]; s[8] = 0; s[9] = gipc::pcg_rho_error(s[0]);
}
__global__ void graph_check_zero_rho(const double* r,int n,double* s)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && s[0]==0 && r[i]!=0)
        atomicCAS(reinterpret_cast<unsigned long long*>(s+9),__double_as_longlong(0.0),__double_as_longlong(3.0));
}
__global__ void graph_alpha(double* s)
{
    s[8] = s[0];
    s[3]=0;if(s[9]!=0 || s[0]==0)return;
    int code=gipc::pcg_rho_error(s[0]);if(!code)code=gipc::pcg_curvature_error(s[1]);
    if(code){s[9]=code;return;}
    s[3]=s[0]/s[1];if(!isfinite(s[3])){s[9]=6;s[3]=0;}
}
__global__ void graph_dx_r(double* x, double* r, const double* p,
                          const double* ap, const double* s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n && s[9]==0) { x[i] += s[3]*p[i]; r[i] -= s[3]*ap[i]; }
}
__global__ void graph_beta(double* s,double tol,int fixed_iterations)
{
    if(s[9]!=0)return;
    // Host checks the previous rho before applying the next preconditioner.
    if(fixed_iterations<=0 && fabs(s[8])<=tol*s[7])return;
    if(int code=gipc::pcg_rho_error(s[2])){s[9]=code;s[0]=s[2];return;}
    s[4] = s[0] == 0 ? 0 : s[2]/s[0];
    s[0] = s[2];
    if(!isfinite(s[4])) s[9] = 7;
}
__global__ void graph_p_continue(double* p, const double* z, double* s,
                                 int n, int max_iter, double tol, int fixed_iterations,
                                 cudaGraphConditionalHandle handle)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) p[i] = z[i] + s[4]*p[i];
    if(i == 0)
    {
        s[6] += 1;
        bool converged = (fixed_iterations>0 ? s[6]>=fixed_iterations : fabs(s[8]) <= tol*s[7]) || (s[0]==0 && s[9]==0);
        s[5] = converged ? 1 : 0;
        cudaGraphSetConditional(handle,
            !converged && s[9] == 0 && s[6] < max_iter-1);
    }
}
}

namespace gipc
{
void PCGSolver::release_graph()
{
    if(graph_exec) cudaGraphExecDestroy(graph_exec);
    if(graph) cudaGraphDestroy(graph);
    graph_exec = nullptr; graph = nullptr;
}
PCGSolver::~PCGSolver() { release_graph(); }

SizeT PCGSolver::pcg_graph(cudatool::DenseVectorView<Float> x,
                         cudatool::CDenseVectorView<Float> b, SizeT max_iter)
{
    if(max_iter <= 1 || b.size() > std::numeric_limits<int>::max()
       || max_iter > std::numeric_limits<int>::max())
        return pcg(x,b,max_iter);
    const int n = static_cast<int>(b.size()), blocks = (n+255)/256;
    const char* fused_env = std::getenv("GIPC_PCG_FUSED_DIAG_UPDATE");
    const bool fused_update = (diagnostic_fused_override>=0 ? diagnostic_fused_override==1 :
                              fused_env && std::strcmp(fused_env, "1") == 0)
                              && fused_diag_update_available();
    graph_scalars.resize(10);
    double* s = graph_scalars.data();
    size_t reduce_bytes = 0;
    CUDA_SAFE_CALL(cub::DeviceReduce::Sum(nullptr, reduce_bytes,
                    p.buffer_view().data(), s, blocks, cudaStreamPerThread));
    graph_reduce_storage.resize(reduce_bytes);
    auto dot = [&](const double* a, const double* v, double* partials, double* result)
    {
        PCG_vdv_Reduction<<<blocks,256>>>(partials,a,v,n);
        CUDA_SAFE_CALL(cub::DeviceReduce::Sum(graph_reduce_storage.data(),
            reduce_bytes,partials,result,blocks,cudaStreamPerThread));
    };
    r.buffer_view().copy_from(b.buffer_view());
    // Outside capture: assembly, dynamic allocation and warm-up are complete.
    apply_preconditioner(z,r);
    dot(r.buffer_view().data(), z.buffer_view().data(),p.buffer_view().data(),s);
    p.copy_from(z);
    graph_init<<<1,1>>>(s);
    graph_check_zero_rho<<<blocks,256>>>(r.buffer_view().data(),n,s);
    double initial[10];CUDA_SAFE_CALL(cudaMemcpy(initial,s,sizeof(initial),cudaMemcpyDeviceToHost));
    auto& stats = Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    stats["execution"]="conditional_graph";stats["rho_initial"]=initial[7];stats["iterations"]=0;
    if(initial[9]!=0)fail_pcg(static_cast<int>(initial[9]),initial[0],0);
    if(initial[0]==0){stats["zero_residual"]=true;return 0;}

    auto key = graph_signature();
    for(const void* address : {static_cast<const void*>(x.buffer_view().data()),
            static_cast<const void*>(r.buffer_view().data()),
            static_cast<const void*>(z.buffer_view().data()),
            static_cast<const void*>(p.buffer_view().data()),
            static_cast<const void*>(Ap.buffer_view().data()),
            static_cast<const void*>(graph_reduce_storage.data()),
            static_cast<const void*>(s)})
        key.push_back(reinterpret_cast<std::uintptr_t>(address));
    key.push_back(n); key.push_back(reduce_bytes);
    key.push_back(fused_update);
    key.push_back(diagnostic_fixed_iterations);
    int device = 0; CUDA_SAFE_CALL(cudaGetDevice(&device)); key.push_back(device);
    if(graph_exec && (key != captured_key || max_iter != captured_max_iter
                     || m_config.global_tol_rate != captured_tol))
    { ++invalidations; release_graph(); }
    const bool cache_hit = graph_exec != nullptr;
    double capture_ms=0;
    if(cache_hit) ++cache_hits;
    else
    {
        auto capture_start=std::chrono::steady_clock::now();
        CUDA_SAFE_CALL(cudaGraphCreate(&graph,0));
        cudaGraphConditionalHandle handle{};
        CUDA_SAFE_CALL(cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault));
        cudaGraphNode_t node{};
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = handle;
        params.conditional.type = cudaGraphCondTypeWhile;
        params.conditional.size = 1;
#if CUDART_VERSION >= 13000
        CUDA_SAFE_CALL(cudaGraphAddNode(&node,graph,nullptr,nullptr,0,&params));
#else
        CUDA_SAFE_CALL(cudaGraphAddNode(&node,graph,nullptr,0,&params));
#endif
        CUDA_SAFE_CALL(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,
            params.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        spmv(p.cview(),Ap.view());
        dot(p.buffer_view().data(),Ap.buffer_view().data(),z.buffer_view().data(),s+1);
        graph_alpha<<<1,1>>>(s);
        if(fused_update)
            fused_diag_update(x, r.view(), p.cview(), Ap.cview(), s + 3, z.view());
        else
        {
            graph_dx_r<<<blocks,256>>>(x.buffer_view().data(),r.buffer_view().data(),
                                      p.buffer_view().data(),Ap.buffer_view().data(),s,n);
            apply_preconditioner(z,r);
        }
        dot(r.buffer_view().data(),z.buffer_view().data(),Ap.buffer_view().data(),s+2);
        graph_beta<<<1,1>>>(s,m_config.global_tol_rate,diagnostic_fixed_iterations);
        graph_check_zero_rho<<<blocks,256>>>(r.buffer_view().data(),n,s);
        graph_p_continue<<<blocks,256>>>(p.buffer_view().data(),z.buffer_view().data(),
            s,n,static_cast<int>(max_iter),m_config.global_tol_rate,diagnostic_fixed_iterations,handle);
        cudaGraph_t body = nullptr;
        CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread,&body));
        CUDA_SAFE_CALL(cudaGraphInstantiate(&graph_exec,graph,nullptr,nullptr,0));
        captured_key = std::move(key);
        captured_max_iter = max_iter; captured_tol = m_config.global_tol_rate;
        ++captures;
        capture_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_start).count();
    }
    CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec,cudaStreamPerThread));
    double report[10];
    CUDA_SAFE_CALL(cudaMemcpy(report,s,sizeof(report),cudaMemcpyDeviceToHost));
    stats["execution"] = "conditional_graph";
    stats["fused_diag_update"] = fused_update;
    stats["graph_cache_hit"] = cache_hit;
    stats["graph_capture_instantiate_host_ms"]=capture_ms;
    stats["graph_captures_total"] = captures;
    stats["graph_invalidations_total"] = invalidations;
    stats["rho_initial"] = report[7]; stats["rho_stop"] = report[8];
    stats["iterations"]=report[6];
    stats["iteration_limit"] = report[5] == 0 && report[9]==0;
    if(report[9] != 0)fail_pcg(static_cast<int>(report[9]),report[0],report[1]);
    return report[5] != 0 ? static_cast<SizeT>(report[6]) : max_iter;
}
}
