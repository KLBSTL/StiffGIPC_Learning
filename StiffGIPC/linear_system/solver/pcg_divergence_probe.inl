// Semantic PCG stages. Extra copy kernels are confined to diagnostic runs.
namespace {
__global__ void divergence_record(double* out,int n,int steps,int stage,int iteration,
    const double* graph_state,int iteration_offset,const double* a,const double* b,const double* c,
    double rho,double curvature,double alpha,double beta,double rho_new)
{
    const int k=graph_state?int(graph_state[6])+iteration_offset:iteration;
    if(k<0||k>steps)return;
    const size_t stride=size_t(n)*3+8;
    double* dst=out+(size_t(k)*5+stage)*stride;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){dst[i]=a[i];dst[n+i]=b[i];dst[2*n+i]=c[i];}
    if(i==0) {
        dst[3*n]=k;dst[3*n+1]=stage;
        dst[3*n+2]=graph_state?graph_state[0]:rho;
        dst[3*n+3]=graph_state?graph_state[1]:curvature;
        dst[3*n+4]=graph_state?graph_state[3]:alpha;
        dst[3*n+5]=graph_state?graph_state[4]:beta;
        dst[3*n+6]=graph_state?graph_state[2]:rho_new;
        dst[3*n+7]=1;
    }
}
}
namespace gipc {
void PCGSolver::divergence_trace_begin(SizeT n)
{
    divergence_trace_steps=divergence_trace_enabled?2:0;
    if(divergence_trace_enabled) {
        if(const char* p=std::getenv("GIPC_DIVERGENCE_TRACE_STEPS"))
            divergence_trace_steps=std::clamp(std::atoi(p),1,64);
        divergence_trace.resize((size_t(divergence_trace_steps)+1)*5*(3*n+8));
        CUDA_SAFE_CALL(cudaMemset(divergence_trace.data(),255,divergence_trace.size()*sizeof(double)));
    }
}
void PCGSolver::divergence_trace_save(const std::string& path,SizeT n)
{
    if(!divergence_trace_enabled)return;
    std::vector<double> host(divergence_trace.size());
    CUDA_SAFE_CALL(cudaMemcpy(host.data(),divergence_trace.data(),host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()),host.size()*sizeof(double));
    if(!out)throw std::runtime_error("Cannot write PCG operator trace");
}
}
