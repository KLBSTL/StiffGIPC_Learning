__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    int valid_items = min(numbers - idof, static_cast<int>(blockDim.x));
    double temp = idx < numbers ? a[idx] * b[idx] : 0.0;

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    temp = BlockReduce(storage).Sum(temp, valid_items);
    if(threadIdx.x == 0)
        squeue[blockIdx.x] = temp;
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

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}

__global__ void audit_residual(double* r,const double* b,const double* ax,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)r[i]=b[i]-ax[i];
}


double My_PCG_General_v_v_Reduction_Algorithm(double*       partials,
                                              const double* A,
                                              const double* B,
                                              double*       result_output,
                                              int           vertexNum)
{
    gipc::CostScope cost_dot("pcg.reduce_with_host_readback");
    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    PCG_vdv_Reduction<<<blockNum, threadNum>>>(partials, A, B, numbers);
    cudatool::DeviceReduce().Sum(partials, result_output, blockNum);

    double result = 0.0;
    {
    gipc::CostScope cost_readback("pcg.scalar_readback",false);
    CUDA_SAFE_CALL(
        cudaMemcpy(&result, result_output, sizeof(result), cudaMemcpyDeviceToHost));
    }
    return result;
}

