// Independent diagnostic of symmetric local factors. No regularization or inverse.
constexpr int CHOL_N=BANKSIZE*3;
__device__ double chol_input(const __GEIGEN__::MasMatrixSymT& m,int r,int c)
{
    const int br=r/3,bc=c/3;
    if(br<=bc)return m.M[BANKSIZE*br-br*(br+1)/2+bc](r%3,c%3);
    return m.M[BANKSIZE*bc-bc*(bc+1)/2+br](c%3,r%3);
}
__global__ void symmetric_cholesky(const __GEIGEN__::MasMatrixSymT* input,double* factors,int* status)
{
    const int b=blockIdx.x,i=threadIdx.x;__shared__ double l[CHOL_N][CHOL_N];
    __shared__ int bad;
    if(i==0)bad=0;
    for(int j=0;j<CHOL_N;++j)
    {
        double v=.5*chol_input(input[b],i,j)+.5*chol_input(input[b],j,i);
        if(i==j && v==0)v=1; // Same unused/padded diagonal convention as native MAS.
        l[i][j]=v;
    }
    __syncthreads();
    for(int j=0;j<CHOL_N;++j)
    {
        if(i==j)
        {
            double p=l[j][j];for(int k=0;k<j;++k)p-=l[j][k]*l[j][k];
            if(!(p>0)||!isfinite(p)){bad=j+1;l[j][j]=1;}else l[j][j]=sqrt(p);
        }
        __syncthreads();
        if(bad)break;
        if(i>j)
        {
            double v=l[i][j];for(int k=0;k<j;++k)v-=l[i][k]*l[j][k];
            l[i][j]=v/l[j][j];
        }
        __syncthreads();
    }
    for(int j=0;j<CHOL_N;++j)factors[(b*CHOL_N+i)*CHOL_N+j]=j<=i?l[i][j]:0;
    if(i==0)status[b]=bad;
}
__global__ void cholesky_action(const double* factors,const Eigen::Vector3d* rhs,double3* output)
{
    const int b=blockIdx.x,i=threadIdx.x;const double* l=factors+b*CHOL_N*CHOL_N;
    __shared__ double w[CHOL_N];w[i]=reinterpret_cast<const double*>(rhs)[b*CHOL_N+i];
    __syncthreads();
    for(int j=0;j<CHOL_N;++j)
    {
        if(i==j)w[j]/=l[j*CHOL_N+j];__syncthreads();
        if(i>j)w[i]-=l[i*CHOL_N+j]*w[j];__syncthreads();
    }
    for(int j=CHOL_N-1;j>=0;--j)
    {
        if(i==j)w[j]/=l[j*CHOL_N+j];__syncthreads();
        if(i<j)w[i]-=l[j*CHOL_N+i]*w[j];__syncthreads();
    }
    reinterpret_cast<double*>(output)[b*CHOL_N+i]=w[i];
}
// One warp owns a local solve. Each lane keeps up to three rows in registers;
// every row retains the original j-ordered subtraction and division sequence.
// The full warp participates in every shuffle, including inactive rows.
__global__ void cholesky_action_warp(const double* factors,const Eigen::Vector3d* rhs,double3* output)
{
    static_assert(CHOL_N>=32 && CHOL_N<=96,"Unsupported MAS block size");
    constexpr int ROWS=(CHOL_N+31)/32;
    const int b=blockIdx.x,lane=threadIdx.x;
    const double* l=factors+b*CHOL_N*CHOL_N;
    // Coalesced factor loading, padded rows to avoid strided shared-bank access.
    // Only the lower triangular entries are read by either substitution.
    __shared__ double cached[CHOL_N][CHOL_N+1];
    for(int index=lane;index<CHOL_N*CHOL_N;index+=32)
        cached[index/CHOL_N][index%CHOL_N]=l[index];
    __syncwarp();
    const double* input=reinterpret_cast<const double*>(rhs)+b*CHOL_N;
    double w[ROWS];
    #pragma unroll
    for(int k=0;k<ROWS;++k){int i=lane+32*k;w[k]=i<CHOL_N?input[i]:0;}
    for(int j=0;j<CHOL_N;++j)
    {
        double pivot=0;
        if(lane==(j&31)){w[j/32]/=cached[j][j];pivot=w[j/32];}
        pivot=__shfl_sync(0xffffffffu,pivot,j&31);
        #pragma unroll
        for(int k=0;k<ROWS;++k)
        {const int i=lane+32*k;if(i>j && i<CHOL_N)w[k]-=cached[i][j]*pivot;}
    }
    for(int j=CHOL_N-1;j>=0;--j)
    {
        double pivot=0;
        if(lane==(j&31)){w[j/32]/=cached[j][j];pivot=w[j/32];}
        pivot=__shfl_sync(0xffffffffu,pivot,j&31);
        #pragma unroll
        for(int k=0;k<ROWS;++k)
        {const int i=lane+32*k;if(i<j)w[k]-=cached[j][i]*pivot;}
    }
    #pragma unroll
    for(int k=0;k<ROWS;++k){int i=lane+32*k;if(i<CHOL_N)reinterpret_cast<double*>(output)[b*CHOL_N+i]=w[k];}
}
// CSR rows list original FEM nodes contributing to each hierarchy cluster.
// One scalar per thread, sorted FEM-node order, no atomics or parallel reduction.
__global__ void deterministic_restrict(const double* input,Eigen::Vector3d* output,
                                     const int* starts,const int* nodes,int clusters)
{
    int scalar=blockIdx.x*blockDim.x+threadIdx.x;if(scalar>=clusters*3)return;
    int row=scalar/3,axis=scalar%3;double value=0;
    for(int j=starts[row];j<starts[row+1];++j)value+=input[nodes[j]*3+axis];
    reinterpret_cast<double*>(output)[scalar]=value;
}
