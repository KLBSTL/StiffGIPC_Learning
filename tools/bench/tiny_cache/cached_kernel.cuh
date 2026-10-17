// Private prototype: grid1/block32, disjoint L[64] and a[8], FP64 only.
static __global__ void balanced_tiny_solve_cached(const double* L,double* a) {
    __shared__ double lower[64];
    lower[threadIdx.x]=L[threadIdx.x];
    lower[threadIdx.x+32]=L[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x||blockIdx.x)return;
    double work[8];
    #pragma unroll
    for(int i=0;i<8;++i)work[i]=a[i];
    #pragma unroll
    for(int i=0;i<8;++i){
        double t=work[i];
        #pragma unroll
        for(int j=0;j<i;++j)t-=lower[i+8*j]*work[j];
        work[i]=t/lower[i+8*i];
    }
    #pragma unroll
    for(int i=7;i>=0;--i){
        double t=work[i];
        #pragma unroll
        for(int j=i+1;j<8;++j)t-=lower[j+8*i]*work[j];
        work[i]=t/lower[i+8*i];
    }
    #pragma unroll
    for(int i=0;i<8;++i)a[i]=work[i];
}
