// Included inside namespace gipc in the private overlay, after the original kernel.
// Launch contract: grid=(1,1,1), block=(32,1,1), disjoint L[64] and a[8].
static __global__ void balanced_tiny_solve_prefetch(const double* L,double* a) {
    __shared__ double lower[64];
    lower[threadIdx.x]=L[threadIdx.x];
    lower[threadIdx.x+32]=L[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x||blockIdx.x)return;
    for(int i=0;i<8;++i){double t=a[i];for(int j=0;j<i;++j)t-=lower[i+8*j]*a[j];a[i]=t/lower[i+8*i];}
    for(int i=7;i>=0;--i){double t=a[i];for(int j=i+1;j<8;++j)t-=lower[j+8*i]*a[j];a[i]=t/lower[i+8*i];}
}
