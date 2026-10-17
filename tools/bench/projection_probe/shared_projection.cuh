#pragma once
// Prototype only. Every lane reaches the barrier, including inactive tail lanes.
static __global__ void shared_output_projection(int n,const double* Z,const double* a,
                                               const double* c,double* z) {
    __shared__ double delta[8];
    if(threadIdx.x<8)delta[threadIdx.x]=a[threadIdx.x]-c[threadIdx.x];
    __syncthreads();
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n))return;
    double sum=0;
    #pragma unroll
    for(int j=0;j<8;++j)sum+=Z[i+size_t(j)*n]*delta[j];
    z[i]+=sum;
}
