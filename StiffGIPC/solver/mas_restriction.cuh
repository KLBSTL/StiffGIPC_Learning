#pragma once
// Same CSR restriction and FP64 arithmetic as deterministic_restrict. Every
// row has a fixed lane assignment and fixed reduction tree; no atomics. The
// prolongation is unchanged, so its mathematical transpose relation remains.
// Summation order differs from serial and is independently checked on CPU.
__global__ void warp_deterministic_restrict(const double* input,Eigen::Vector3d* output,
                                          const int* starts,const int* nodes,int clusters)
{
    const int thread=blockIdx.x*blockDim.x+threadIdx.x;
    const int row=thread/32,lane=thread%32;
    if(row>=clusters)return; // Uniform across each warp, including padded rows.
    double x=0,y=0,z=0;
    for(int j=starts[row]+lane;j<starts[row+1];j+=32)
    {
        const int vertex=nodes[j];
        x+=input[vertex*3];y+=input[vertex*3+1];z+=input[vertex*3+2];
    }
    for(int shift=16;shift>0;shift/=2)
    {
        x+=__shfl_down_sync(0xffffffff,x,shift);
        y+=__shfl_down_sync(0xffffffff,y,shift);
        z+=__shfl_down_sync(0xffffffff,z,shift);
    }
    if(lane==0)
    {
        double* result=reinterpret_cast<double*>(output)+row*3;
        result[0]=x;result[1]=y;result[2]=z;
    }
}
