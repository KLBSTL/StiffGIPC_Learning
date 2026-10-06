#pragma once
// Fine rows retain the old double -> float conversion and partition mapping.
__global__ void legacy_restrict_fine(const double3* input,Eigen::Vector3f* output,
                                     const int* part,int mapped)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=mapped)return;
    const int vertex=part[row];
    if(vertex>=0)output[row]=Eigen::Vector3f(float(input[vertex].x),float(input[vertex].y),float(input[vertex].z));
    else output[row].setZero();
}
// Only coarse summation changes: fixed contributors/tree, FP64 sum of the
// original rounded FP32 inputs, then FP32 output. Local inverse/prolong unchanged.
__global__ void legacy_restrict_coarse(const double3* input,Eigen::Vector3f* output,
                                       const int* starts,const int* nodes,int mapped,int clusters)
{
    const int thread=blockIdx.x*blockDim.x+threadIdx.x;
    const int row=mapped+thread/32,lane=thread%32;
    if(row>=clusters)return; // uniform within a warp
    double x=0,y=0,z=0;
    for(int j=starts[row]+lane;j<starts[row+1];j+=32)
    {
        const double3 r=input[nodes[j]];
        x+=double(float(r.x));y+=double(float(r.y));z+=double(float(r.z));
    }
    for(int shift=16;shift>0;shift/=2)
    {
        x+=__shfl_down_sync(0xffffffff,x,shift);
        y+=__shfl_down_sync(0xffffffff,y,shift);
        z+=__shfl_down_sync(0xffffffff,z,shift);
    }
    if(lane==0)output[row]=Eigen::Vector3f(float(x),float(y),float(z));
}
