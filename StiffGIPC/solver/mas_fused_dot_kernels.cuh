#pragma once
#include <cub/block/block_reduce.cuh>

// Each thread owns a final FEM double3. All local actions have completed in the
// same stream. Keep the original per-component accumulation precision/order;
// only the subsequent scalar dot reduction has a different grouping.
template<class Value3>
__global__ void mas_collect_final_z_dot(double3* z,const double3* r,
    const Value3* levels,const __GEIGEN__::itable* coarse,const int* map,
    int level_count,int count,double* partials)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    double value=0;
    if(i<count)
    {
        Value3 sum=levels[map[i]];
        const auto table=coarse[i];
        for(int level=1;level<level_count;++level)
        {
            const auto contribution=levels[table.index[level-1]];
            sum.x+=contribution.x;sum.y+=contribution.y;sum.z+=contribution.z;
        }
        const double3 output=make_double3(sum.x,sum.y,sum.z);
        z[i]=output;
        value=__dadd_rn(__dadd_rn(__dmul_rn(r[i].x,output.x),
                                __dmul_rn(r[i].y,output.y)),
                                __dmul_rn(r[i].z,output.z));
    }
    using Reduce=cub::BlockReduce<double,256>;
    __shared__ typename Reduce::TempStorage storage;
    const double result=Reduce(storage).Sum(value);
    if(threadIdx.x==0)partials[blockIdx.x]=result;
}
