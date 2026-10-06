#pragma once
#include <gipc/type_define.h>

namespace gipc::details
{
// One thread owns one complete 3x3 block. Residual components are rounded
// before multiplication, just as in the original two-kernel pipeline.
static __global__ void diag_fused_update_kernel(
    int blocks, const Matrix3x3* inverse, Float* x, Float* r,
    const Float* p, const Float* ap, const Float* alpha, Float* z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= blocks)
        return;
    Vector3 residual;
    for(int axis = 0; axis < 3; ++axis)
    {
        const int j = 3 * i + axis;
        x[j] += *alpha * p[j];
        residual[axis] = r[j] - *alpha * ap[j];
        r[j] = residual[axis];
    }
    const Vector3 value = inverse[i] * residual;
    for(int axis = 0; axis < 3; ++axis)
        z[3 * i + axis] = value[axis];
}
}
