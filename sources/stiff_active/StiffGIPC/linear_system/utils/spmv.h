#pragma once
#include <gipc/type_define.h>

#include <cuda_tools/cuda_all.h>

namespace gipc
{
class Spmv
{
  public:
    // Fixed a=1,b=0 candidate, one scalar partial per 256 stored blocks;
    // count==0 writes one zero partial. Caller owns the reduction workspace.
    void warp_reduce_sym_spmv_quadratic(Eigen::Matrix3d* values,int* rows,int* cols,
        int count,cudatool::CDenseVectorView<Float> x,cudatool::DenseVectorView<Float> y,
        Float* partials);

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y);
};
}  // namespace gipc
