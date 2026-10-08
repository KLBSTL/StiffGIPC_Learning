#include <linear_system/utils/spmv.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>

namespace gipc
{
namespace
{
__global__ void scale_y_kernel(int size, Float b, cudatool::DenseVectorViewer<Float> y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        y(i) = b * y(i);
}

__global__ void fill_y_zero_kernel(int size, Float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        y[i] = Float(0);
}

__global__ void warp_reduce_sym_spmv_kernel(Float            a,
                                            Eigen::Matrix3d* Mats3,
                                            int*             rows,
                                            int*             cols,
                                            int              triplet_count,
                                            cudatool::CDenseVectorViewer<Float> x,
                                            Float                                 b,
                                            cudatool::DenseVectorViewer<Float>  y)
{
    using WarpReduceFloat = cub::WarpReduce<Float, 32>;
    auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const bool valid = global_thread_id < triplet_count;
    auto thread_id_in_block = threadIdx.x;
    auto warp_id            = thread_id_in_block / 32;
    auto lane_id            = thread_id_in_block & (32 - 1);

    __shared__ WarpReduceFloat::TempStorage temp_storage_float[256 / 32];

    int     prev_i = -1;
    int     i      = -1;
    char    flags = 1;  // Invalid lanes delimit the last real segment.
    Vector3 vec = Vector3::Zero();

    // set the previous row index
    if(valid && global_thread_id > 0)
    {
        prev_i = rows[global_thread_id - 1];
    }

    if(valid)
    {
        i                = rows[global_thread_id];
        auto j           = cols[global_thread_id];
        auto block_value = Mats3[global_thread_id];
        vec = block_value * x.segment<3>(j * 3).as_eigen();

        if(i != j)  // process lower triangle
        {
            Vector3 vec_ = a * block_value.transpose() * x.segment<3>(i * 3).as_eigen();
            y.segment<3>(j * 3).atomic_add(vec_);
        }
    }

    if(valid)
        flags = (lane_id == 0) || (prev_i != i);

    // CUB's 32-lane collective requires every lane, including the tail.
    vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.x(), flags, cudatool::Plus<Float>{});
    __syncwarp();
    vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.y(), flags, cudatool::Plus<Float>{});
    __syncwarp();
    vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.z(), flags, cudatool::Plus<Float>{});

    if(valid && flags)
    {
        auto seg_y  = y.segment<3>(i * 3);
        auto result = a * vec;
        seg_y.atomic_add(result.eval());
    }
}
}  // namespace

void Spmv::warp_reduce_sym_spmv(Float                         a,
                                Eigen::Matrix3d*              triplet_values,
                                int*                          row_ids,
                                int*                          col_ids,
                                int                           triplet_count,
                                cudatool::CDenseVectorView<Float> x,
                                Float                         b,
                                cudatool::DenseVectorView<Float>  y)
{
    using namespace cudatool;
    constexpr int N = 3;

    if(b != 0)
    {
        LaunchCudaKernal_default(y.size(), 256, 0, scale_y_kernel, y.size(), b, y.viewer());
    }
    else
    {
        LaunchCudaKernal_default(y.size(), 256, 0, fill_y_zero_kernel, y.size(), y.data());
    }

    constexpr int block_dim = 256;
    LaunchCudaKernal_default(triplet_count,
                             block_dim,
                             0,
                             warp_reduce_sym_spmv_kernel,
                             a,
                             triplet_values,
                             row_ids,
                             col_ids,
                             triplet_count,
                             x.cviewer(),
                             b,
                             y.viewer());
}
}  // namespace gipc
