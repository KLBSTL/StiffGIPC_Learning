// Included inside gipc's anonymous namespace in spmv.cu. The original SRBK
// kernel remains separate. This candidate keeps its per-thread Ap operations,
// and computes a scalar from loaded blocks, never from incomplete atomic Ap.
__global__ void warp_reduce_sym_spmv_quadratic_kernel(
    Float a,Eigen::Matrix3d* Mats3,int* rows,int* cols,int triplet_count,
    cudatool::CDenseVectorViewer<Float> x,cudatool::DenseVectorViewer<Float> y,
    Float* partials)
{
    using WarpReduceFloat=cub::WarpReduce<Float,32>;
    using BlockReduceFloat=cub::BlockReduce<Float,256>;
    const auto global_thread_id=blockDim.x*blockIdx.x+threadIdx.x;
    const auto warp_id=threadIdx.x/32;
    const auto lane_id=threadIdx.x&31;
    __shared__ WarpReduceFloat::TempStorage warp_storage[256/32];
    __shared__ BlockReduceFloat::TempStorage block_storage;
    Float quadratic=0;
    Vector3 vec=Vector3::Zero();
    int i=-1;
    char flags=1;
    const bool valid=global_thread_id<triplet_count;
    // No return here: every thread, including the final block's tail, must
    // participate in both warp and block reductions. An invalid lane cannot
    // skip full-mask WarpReduce and later return to a block collective.
    // Its head flag ends the last valid segment; its zero is never written.
    if(valid)
    {
        i=rows[global_thread_id];
        const int j=cols[global_thread_id];
        const int prev_i=global_thread_id>0?rows[global_thread_id-1]:-1;
        const auto block_value=Mats3[global_thread_id];
        vec=block_value*x.segment<3>(j*3).as_eigen();
        if(i!=j)
        {
            Vector3 vec_=a*block_value.transpose()*x.segment<3>(i*3).as_eigen();
            y.segment<3>(j*3).atomic_add(vec_);
        }
        // Every stored off-diagonal block is symmetrically expanded by the
        // existing operator, even when i>j. Do not filter either orientation.
        quadratic=x.segment<3>(i*3).as_eigen().dot(vec);
        if(i!=j)quadratic*=2;
        flags=(lane_id==0 || prev_i!=i)?1:0;
    }
    vec.x()=WarpReduceFloat(warp_storage[warp_id])
        .HeadSegmentedReduce(vec.x(),flags,cudatool::Plus<Float>{});
    vec.y()=WarpReduceFloat(warp_storage[warp_id])
        .HeadSegmentedReduce(vec.y(),flags,cudatool::Plus<Float>{});
    vec.z()=WarpReduceFloat(warp_storage[warp_id])
        .HeadSegmentedReduce(vec.z(),flags,cudatool::Plus<Float>{});
    if(valid && flags)
    {
        auto result=a*vec;
        y.segment<3>(i*3).atomic_add(result.eval());
    }
    const Float total=BlockReduceFloat(block_storage).Sum(quadratic);
    if(threadIdx.x==0)partials[blockIdx.x]=total;
}
