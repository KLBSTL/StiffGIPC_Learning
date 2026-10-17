namespace
{
using gipc::Float;

__global__ void abd_preconditioner_apply_kernel(int                                 n,
                                                cudatool::CDenseVectorViewer<Float> r,
                                                cudatool::DenseVectorViewer<Float>  z,
                                                cudatool::Dense1D<gipc::Matrix12x12> inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    z.segment<12>(i * 12).as_eigen() =
        inv(i) * r.segment<12>(i * 12).as_eigen();
}
}  // namespace
