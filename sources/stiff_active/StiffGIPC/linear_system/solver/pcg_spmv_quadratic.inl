namespace gipc
{
void PCGSolver::prepare_spmv_quadratic(SizeT count)
{
    spmv_quadratic_requested=spmv_fused_quadratic_requested();
    const bool study=spmv_quadratic_study_requested();
    if(study && mas_fused_dot_study_requested())
        throw std::runtime_error("Choose one protected component study per run");
    spmv_quadratic_reason="disabled_by_request";
    spmv_quadratic_effective=false;spmv_quadratic_supported=false;
    spmv_quadratic_partials_count=0;
    if(spmv_quadratic_requested || study)
    {
        spmv_quadratic_reason=spmv_quadratic_unavailable_reason(count);
        spmv_quadratic_supported=spmv_quadratic_reason.empty();
        spmv_quadratic_effective=spmv_quadratic_requested && spmv_quadratic_supported;
        if(spmv_quadratic_supported)
        {
            spmv_quadratic_partials_count=spmv_quadratic_partial_count();
            if(spmv_quadratic_partials_count<=0 || spmv_quadratic_partials_count>std::numeric_limits<int>::max())
                throw std::runtime_error("Invalid SpMV quadratic partial count");
            spmv_quadratic_partials.resize(spmv_quadratic_partials_count);
            spmv_quadratic_reduce_bytes=0;
            CUDA_SAFE_CALL(cub::DeviceReduce::Sum(nullptr,spmv_quadratic_reduce_bytes,
                spmv_quadratic_partials.data(),reduction_result.data(),
                static_cast<int>(spmv_quadratic_partials_count),cudaStreamPerThread));
            spmv_quadratic_reduce_storage.resize(spmv_quadratic_reduce_bytes);
        }
    }
    auto& info=Statistics::instance().at_current_frame()["newton"].back()["pcg"];
    info["spmv_fused_quadratic_requested"]=spmv_quadratic_requested;
    info["spmv_fused_quadratic_effective"]=spmv_quadratic_effective;
    info["spmv_fused_quadratic_fallback_reason"]=spmv_quadratic_requested?spmv_quadratic_reason:"disabled_by_request";
    info["spmv_fused_quadratic_partial_count"]=spmv_quadratic_partials_count;
}
void PCGSolver::apply_spmv_quadratic(cudatool::CDenseVectorView<Float> input,
    cudatool::DenseVectorView<Float> output,Float* result)
{
    if(!spmv_quadratic_supported || spmv_quadratic_partials_count<=0)
        throw std::runtime_error("SpMV quadratic was not prepared before PCG/capture");
    spmv_quadratic(input,output,spmv_quadratic_partials.data());
    CostScope scope("pcg.fused_quadratic_final_reduce");
    CUDA_SAFE_CALL(cub::DeviceReduce::Sum(spmv_quadratic_reduce_storage.data(),
        spmv_quadratic_reduce_bytes,spmv_quadratic_partials.data(),result,
        static_cast<int>(spmv_quadratic_partials_count),cudaStreamPerThread));
}
Float PCGSolver::apply_spmv_quadratic_host(cudatool::CDenseVectorView<Float> input,
    cudatool::DenseVectorView<Float> output)
{
    apply_spmv_quadratic(input,output,reduction_result.data());
    Float value=0;
    CostScope scope("pcg.scalar_readback",false);
    CUDA_SAFE_CALL(cudaMemcpy(&value,reduction_result.data(),sizeof(value),cudaMemcpyDeviceToHost));
    return value;
}
}
