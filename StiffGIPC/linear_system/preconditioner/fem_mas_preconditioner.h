#pragma once
#include <functional>
#include <linear_system/linear_system/i_preconditioner.h>
class BHessian;
class MASPreconditioner;

namespace gipc
{
class FEMLinearSubsystem;

class MAS_Preconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    MASPreconditioner&                  MAS_Prec;
    double*                             masses;
    uint32_t*                           cpNum;
    const cudatool::DeviceBuffer<int4>& collision_pairs;

  public:
    MAS_Preconditioner(FEMLinearSubsystem&                 subsystem,
                       MASPreconditioner&                  mMAS,
                       double*                             mMasses,
                       uint32_t*                           mCpNum,
                       const cudatool::DeviceBuffer<int4>& mCollisionPairs);
    virtual void assemble() override;
    std::vector<std::uintptr_t> graph_signature() const override;
    int fused_dot_nodes() const;
    bool fused_dot_supported() const;
    void apply_fused_dot(cudatool::CDenseVectorView<Float> r,
                         cudatool::DenseVectorView<Float> z,Float* partials,bool prepared_only=false);
    void dot_diagnostic_scratch(const std::function<void(void*,size_t)>& visitor) const;
    Json diagnostic_snapshot(const std::string& prefix) const;
    virtual void apply(cudatool::CDenseVectorView<Float> r,
                       cudatool::DenseVectorView<Float>  z) override;
    //const int preconditioner_id = 1;
};
}  // namespace gipc
