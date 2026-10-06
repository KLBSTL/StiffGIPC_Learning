#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class DiagPreconditioner : public GlobalPreconditioner
{
  public:
    DiagPreconditioner() = default;
    std::vector<std::uintptr_t> graph_signature() const override
    {
        return {reinterpret_cast<std::uintptr_t>(m_diag3x3.data()), m_diag3x3.size()};
    }


  public:

    virtual void assemble(GIPCTripletMatrix& global_triplets) override;

    virtual void apply(cudatool::CDenseVectorView<gipc::Float> r,
                       cudatool::DenseVectorView<gipc::Float>  z) override;

    void update_and_apply(cudatool::DenseVectorView<Float> x,
                          cudatool::DenseVectorView<Float> r,
                          cudatool::CDenseVectorView<Float> p,
                          cudatool::CDenseVectorView<Float> ap,
                          const Float* alpha,
                          cudatool::DenseVectorView<Float> z);

  private:
    cudatool::DeviceBuffer<gipc::Matrix3x3> m_diag3x3;
};
}  // namespace gipc
