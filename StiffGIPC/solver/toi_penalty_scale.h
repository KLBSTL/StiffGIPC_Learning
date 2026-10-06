#pragma once
#include <gipc/type_define.h>
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>

namespace gipc
{
// Contact-free body compliance. Under q=T*q', H'=T^T*H*T and J'=J*T,
// J'*H'^{-1}*J'^T is unchanged. Keep the full 12x12 block: its affine
// diagonal alone is neither a world-space stiffness nor coordinate invariant.
inline Matrix12x12 toi_body_compliance(const Matrix12x12& h)
{
    const double norm=h.norm();
    if(!h.allFinite() || norm==0 || (h-h.transpose()).norm()>1e-10*norm)
        throw std::runtime_error("TOI world penalty: invalid ABD Hessian");
    const Matrix12x12 symmetric=.5*(h+h.transpose());
    Eigen::LLT<Matrix12x12> factor(symmetric);
    if(factor.info()!=Eigen::Success)
        throw std::runtime_error("TOI world penalty: nonpositive ABD pivot");
    const Matrix12x12 inverse=factor.solve(Matrix12x12::Identity());
    if(!inverse.allFinite() || (symmetric*inverse-Matrix12x12::Identity()).norm()>1e-8)
        throw std::runtime_error("TOI world penalty: inaccurate ABD compliance");
    return inverse;
}
} // namespace gipc
