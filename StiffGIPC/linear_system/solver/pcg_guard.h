#pragma once
#include <cmath>

namespace gipc {
// Zero rho needs a separate exact-residual check; it is not by itself convergence.
__host__ __device__ inline int pcg_rho_error(double rho)
{
    if(!isfinite(rho))return 1;
    return rho<0?2:0;
}
__host__ __device__ inline int pcg_curvature_error(double pap)
{
    if(!isfinite(pap))return 4;
    return pap<=0?5:0;
}
inline const char* pcg_error_name(int code)
{
    switch(code){case 1:return "nonfinite_rho";case 2:return "negative_rho";
    case 3:return "zero_rho_nonzero_residual";case 4:return "nonfinite_curvature";
    case 5:return "nonpositive_curvature";case 6:return "nonfinite_alpha";
    case 7:return "nonfinite_beta";default:return "none";}
}
}
