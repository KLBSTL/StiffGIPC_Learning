#pragma once
#include <cuda_runtime.h>
#include <gipc/utils/json.h>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace gipc
{
struct IpcBoundedCCDConfig {bool enabled=false,validate=false;};
inline bool ipc_bounded_ccd_bool(const char* name)
{
    const char* value=std::getenv(name);
    if(!value || std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
}
inline const IpcBoundedCCDConfig& ipc_bounded_ccd_config()
{
    static const IpcBoundedCCDConfig config=[] {
        IpcBoundedCCDConfig c;
        c.enabled=ipc_bounded_ccd_bool("GIPC_BOUNDED_CCD");
        c.validate=ipc_bounded_ccd_bool("GIPC_BOUNDED_CCD_VALIDATE");
        if(c.validate && !c.enabled)
            throw std::runtime_error("GIPC_BOUNDED_CCD_VALIDATE requires GIPC_BOUNDED_CCD=1");
        return c;
    }();
    return config;
}

// Include ipc_bounded_ccd.inl in exactly one CUDA translation unit. The caller
// owns queue[ceil(count/256)] and the unchanged final max-to-host reduction.
// No production state or allocation is retained by this module.
void ipc_bounded_ccd_launch(const double3* x,const int4* pairs,const double3* direction,
    double* queue,double slackness,int count,double requested_bound);
// Diagnostic only: independent buffers, original ACCD truth, per-pair and full
// reciprocal/max/reciprocal comparisons. Inputs are borrowed read-only.
Json ipc_bounded_ccd_compare(const double3* x,const int4* pairs,const double3* direction,
    double slackness,int count,double requested_bound);
int ipc_bounded_ccd_fixture(const char* output);
} // namespace gipc
