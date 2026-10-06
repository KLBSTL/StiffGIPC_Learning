#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <gipc/utils/json.h>

class lbvh_f;
class lbvh_e;
namespace gipc
{
struct IpcContactPoolConfig { bool enabled=false,validate=false; };
inline const IpcContactPoolConfig& ipc_contact_pool_config()
{
    static const IpcContactPoolConfig c=[] {
        auto read=[](const char* name) {
            const char* value=std::getenv(name);
            if(!value || std::string(value)=="0")return false;
            if(std::string(value)=="1")return true;
            throw std::runtime_error(std::string(name)+" must be exactly 0 or 1");
        };
        IpcContactPoolConfig v{read("GIPC_CONTACT_POOL"),read("GIPC_CONTACT_POOL_VALIDATE")};
        if(v.validate&&!v.enabled)throw std::runtime_error("GIPC_CONTACT_POOL_VALIDATE requires GIPC_CONTACT_POOL=1");
        return v;
    }();
    return c;
}
// Original primitive identities and orientation; never canonicalize vertex tuples.
// VF: first=vertex, second=face. EE: first=query edge, second=leaf edge.
struct IpcContactPoolIdentity { uint32_t first,second,kind,epoch; };
static_assert(sizeof(IpcContactPoolIdentity)==16,"Pool identity ABI");
struct IpcContactPoolReference
{
    bool valid=false;
    uint32_t count=0,counts[5]={};
    const int4* dcd=nullptr;
    const int4* ccd=nullptr;
    const int* matrix_indices=nullptr;
};
void ipc_contact_pool_begin(lbvh_f&,lbvh_e&,const double3* direction,
    uint32_t vertex_count,double max_alpha,double dHat,uint64_t generation);
IpcContactPoolIdentity* ipc_contact_pool_capture_pass(uint32_t capacity);
uint32_t ipc_contact_pool_capture_epoch();
void ipc_contact_pool_seal(uint32_t final_count);
void ipc_contact_pool_set_trial(double alpha);
bool ipc_contact_pool_try_discrete(lbvh_f&,lbvh_e&,double dHat,int4* dcd,int4* ccd,
    uint32_t* counts,int* matrix_indices,uint32_t capacity);
IpcContactPoolReference ipc_contact_pool_reference();
void ipc_contact_pool_end();
void ipc_contact_pool_reset_stats();
Json ipc_contact_pool_stats_json();
int ipc_contact_pool_fixture(const char* output);
}
