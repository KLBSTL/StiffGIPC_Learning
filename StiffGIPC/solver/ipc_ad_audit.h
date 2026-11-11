#pragma once
#include <cstdlib>
#include <string>
#include <functional>
#include <vector>
#include <cuda_runtime_api.h>
#include <gipc/utils/json.h>
namespace gipc {
inline bool ad_flag(const char* key){const char* v=std::getenv(key);return v&&std::string(v)=="1";}
inline thread_local int ad_iteration=0,ad_accepted=0;
inline thread_local double ad_beta=1,ad_total_step=0;
inline thread_local Json ad_energy_parts;
inline thread_local bool ad_failure_replay=false;
inline Json ad_hash_device(const void* p,size_t bytes){
    std::vector<unsigned char> value(bytes);
    if(bytes&&cudaMemcpy(value.data(),p,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("A-D state readback failed");
    uint64_t hash=14695981039346656037ull;for(auto v:value){hash^=v;hash*=1099511628211ull;}
    return {{"bytes",bytes},{"fnv1a64",hash}};
}
struct AdFinally {std::function<void()> fn;~AdFinally(){try{fn();}catch(...){}}};
}
