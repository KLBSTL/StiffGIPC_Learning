#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace gipc
{
// Independent opt-in. Default capture and stopping rule remain K=1.
inline int pcg_graph_chunk_iterations()
{
    const char* value=std::getenv("GIPC_PCG_GRAPH_CHUNK");
    if(!value || std::strcmp(value,"1")==0)return 1;
    if(std::strcmp(value,"4")!=0)
        throw std::runtime_error("GIPC_PCG_GRAPH_CHUNK must be 1 or 4");
    const char* execution=std::getenv("GIPC_PCG_EXECUTION");
    if(!execution || std::strcmp(execution,"conditional_graph")!=0)
        throw std::runtime_error("K=4 requires conditional_graph PCG");
    if(const char* fused=std::getenv("GIPC_PCG_FUSED_DIAG_UPDATE");fused && std::strcmp(fused,"1")==0)
        throw std::runtime_error("K=4 does not support fused diagonal update");
    return 4;
}
}
