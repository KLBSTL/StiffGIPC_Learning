#pragma once
#include <cuda_tools/cuda_tools.h>
#include <gipc/statistics.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
extern int total_Frames;
namespace gipc
{
inline bool linear_stage_enabled()
{
    const char* path=std::getenv("GIPC_LINEAR_STAGE_LOG");
    const char* from=std::getenv("GIPC_STAGE_FROM_FRAME");
    return path && path[0] && total_Frames+1>=(from?std::stoi(from):1);
}
inline void linear_stage(const char* stage,int index=-1)
{
    if(!linear_stage_enabled())return;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    static std::ofstream log(std::getenv("GIPC_LINEAR_STAGE_LOG"),std::ios::app);
    static uint64_t sequence=0;
    Json row={{"frame",total_Frames+1},{"direction",Statistics::instance().at_current_frame()["newton"].size()},
        {"stage",stage},{"index",index},{"sequence",++sequence},
        {"steady_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()}};
    log<<row.dump()<<std::endl;
    if(!log)throw std::runtime_error("Failed to persist linear stage");
}
}
