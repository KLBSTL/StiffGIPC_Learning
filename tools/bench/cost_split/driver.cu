#include <solver/MASPreconditioner.cu>
#include <linear_system/solver/native_balanced_gpu.cuh>
#include <gipc/type_define.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <filesystem>
#include <map>
#include <memory>
#include "abd_kernel.inc"
int total_Frames=0;
#include "cost_fixture.inl"
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Expected contracts|stability|profile|trace manifest output");
        return MASPreconditioner::full_apply_cost_fixture(argv[2],argv[3],argv[1]);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
