#include <solver/MASPreconditioner.cu>
#include <linear_system/solver/native_balanced_gpu.cuh>
#include <linear_system/utils/spmv.cu>
#include <cuda_tools/cuda_cub_wrappers.h>
#include <cub/block/block_reduce.cuh>
#include <gipc/type_define.h>
#include <filesystem>
#include <map>
#include <memory>
#include <cstring>
int total_Frames=0;
#include "native_kernels.inl"
#include "abd_kernel.inc"
#include "weighted_check_schedule.h"
#include "fixture.inl"
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Expected contracts|measure manifest output");
        return MASPreconditioner::tiny_prefetch_fixture(argv[2],argv[3],argv[1]);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
