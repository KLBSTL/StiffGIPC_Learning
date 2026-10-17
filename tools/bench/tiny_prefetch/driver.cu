#include <solver/MASPreconditioner.cu>
#include <linear_system/solver/native_balanced_gpu.cuh>
#include <gipc/type_define.h>
#include <filesystem>
#include <map>
#include <memory>
#include <cstring>
int total_Frames=0;
#include "abd_kernel.inc"
#include "tiny_fixture.inl"
#include "tiny_contracts.inl"
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Expected tiny|tiny_measure|contracts|measure manifest output");
        if(std::string(argv[1])=="tiny"||std::string(argv[1])=="tiny_measure")
            return tiny_probe::run(argv[2],argv[3],std::string(argv[1])=="tiny_measure");
        return MASPreconditioner::tiny_prefetch_fixture(argv[2],argv[3],argv[1]);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
