// Original MAS, original balanced projections, additive workspace experiment.
#include <solver/MASPreconditioner.cu>
#include <linear_system/solver/native_balanced_gpu.cuh>
#include <gipc/type_define.h>
#include <filesystem>
#include <map>
#include <memory>
#include "abd_kernel.inc"
int total_Frames=0;
#include "workspace_fixture.inl"
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Expected contracts|measure manifest output");
        const std::string mode=argv[1];
        if(mode!="contracts"&&mode!="measure")throw std::runtime_error("Unknown action");
        return MASPreconditioner::full_apply_graph_fixture(argv[2],argv[3],mode=="measure");
    } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
