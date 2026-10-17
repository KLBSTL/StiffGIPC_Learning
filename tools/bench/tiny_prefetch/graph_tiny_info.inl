namespace full_apply_probe {
Json tiny_launch_info(const Graph& graph,bool prefetch,size_t* opaque_count=nullptr) {
    size_t count=0;balanced_cuda(cudaGraphGetNodes(graph.graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);balanced_cuda(cudaGraphGetNodes(graph.graph,nodes.data(),&count));
    Json result=Json::array();size_t opaque=0;
    for(auto node:nodes){cudaGraphNodeType type;balanced_cuda(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};
        const auto status=cudaGraphKernelNodeGetParams(node,&p);
        // cuBLAS internal driver-loaded kernels lack a Runtime host symbol.
        // Keep the exact opaque count; own-kernel count/geometry must still pass.
        if(status==cudaErrorInvalidDeviceFunction){
            const auto cleared=cudaGetLastError();balanced_require(cleared==cudaSuccess||cleared==status,"Unexpected Graph query error");
            ++opaque;continue;
        }
        balanced_cuda(status);
        const bool original=p.func==reinterpret_cast<void*>(gipc::balanced_tiny_solve);
        const bool candidate=p.func==reinterpret_cast<void*>(gipc::balanced_tiny_solve_prefetch);
        if(!original&&!candidate)continue;
        balanced_require(candidate==prefetch,"Wrong captured tiny kernel");
        balanced_require(p.gridDim.x==1&&p.gridDim.y==1&&p.gridDim.z==1&&p.blockDim.x==(candidate?32:1)
            &&p.blockDim.y==1&&p.blockDim.z==1&&p.sharedMemBytes==0,"Wrong tiny launch geometry");
        cudaFuncAttributes a{};balanced_cuda(cudaFuncGetAttributes(&a,p.func));
        balanced_require(a.sharedSizeBytes==(candidate?512:0)&&a.localSizeBytes==0,"Unexpected tiny shared/local memory");
        result.push_back({{"kernel",candidate?"prefetch":"original"},{"grid",1},{"block",p.blockDim.x},
            {"static_shared_bytes",a.sharedSizeBytes},{"local_bytes",a.localSizeBytes},{"registers_per_thread",a.numRegs},
            {"function_address",reinterpret_cast<std::uintptr_t>(p.func)}});
    }
    if(opaque_count)*opaque_count=opaque;else balanced_require(opaque==0,"Unexpected opaque standalone kernel");
    return result;
}
}
