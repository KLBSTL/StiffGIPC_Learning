// Rank-eight coordinate-only ablation. Frozen runtime origin is recorded beside this file.
#include "frozen_runtime.cuh"

void run(const fs::path& folder,const fs::path& output){
    need(fs::is_directory(output)&&fs::is_empty(output),"Fresh empty output directory required");
    Json m;std::ifstream(folder/"manifest.json")>>m;
    need(m.at("schema")=="cgstats_pod_gpu_input.v1","Input ABI");
    const int n=m.at("n"),width=m.at("history_width"),steps=m.at("source_iterations"),rank=8;
    need(n>rank&&width>=rank&&width<=128&&steps>=width&&steps<=2000,"Dimension budget");
    size_t free,total;CU(cudaMemGetInfo(&free,&total));need(free>size_t(2)*1024*1024*1024,"2GiB free required");
    auto init=Clock::now();Device d;Sparse target(d,folder,m,"target",n);Mas mas(d,folder,m);
    auto Xh=load<double>(folder,m,"iterates"),last=load<double>(folder,m,"final_source_x"),probes=load<double>(folder,m,"probes");
    auto writes=load<int>(folder,m,"history_writes"),indices=load<int>(folder,m,"bank_indices");
    auto scalars=load<double>(folder,m,"CG_scalars");
    need(Xh.size()==size_t(n)*steps&&last.size()==size_t(n)&&probes.size()==size_t(n)*3,"Input vector shape");
    need(writes.size()%2==0&&indices.size()==size_t(width)&&scalars.size()==size_t(2)*steps&&m.at("source_restarts")==0,"Statistics ABI");
    std::vector<int> observed(width,-1);for(size_t i=0;i<writes.size();i+=2){
        need(writes[i]>=0&&writes[i]<steps&&writes[i+1]>=0&&writes[i+1]<width,"History bounds");observed[writes[i+1]]=writes[i];}
    need(observed==indices,"History/statistic context mismatch");
    for(int i:indices)need(i>=0&&i<steps,"Uninitialized bank index");
    double *X=upload(d,Xh),*final_x=upload(d,last),*input=upload(d,probes),*scalar_device=upload(d,scalars);
    double *bank=d.alloc<double>(size_t(n)*width),*S=d.alloc<double>(size_t(n)*width),*U=d.alloc<double>(width*rank);
    double *Z=d.alloc<double>(size_t(n)*rank),*W=d.alloc<double>(size_t(n)*rank),*E=d.alloc<double>(rank*rank);
    double *tau=d.alloc<double>(rank),*diag=d.alloc<double>(rank),*intermediate=d.alloc<double>(n),*answer=d.alloc<double>(n);
    double *a=d.alloc<double>(rank),*b=d.alloc<double>(rank);int* info=d.alloc<int>(1);
    Product target_product(d,target,rank,Z,W);int workspace_size=1,v;
    SO(cusolverDnDgeqrf_bufferSize(d.solver,n,rank,Z,n,&v));workspace_size=std::max(workspace_size,v);
    SO(cusolverDnDorgqr_bufferSize(d.solver,n,rank,rank,Z,n,tau,&v));workspace_size=std::max(workspace_size,v);
    SO(cusolverDnDpotrf_bufferSize(d.solver,CUBLAS_FILL_MODE_LOWER,rank,E,rank,&v));workspace_size=std::max(workspace_size,v);
    auto workspace=d.alloc<double>(workspace_size);
    std::vector<double> host_U(width*rank),H(width*width),eigenvalues(width),suffix(steps+1);
    Json CPU_last;double coarse_condition=0.;
    auto check_info=[&](const std::string& label){need(d.host(info,1)[0]==0,"cuSOLVER "+label+" failed");};
    auto capture=[&](){for(size_t i=0;i<writes.size();i+=2)
        CU(cudaMemcpyAsync(bank+size_t(n)*writes[i+1],X+size_t(n)*writes[i],size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));};
    auto make_S=[&](){difference<<<unsigned((size_t(n)*width+255)/256),256,0,d.stream>>>(n,width,final_x,bank,S);CU(cudaGetLastError());};
    auto extract=[&](bool qrfree){
        auto start=Clock::now();auto values=d.host(scalar_device,size_t(2)*steps);double readback=millis(start);start=Clock::now();suffix[steps]=0.;
        for(int i=steps-1;i>=0;--i){double alpha=values[i],rho=values[steps+i];
            need(std::isfinite(alpha)&&std::isfinite(rho)&&alpha>0&&rho>0,"Invalid CG energy statistics");
            suffix[i]=suffix[i+1]+alpha*rho;need(std::isfinite(suffix[i]),"Statistic overflow");}
        for(int j=0;j<width;++j)for(int i=0;i<width;++i)H[i+width*j]=suffix[std::max(indices[i],indices[j])];
        double gram_time=millis(start);start=Clock::now();
        Eigen::Map<const Eigen::MatrixXd> map(H.data(),width,width);Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(map);
        need(solver.info()==Eigen::Success,"CPU eigensolve failed");
        for(int i=0;i<width;++i)eigenvalues[i]=solver.eigenvalues()[i];
        need(eigenvalues.back()>0&&eigenvalues[width-rank]>1e-10*eigenvalues.back(),"Insufficient independent POD modes");
        for(int j=0;j<rank;++j){double scale=qrfree?1./std::sqrt(eigenvalues[width-1-j]):1.;
            for(int i=0;i<width;++i)host_U[i+width*j]=scale*solver.eigenvectors()(i,width-1-j);}
        CPU_last={{"statistics_D2H_ms",readback},{"Gram_ms",gram_time},{"eigh_and_coefficients_ms",millis(start)}};
        CU(cudaMemcpyAsync(U,host_U.data(),size_t(width)*rank*sizeof(double),cudaMemcpyHostToDevice,d.stream));
        BL(cublasDgemm(d.blas,CUBLAS_OP_N,CUBLAS_OP_N,n,rank,width,&one,S,n,U,width,&zero,Z,n));};
    auto qr=[&](){SO(cusolverDnDgeqrf(d.solver,n,rank,Z,n,tau,workspace,workspace_size,info));check_info("geqrf");
        diagonals<<<1,32,0,d.stream>>>(n,rank,Z,diag);CU(cudaGetLastError());auto v=d.host(diag,rank);double lo=1e300,hi=0.;
        for(double t:v){need(std::isfinite(t),"Nonfinite QR diagonal");lo=std::min(lo,std::abs(t));hi=std::max(hi,std::abs(t));}
        need(hi>0&&lo>1e-10*hi,"QR rank deficiency");
        SO(cusolverDnDorgqr(d.solver,n,rank,rank,Z,n,tau,workspace,workspace_size,info));check_info("orgqr");};
    auto factor=[&](bool qrfree){BL(cublasDgemm(d.blas,CUBLAS_OP_T,CUBLAS_OP_N,rank,rank,n,&one,Z,n,W,n,&zero,E,rank));
        if(qrfree){auto raw=d.host(E,rank*rank);Eigen::Map<const Eigen::MatrixXd> map(raw.data(),rank,rank);
            need(map.allFinite()&&(map-map.transpose()).norm()<1e-8*std::max(map.norm(),1e-300),"QR-free coarse asymmetry rejected");
            Eigen::MatrixXd symmetric_E=.5*(map+map.transpose());Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(symmetric_E,Eigen::EigenvaluesOnly);
            need(eig.info()==Eigen::Success&&eig.eigenvalues().allFinite()&&eig.eigenvalues()[rank-1]>0&&
                 eig.eigenvalues()[0]>1e-12*eig.eigenvalues()[rank-1],"QR-free coarse rank/conditioning rejected");
            coarse_condition=eig.eigenvalues()[rank-1]/eig.eigenvalues()[0];}
        symmetric<<<1,256,0,d.stream>>>(rank,E);CU(cudaGetLastError());
        SO(cusolverDnDpotrf(d.solver,CUBLAS_FILL_MODE_LOWER,rank,E,rank,workspace,workspace_size,info));check_info("potrf");};
    auto pipeline=[&](bool free){capture();make_S();extract(free);if(!free)qr();target_product.run();factor(free);};
    CU(cudaStreamSynchronize(d.stream));Json result={{"schema","qrfree_cost.v1"},{"rank",rank},{"n",n},
        {"initialization_and_upload_ms",millis(init)},{"scope","FP64 reference only; no native PCG"},{"cases",Json::array()}};
    const bool rejection=m.value("test_rejection_only",false);
    for(bool free:{false,true}){
        if(rejection&&!free)continue;
        const std::string tag=free?"QR_free":"QR";Json entry={{"method",tag},{"pipeline_samples",Json::array()},{"apply_samples",Json::array()}};
        entry["first_pipeline"]=timed(d,[&](){pipeline(free);});
        entry["components"]["history"]=timed(d,capture);entry["components"]["S"]=timed(d,make_S);
        entry["components"]["statistics_extract"]=timed(d,[&](){extract(free);});entry["CPU_breakdown"]=CPU_last;
        if(!free)entry["components"]["QR"]=timed(d,qr);
        entry["components"]["target_AZ"]=timed(d,[&](){target_product.run();});
        entry["components"]["E_check_factor"]=timed(d,[&](){factor(free);});if(free)entry["coarse_condition"]=coarse_condition;
        save(output/("Z_"+tag+".bin"),d.host(Z,size_t(n)*rank));save(output/("W_"+tag+".bin"),d.host(W,size_t(n)*rank));
        save(output/("factor_"+tag+".bin"),d.host(E,rank*rank));
        if(!free)save(output/"history_bank.bin",d.host(bank,size_t(n)*width));
        auto balanced=[&](const double* r){
            BL(cublasDgemv(d.blas,CUBLAS_OP_T,n,rank,&one,Z,n,r,1,&zero,a,1));tiny_solve<<<1,1,0,d.stream>>>(rank,E,a);
            CU(cudaMemcpyAsync(intermediate,r,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
            BL(cublasDgemv(d.blas,CUBLAS_OP_N,n,rank,&minus_one,W,n,a,1,&one,intermediate,1));mas.apply(intermediate,answer);
            BL(cublasDgemv(d.blas,CUBLAS_OP_T,n,rank,&one,W,n,answer,1,&zero,b,1));tiny_solve<<<1,1,0,d.stream>>>(rank,E,b);
            coefficients<<<1,32,0,d.stream>>>(rank,a,b);BL(cublasDgemv(d.blas,CUBLAS_OP_N,n,rank,&one,Z,n,b,1,&one,answer,1));CU(cudaGetLastError());};
        auto expected=load<double>(folder,m,"expected_CG_B_8");entry["action_errors"]=Json::array();
        for(int probe=0;probe<3;++probe){balanced(input+size_t(n)*probe);auto out=d.host(answer,n);
            std::vector<double> ref(expected.begin()+size_t(n)*probe,expected.begin()+size_t(n)*(probe+1));double e=relative(out,ref);
            need(e<1e-8,"Independent CPU action failed");entry["action_errors"].push_back(e);save(output/(tag+"_action_"+std::to_string(probe)+".bin"),out);}
        auto base=[&](){mas.apply(input,answer);};auto augmented=[&](){balanced(input);};
        for(int i=0;i<10;++i){base();augmented();}CU(cudaStreamSynchronize(d.stream));Replay bg(d,base),ag(d,augmented);
        entry["graph_capture_ms"]={{"base",bg.capture_ms},{"candidate",ag.capture_ms}};
        for(int i=0;i<10;++i){bg.run();ag.run();}CU(cudaStreamSynchronize(d.stream));
        ag.run();auto ga=d.host(answer,n);augmented();auto ha=d.host(answer,n);double ge=relative(ga,ha);need(ge<1e-12,"Graph action changed");entry["graph_action_error"]=ge;
        for(const std::string mode:{"graph","host"})for(int pair=0;pair<3;++pair)for(int order=0;order<2;++order){
            bool augmented_arm=((order+pair)%2)!=0;auto t=timed(d,[&](){for(int i=0;i<200;++i){
                if(mode=="graph"){if(augmented_arm)ag.run();else bg.run();}else{if(augmented_arm)augmented();else base();}}});
            entry["apply_samples"].push_back({{"mode",mode},{"pair",pair},{"order",order},{"arm",augmented_arm?"balanced":"MAS_reference"},
                {"repeats",200},{"gpu_us",t["gpu_ms"].get<double>()*5.},{"wall_us",t["wall_ms"].get<double>()*5.}});}
        result["cases"].push_back(entry);std::cout<<tag<<" validated"<<std::endl;
    }
    for(int pair=0;pair<5;++pair)for(int order=0;order<2;++order){bool free=((pair+order)%2)!=0;
        auto t=timed(d,[&](){pipeline(free);});t["pair"]=pair;t["order"]=order;result["cases"][free?1:0]["pipeline_samples"].push_back(t);}
    result["explicit_device_bytes"]=d.bytes;result["passed"]=true;result["performance_certified"]=false;
    std::ofstream(output/"result.json")<<result.dump(2);
}
int main(int argc,char** argv){try{need(argc==3,"Usage: qrfree_cost INPUT_DIR EMPTY_OUTPUT_DIR");run(argv[1],argv[2]);return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<std::endl;if(argc==3&&!fs::exists(fs::path(argv[2])/"failure.json"))
        std::ofstream(fs::path(argv[2])/"failure.json")<<Json{{"passed",false},{"error",e.what()}}.dump(2);return 1;}}
