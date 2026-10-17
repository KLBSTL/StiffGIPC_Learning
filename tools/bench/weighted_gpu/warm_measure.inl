// Negative budget cases run before successful contracts to expose state leaks.
if(action=="contracts") {
    max_iter=3;
    for(int mode=0;mode<2;++mode)for(int arm=0;arm<4;++arm) {
        const int method=arm%2?1:2;current_method=method;execution=mode;current_cooldown=arm>=2;
        reset();balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));Json failed;bool rejected=false;
        try{solve_candidate(method!=2,true,method!=2,failed);}
        catch(const NativeBalancedRejected& error){
            balanced_require(std::string(error.what())=="Native diagnostic iteration budget reached","Wrong budget failure");rejected=true;
        }
        balanced_require(rejected&&!failed.at("converged").get<bool>()&&failed.at("iterations")==2
            &&failed.at("last_quality_iteration")==2&&failed.at("trace").back().at("forced_quality_check").get<bool>(),"Budget did not force explicit rejection");
        balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
        const auto file=save(out/(name+"_budget_a"+std::to_string(arm)+"_e"+std::to_string(mode)+"_x.bin"),download(work_x.value,n));
        reset();check_state();guards();c["budget_checks"].push_back({{"arm",arm},{"method",methods[method]},
            {"policy",current_cooldown?"cooldown2":"legacy"},{"execution",mode?"graph_apply1":"host"},
            {"solution_file",file},{"result",failed},{"state_restored",true}});
    }
    max_iter=2001;
}
for(int mode=0;mode<2;++mode)for(int arm=0;arm<4;++arm)
    c["warm"].push_back(run(arm,mode,name+"_warm_a"+std::to_string(arm)+"_e"+std::to_string(mode)));
if(measure) {
    const int comparison_arms[6][2]={{0,0},{2,2},{0,1},{2,3},{0,2},{1,3}};
    const char* comparison_names[6]={"AA_legacy","AA_cooldown","legacy_net","cooldown_net","MAS_policy","cached_policy"};
    for(int round=0;round<3;++round)for(int mo=0;mo<2;++mo)for(int q=0;q<4;++q)
        for(int co=0;co<6;++co)for(int pos=0;pos<4;++pos) {
            const int mode=(mo+round)%2,comparison=(co+q+round)%6;
            const int label=int(pos==0||pos==3)^((round+q)%2),arm=comparison_arms[comparison][label];
            const std::string tag=name+"_r"+std::to_string(round+1)+"_e"+std::to_string(mode)+"_q"+std::to_string(q+1)+"_c"+std::to_string(comparison)+"_p"+std::to_string(pos+1);
            auto row=run(arm,mode,tag);row["round"]=round+1;row["quartet"]=q+1;row["position"]=pos+1;
            row["comparison"]=comparison_names[comparison];row["label"]=label?"B":"A";c["runs"].push_back(row);
        }
}
