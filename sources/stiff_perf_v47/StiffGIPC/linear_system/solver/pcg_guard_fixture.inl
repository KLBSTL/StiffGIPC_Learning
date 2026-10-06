// Boundary cases for the actual device guard kernels and shared host checks.
int gipc::PCGSolver::guard_fixture(const char* output)
{
    struct Case {const char* name;double rho,initial,pap,next,residual;int stage,expected;};
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    const Case cases[]={
        {"positive",1,1,2,1,1,0,0},{"zero_rhs",0,0,0,0,0,0,0},
        {"zero_rho_nonzero_residual",0,0,1,0,1,0,3},{"negative_initial_rho",-1,-1,1,1,1,0,2},
        {"nonfinite_initial_rho",nan,1,1,1,1,0,1},{"negative_curvature",1,1,-1,1,1,0,5},
        {"zero_curvature",1,1,0,1,1,0,5},{"nonfinite_curvature",1,1,nan,1,1,0,4},
        {"alpha_overflow",1e300,1e300,1e-300,1,1,0,6},
        {"negative_updated_rho",1,1,1,-1,1,1,2},{"exact_zero_updated_residual",1,1,1,0,0,1,0},
        {"zero_updated_rho_nonzero_residual",1,1,1,0,1,1,3},
        {"nonfinite_updated_rho",1,1,1,inf,1,1,1},
        {"beta_overflow",1e-300,1e-300,1,1e300,1,1,7},
        {"previous_rho_already_converged",1e-8,1,1,-1,1,1,0}};
    cudatool::DeviceBuffer<double> scalars(10),residual(1);
    gipc::Json report={{"scope","Shared host classification and actual graph scalar guard kernels"},{"cases",gipc::Json::array()},{"passed",true}};
    for(const auto& c:cases)
    {
        double s[10]={};s[0]=c.rho;s[1]=c.pap;s[2]=c.next;s[7]=c.initial;s[8]=c.rho;
        CUDA_SAFE_CALL(cudaMemcpy(scalars.data(),s,sizeof(s),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(residual.data(),&c.residual,sizeof(double),cudaMemcpyHostToDevice));
        int host=0;
        if(c.stage==0)
        {
            graph_init<<<1,1>>>(scalars.data());graph_check_zero_rho<<<1,1>>>(residual.data(),1,scalars.data());graph_alpha<<<1,1>>>(scalars.data());
            host=gipc::pcg_rho_error(c.rho);
            if(!host && c.rho==0 && c.residual!=0)host=3;
            if(!host && c.rho!=0){host=gipc::pcg_curvature_error(c.pap);if(!host && !std::isfinite(c.rho/c.pap))host=6;}
        }
        else
        {
            graph_beta<<<1,1>>>(scalars.data(),1e-4,0);graph_check_zero_rho<<<1,1>>>(residual.data(),1,scalars.data());
            if(!(std::abs(c.rho)<=1e-4*c.initial))
            {host=gipc::pcg_rho_error(c.next);if(!host && c.next==0 && c.residual!=0)host=3;
             if(!host && !std::isfinite(c.next/c.rho))host=7;}
        }
        CUDA_SAFE_CALL(cudaMemcpy(s,scalars.data(),sizeof(s),cudaMemcpyDeviceToHost));
        bool pass=host==c.expected && static_cast<int>(s[9])==c.expected;
        report["cases"].push_back({{"name",c.name},{"expected",c.expected},{"host",host},{"device",s[9]},{"passed",pass}});
        if(!pass)report["passed"]=false;
    }
    std::ofstream(output)<<report.dump(2);
    return report["passed"].get<bool>()?0:1;
}
