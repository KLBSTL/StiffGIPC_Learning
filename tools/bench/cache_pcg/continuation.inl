        // Immutable block COO and native kernels; no CSR conversion on the GPU.
        const int count=meta.at("blocks"),blocks=(n+255)/256,max_iter=2001;
        gipc::BalancedBuffer<int> rows(count),cols(count);
        gipc::BalancedBuffer<Eigen::Matrix3d> values(count);
        auto row_host=full_apply_probe::load<int>(prefix+"_rows.bin",count);
        auto col_host=full_apply_probe::load<int>(prefix+"_cols.bin",count);
        auto val_host=full_apply_probe::load<Eigen::Matrix3d>(prefix+"_values.bin",count);
        balanced_cuda(cudaMemcpy(rows.value,row_host.data(),count*sizeof(int),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(cols.value,col_host.data(),count*sizeof(int),cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(values.value,val_host.data(),count*sizeof(Eigen::Matrix3d),cudaMemcpyHostToDevice));
        readonly.emplace_back("rows",rows.value,count*sizeof(int));readonly.emplace_back("cols",cols.value,count*sizeof(int));
        readonly.emplace_back("values",values.value,count*sizeof(Eigen::Matrix3d));
        gipc::BalancedBuffer<double> b(n),work_x(n),p(n),Ap(n),quality_r(n),quality_z(n),residual_ax(n),dot_partials(blocks),reduction_result(1);
        upload(b.value,prefix+"_rhs.bin",n);
        balanced_require(download(b.value,n)==download(r.value,n),"RHS differs from sealed apply input");
        // r becomes the recurrence scratch, so remove only its read-only entry.
        readonly.erase(std::remove_if(readonly.begin(),readonly.end(),[](const Saved& s){return s.name=="r";}),readonly.end());
        readonly.emplace_back("b",b.value,n*sizeof(double));
        gipc::Spmv spmv;
        auto multiply=[&](const double* input,double* output){
            spmv.warp_reduce_sym_spmv(1.,values.value,rows.value,cols.value,count,
                cudatool::CDenseVectorView<double>{input,n},0.,cudatool::DenseVectorView<double>{output,n});
            balanced_cuda(cudaGetLastError());
        };
        auto check=[](cudaError_t status){balanced_cuda(status);};
        auto milliseconds=[&](Clock::time_point start){return elapsed(start);};
        auto dot=[&](const double* a,const double* c){return My_PCG_General_v_v_Reduction_Algorithm(dot_partials.value,a,c,reduction_result.value,n);};
        const double rhs_norm2=dot(b.value,b.value);
        reset();base(b.value,quality_z.value);const double common_bMb=dot(b.value,quality_z.value);
        balanced_require(std::isfinite(rhs_norm2)&&rhs_norm2>0&&std::isfinite(common_bMb)&&common_bMb>0,"Invalid RHS quadratic");
        multiply(b.value,residual_ax.value);
        Json c={{"name",name},{"n",n},{"rhs_norm2",rhs_norm2},{"common_bMb",common_bMb},
            {"native_A_b_file",save(out/(name+"_native_A_b.bin"),download(residual_ax.value,n))},
            {"native_MAS_b_file",save(out/(name+"_native_MAS_b.bin"),download(quality_z.value,n))},
            {"graphs",Json::array()},{"warm",Json::array()},{"runs",Json::array()},{"zero",Json::array()}};
        auto direct=[&](int method){
            if(method==2)base(r.value,z.value);
            else space.apply(r.value,z.value,base,nullptr,Mode::FusedProjection,method?2:0);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::unique_ptr<Graph>,3> graphs;
        for(int method=0;method<3;++method){
            reset();direct(method);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            graphs[method]=std::make_unique<Graph>();graphs[method]->capture([&]{direct(method);});
            size_t opaque=0;auto info=tiny_launch_info(*graphs[method],method==1?2:0,&opaque);
            balanced_require(info.size()==(method==2?0:2)&&opaque==(method==2?0:4),"Wrong PCG apply graph");
            c["graphs"].push_back({{"method",methods[method]},{"tiny",info},{"opaque",opaque},{"info",graphs[method]->info}});
        }
        int current_method=0,execution=0;
        auto op=[&](int method){
            if(execution)balanced_cuda(cudaGraphLaunch(graphs[method]->exec,cudaStreamPerThread));else direct(method);
        };
        const std::string balanced_fault="none";
        using gipc::NativeBalancedRejected;using gipc::native_quality_check_due;
        #include "recurrence.inl"
        auto run=[&](int method,int mode,const std::string& tag){
            current_method=method;execution=mode;
            double reset_ms=reset();auto t=Clock::now();
            balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));reset_ms+=elapsed(t);
            Events events;Json output;
            const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));
            solve_candidate(method!=2,true,method!=2,output);
            balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));balanced_cuda(cudaEventSynchronize(events.last));
            const double host_ms=elapsed(start),event_ms=events.ms();const auto export_start=Clock::now();
            auto x=download(work_x.value,n);for(double value:x)balanced_require(std::isfinite(value),"Nonfinite solution");
            const std::string file=save(out/(tag+"_x.bin"),x);
            // Explicit terminal vector and native MAS action are outside timing.
            multiply(work_x.value,residual_ax.value);
            audit_residual<<<blocks,256,0,cudaStreamPerThread>>>(quality_r.value,b.value,residual_ax.value,n);
            base(quality_r.value,quality_z.value);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            auto residual=download(quality_r.value,n),weighted=download(quality_z.value,n);
            const auto rf=save(out/(tag+"_residual.bin"),residual),mf=save(out/(tag+"_MAS_residual.bin"),weighted);
            const double terminal_l2=std::sqrt(dot(quality_r.value,quality_r.value)/rhs_norm2);
            const double terminal_mas=std::sqrt(dot(quality_r.value,quality_z.value)/common_bMb);
            balanced_require(std::isfinite(terminal_l2)&&std::isfinite(terminal_mas)&&terminal_l2<=.0095&&terminal_mas<=.0095,"Post-solve explicit gate failed");
            const double export_ms=elapsed(export_start);
            reset();check_state();guards();
            return Json{{"method",methods[method]},{"execution",mode?"graph_apply1":"host"},{"solve_host_ms",host_ms},{"solve_event_ms",event_ms},
                {"reset_host_ms",reset_ms},{"export_audit_host_ms",export_ms},{"solution_file",file},{"residual_file",rf},{"MAS_residual_file",mf},
                {"terminal_l2",terminal_l2},{"terminal_MAS",terminal_mas},{"result",output}};
        };
        // Algebra-only zero contracts; the nonzero-RHS PCG recurrence is not run.
        for(int mode=0;mode<2;++mode)for(int method=0;method<3;++method){
            balanced_cuda(cudaMemsetAsync(work_x.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            balanced_cuda(cudaMemsetAsync(r.value,0,size_t(n)*sizeof(double),cudaStreamPerThread));
            execution=mode;op(method);multiply(work_x.value,residual_ax.value);
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            for(double value:download(residual_ax.value,n))balanced_require(value==0,"A zero input not zero");
            for(double value:download(z.value,n))balanced_require(value==0,"Preconditioner zero input not zero");
            const auto file=save(out/(name+"_zero_m"+std::to_string(method)+"_e"+std::to_string(mode)+".bin"),download(z.value,n));
            c["zero"].push_back({{"execution",mode?"graph_apply1":"host"},{"method",methods[method]},{"output_file",file},{"algebra_only",true}});
        }
        for(int mode=0;mode<2;++mode)for(int method=0;method<3;++method)
            c["warm"].push_back(run(method,mode,name+"_warm_m"+std::to_string(method)+"_e"+std::to_string(mode)));
        if(measure){
            for(int round=0;round<3;++round)for(int mo=0;mo<2;++mo)for(int q=0;q<4;++q)
                for(int co=0;co<2;++co)for(int pos=0;pos<4;++pos){
                    const int mode=(mo+round)%2,comparison=(co+q+round)%2;
                    const int label=int(pos==0||pos==3)^((round+q)%2),method=comparison?label:0;
                    const std::string tag=name+"_r"+std::to_string(round+1)+"_e"+std::to_string(mode)+"_q"+std::to_string(q+1)+"_c"+std::to_string(comparison)+"_p"+std::to_string(pos+1);
                    auto row=run(method,mode,tag);row["round"]=round+1;row["quartet"]=q+1;row["position"]=pos+1;
                    row["comparison"]=comparison?"AB":"AA";row["label"]=label?"B":"A";c["runs"].push_back(row);
                }
        }
        reset();check_state();guards();c["readonly_and_scratch_unchanged"]=true;result["cases"].push_back(c);
    }
    result["passed"]=true;result["action"]=action;result["expected_measured_solves"]=measure?384:0;
    write_json(out/"data.json",result);return 0;
}
