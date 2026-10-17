        // Storage precedes Space in construction; Graph objects below die first.
        unsigned char* workspace=workspaceStorage.value+guard_bytes;
        balanced_require(reinterpret_cast<std::uintptr_t>(workspace)%256==0,"Workspace alignment");
        balanced_cuda(cudaMemset(workspaceStorage.value,0x5a,workspace_bytes+2*guard_bytes));
        balanced_cuda(cudaMemset(workspace,0xa5,workspace_bytes));
        auto guards=[&](){
            const auto left=download(workspaceStorage.value,guard_bytes),right=download(workspace+workspace_bytes,guard_bytes);
            for(auto b:left)balanced_require(b==0x5a,"Workspace left guard damaged");
            for(auto b:right)balanced_require(b==0x5a,"Workspace right guard damaged");
        };
        auto select_workspace=[&](int user){
            // SetStream resets the default pool. nullptr/0 would DISABLE it.
            gipc::balanced_blas(cublasSetStream(space.blas.handle,cudaStreamPerThread));
            if(user)gipc::balanced_blas(cublasSetWorkspace(space.blas.handle,workspace,workspace_bytes));
        };
        auto op=[&](int method){
            if(method==0)base(r.value,z.value);
            else space.apply(r.value,z.value,base,nullptr,method==2?Mode::FusedProjection:Mode::Reference);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::vector<double>,3> cpu={full_apply_probe::load<double>(item.at("files").at("MAS_expected"),n),
            full_apply_probe::load<double>(item.at("files").at("balanced_expected"),n),{}};cpu[2]=cpu[1];
        std::array<std::array<std::vector<double>,3>,2> host_reference;
        const std::array<std::string,2> workspaces={"default","user4MiB"};
        std::array<std::array<std::array<std::unique_ptr<Graph>,2>,3>,2> graphs;
        int version=0;gipc::balanced_blas(cublasGetVersion(space.blas.handle,&version));
        cublasMath_t math;cublasAtomicsMode_t atomics;cublasPointerMode_t pointer;cudaStream_t stream;
        gipc::balanced_blas(cublasGetMathMode(space.blas.handle,&math));
        gipc::balanced_blas(cublasGetAtomicsMode(space.blas.handle,&atomics));
        gipc::balanced_blas(cublasGetPointerMode(space.blas.handle,&pointer));
        gipc::balanced_blas(cublasGetStream(space.blas.handle,&stream));
        balanced_require(math==CUBLAS_PEDANTIC_MATH&&atomics==CUBLAS_ATOMICS_NOT_ALLOWED&&pointer==CUBLAS_POINTER_MODE_HOST&&stream==cudaStreamPerThread,"BLAS configuration changed");
        Json case_record={{"name",name},{"probe",probe},{"n",n},{"graphs",Json::array()},{"warm",Json::array()},
            {"workspace_bytes",workspace_bytes},{"guard_bytes_each",guard_bytes},{"cublas_version",version},
            {"math_mode",int(math)},{"atomics_mode",int(atomics)},{"pointer_mode",int(pointer)},
            {"same_handle_and_stream",true},{"restoration_passed",false}};
        auto validate=[&](int method,const std::vector<double>& output){const double error=relative(output,cpu[method]);
            balanced_require(std::isfinite(error)&&error<=1e-3,"Native action differs from CPU reference");
            if(probe=="zero")for(double value:output)balanced_require(value==0,"Zero input produced nonzero action");return error;};
        for(int user=0;user<2;++user)for(int method=0;method<3;++method){
            select_workspace(user);reset();for(int k=0;k<calls;++k)op(method);
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));host_reference[user][method]=download(z.value,n);
            validate(method,host_reference[user][method]);guards();
            if(user)balanced_require(relative(host_reference[1][method],host_reference[0][method])<=1e-3,"Workspace changes action");
            for(int g=0;g<2;++g){select_workspace(user);reset();graphs[user][method][g]=std::make_unique<Graph>();const int count=g?16:1;
                graphs[user][method][g]->capture([&]{for(int k=0;k<count;++k)op(method);});
                auto info=graphs[user][method][g]->info;const auto& types=info.at("types_cuda_enum");
                balanced_require(types.value("10",0)==(user||method==0?0:2*count)&&types.value("11",0)==(user||method==0?0:2*count),"Unexpected graph allocation/free nodes");
                reset();const auto start=Clock::now();balanced_cuda(cudaGraphLaunch(graphs[user][method][g]->exec,cudaStreamPerThread));
                balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));info["first_launch_host_ms"]=elapsed(start);
                info["calls"]=count;info["method"]=methods[method];info["workspace"]=workspaces[user];
                validate(method,download(z.value,n));guards();case_record["graphs"].push_back(info);
            }
        }
        auto execute=[&](int method,int mode,int user){
            if(mode==0)for(int k=0;k<calls;++k)op(method);
            else if(mode==1)for(int k=0;k<calls/16;++k)balanced_cuda(cudaGraphLaunch(graphs[user][method][1]->exec,cudaStreamPerThread));
            else for(int k=0;k<calls;++k)balanced_cuda(cudaGraphLaunch(graphs[user][method][0]->exec,cudaStreamPerThread));
        };
        for(int user=0;user<2;++user)for(int method=0;method<3;++method)for(int mode=0;mode<3;++mode){
            select_workspace(user);reset();execute(method,mode,user);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            const auto values=download(z.value,n);const double error=validate(method,values),diff=relative(values,host_reference[0][method]);
            balanced_require(diff<=1e-3,"Host/Graph or workspace action differs");guards();
            Json warm={{"method",methods[method]},{"execution",executions[mode]},{"workspace",workspaces[user]},
                {"CPU_relative",error},{"default_host_relative",diff},
                {"output",save(out/(name+"_warm_"+workspaces[user]+"_"+methods[method]+"_"+executions[mode]+".bin"),values)}};
            if(probe=="rhs"){
                const auto historical=full_apply_probe::load<double>(item.at("historical_b").at(methods[method]),n);
                const double difference=relative(values,historical);balanced_require(difference<=1e-3,"Historical calibration failed");
                warm["historical_native_relative"]=difference;
            }
            case_record["warm"].push_back(warm);
        }
        if(measure){Events events;
            for(int round=0;round<3;++round)for(int group=0;group<6;++group)for(int order=0;order<9;++order)for(int position=0;position<2;++position){
                const int code=(order+group*5+round*3)%9,method=code/3,mode=code%3,user=(round+group+order+position)%2;
                const auto selection_start=Clock::now();select_workspace(user);const double select_ms=elapsed(selection_start);const double reset_ms=reset();
                const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));execute(method,mode,user);
                balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));const double submit_ms=elapsed(start);
                balanced_cuda(cudaEventSynchronize(events.last));const double host_ms=elapsed(start),event_ms=events.ms();
                const auto check_start=Clock::now();const auto values=download(z.value,n);const double error=validate(method,values);
                const double difference=relative(values,host_reference[0][method]);balanced_require(difference<=1e-3,"Measured action mismatch");guards();
                const std::string tag=name+"_r"+std::to_string(round+1)+"_g"+std::to_string(group+1)+"_"+methods[method]+"_"+executions[mode]+"_"+workspaces[user];
                const std::string file=save(out/(tag+".bin"),values);
                result["samples"].push_back({{"name",name},{"round",round+1},{"group",group+1},{"order",order+1},{"position",position+1},
                    {"method",methods[method]},{"execution",executions[mode]},{"workspace",workspaces[user]},
                    {"calls",calls},{"host_ms",host_ms},{"submit_ms",submit_ms},{"event_ms",event_ms},
                    {"select_ms",select_ms},{"reset_ms",reset_ms},{"verify_export_ms",elapsed(check_start)},
                    {"CPU_relative",error},{"default_host_relative",difference},{"output",file}});
            }
        }
        // Both workspace settings execute a real graph before the injected failure.
        reset();for(auto& s:scratch)if(!s.bytes.empty())balanced_cuda(cudaMemsetAsync(s.p,0xa5,s.bytes.size(),cudaStreamPerThread));
        reset();check_state();Json faults=Json::array();
        for(int user=0;user<2;++user){bool caught=false;select_workspace(user);reset();
            try{execute(2,1,user);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));throw std::runtime_error("probe injected after Graph apply");}
            catch(const std::runtime_error& e){balanced_require(std::string(e.what())=="probe injected after Graph apply","Wrong injected exception");caught=true;reset();}
            balanced_require(caught,"Failure contract did not execute");check_state();guards();
            faults.push_back({{"workspace",workspaces[user]},{"restored",true}});
        }
        // Restore the default pool and verify a subsequent host action before final scratch restore.
        select_workspace(0);reset();op(2);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));validate(2,download(z.value,n));
        reset();check_state();guards();case_record["restoration_passed"]=true;case_record["post_graph_exception_restored"]=true;
        case_record["faults"]=faults;case_record["workspace_guards_passed"]=true;case_record["default_pool_restored"]=true;
        case_record["readonly_buffers_checked"]=readonly.size();case_record["scratch_buffers_checked"]=scratch.size();
        result["cases"].push_back(case_record);
    }
    balanced_require(result["cases"].size()==(measure?4:6),"Case count differs");
    balanced_require(result["samples"].size()==(measure?1296:0),"Measured sample count differs");
    result["passed"]=true;write_json(out/(measure?"measurements.json":"contracts.json"),result);return 0;
}
