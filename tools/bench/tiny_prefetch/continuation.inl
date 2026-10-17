        unsigned char* workspace=workspaceStorage.value+guard_bytes;
        balanced_require(reinterpret_cast<std::uintptr_t>(workspace)%256==0,"Workspace alignment");
        balanced_cuda(cudaMemset(workspaceStorage.value,0x5a,workspace_bytes+2*guard_bytes));
        balanced_cuda(cudaMemset(workspace,0xa5,workspace_bytes));
        gipc::balanced_blas(cublasSetWorkspace(space.blas.handle,workspace,workspace_bytes));
        auto guards=[&]{
            for(auto b:download(workspaceStorage.value,guard_bytes))balanced_require(b==0x5a,"Workspace left guard");
            for(auto b:download(workspace+workspace_bytes,guard_bytes))balanced_require(b==0x5a,"Workspace right guard");
        };
        auto op=[&](int method){
            if(method==0)base(r.value,z.value);
            else space.apply(r.value,z.value,base,nullptr,Mode::FusedProjection,method==2);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::vector<double>,3> cpu={full_apply_probe::load<double>(item.at("files").at("MAS_expected"),n),
            full_apply_probe::load<double>(item.at("files").at("balanced_expected"),n),{}};cpu[2]=cpu[1];
        std::array<std::vector<double>,3> host_reference;
        std::array<std::array<std::unique_ptr<Graph>,2>,3> graphs;
        auto validate=[&](int method,const std::vector<double>& v){const double error=relative(v,cpu[method]);
            balanced_require(std::isfinite(error)&&error<=1e-3,"CPU action calibration failed");
            if(probe=="zero")for(double x:v)balanced_require(x==0,"Zero action not zero");return error;};
        auto same=[&](const std::vector<double>& a,const std::vector<double>& b){
            balanced_require(a.size()==b.size()&&relative(a,b)<=1e-5,"Original/prefetch full action differs");
        };
        Json c={{"name",name},{"probe",probe},{"graphs",Json::array()},{"warm",Json::array()},
            {"explicit_workspace_bytes",workspace_bytes},{"guard_bytes_each",guard_bytes}};
        for(int method=0;method<3;++method){reset();for(int k=0;k<16;++k)op(method);
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));host_reference[method]=download(z.value,n);validate(method,host_reference[method]);
            if(method==2)same(host_reference[1],host_reference[2]);
            for(int mode=0;mode<2;++mode){reset();graphs[method][mode]=std::make_unique<Graph>();const int count=mode?16:1;
                graphs[method][mode]->capture([&]{for(int k=0;k<count;++k)op(method);});
                auto info=graphs[method][mode]->info;
                balanced_require(info.at("types_cuda_enum").value("10",0)==0&&info.at("types_cuda_enum").value("11",0)==0,"Graph allocation nodes remain");
                size_t opaque=0;info["tiny_launches"]=tiny_launch_info(*graphs[method][mode],method==2,&opaque);
                info["opaque_library_kernels"]=opaque;
                balanced_require(opaque==size_t(method?4*count:0),"Wrong opaque library kernel count");
                balanced_require(info["tiny_launches"].size()==size_t(method?2*count:0),"Wrong tiny launch count");
                reset();const auto start=Clock::now();balanced_cuda(cudaGraphLaunch(graphs[method][mode]->exec,cudaStreamPerThread));
                balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));info["first_launch_host_ms"]=elapsed(start);
                info["method"]=methods[method];info["calls"]=count;c["graphs"].push_back(info);
                const auto v=download(z.value,n);const double error=validate(method,v),difference=relative(v,host_reference[method]);
                same(v,host_reference[method]);if(method==2)same(v,host_reference[1]);guards();
                Json warm={{"method",methods[method]},{"capture_calls",count},{"CPU_relative",error},{"host_relative",difference},
                    {"original_relative",method?relative(v,host_reference[1]):relative(v,host_reference[0])},
                    {"original_bitwise_equal",std::memcmp(v.data(),host_reference[method?1:0].data(),v.size()*sizeof(double))==0},
                    {"output",save(out/(name+"_warm_"+methods[method]+"_g"+std::to_string(count)+".bin"),v)}};
                if(probe=="rhs"){
                    const auto old=full_apply_probe::load<double>(item.at("historical_b").at(method==0?"MAS":"fused"),n);
                    const double d=relative(v,old);balanced_require(d<=1e-3,"Historical native action differs");warm["historical_relative"]=d;
                }
                c["warm"].push_back(warm);
            }
        }
        auto execute=[&](int method,int mode,int calls){
            const int width=mode?16:1;balanced_require(calls>0&&calls%width==0,"Invalid work count");
            for(int k=0;k<calls/width;++k)balanced_cuda(cudaGraphLaunch(graphs[method][mode]->exec,cudaStreamPerThread));
        };
        Events events;
        if(measure){
            const int calls=manifest.at("calls");balanced_require(calls==2048,"Wrong batch");
            for(int round=0;round<3;++round)for(int mode_order=0;mode_order<2;++mode_order)for(int quartet=0;quartet<4;++quartet)
                for(int comparison_order=0;comparison_order<2;++comparison_order)for(int position=0;position<4;++position){
                    const int mode=(mode_order+round)%2,comparison=(comparison_order+quartet+round)%2;
                    const bool outer=position==0||position==3;const int label=int(outer)^((round+quartet)%2);
                    const int method=comparison?(label?2:1):0;
                    const std::string tag="pair_"+name+"_g"+std::to_string(mode?16:1)+"_r"+std::to_string(round+1)+
                        "_q"+std::to_string(quartet+1)+"_c"+std::to_string(comparison)+"_p"+std::to_string(position+1);
                    const double reset_ms=reset();const auto start=Clock::now();
                    balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));execute(method,mode,calls);
                    balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));const double submit_ms=elapsed(start);
                    balanced_cuda(cudaEventSynchronize(events.last));const double host_ms=elapsed(start),event_ms=events.ms();
                    const auto verify_start=Clock::now();const auto v=download(z.value,n);const double error=validate(method,v);
                    same(v,host_reference[method]);if(method==2)same(v,host_reference[1]);guards();
                    const auto file=save(out/(tag+".bin"),v);
                    result["samples"].push_back({{"name",name},{"method",methods[method]},{"capture_calls",mode?16:1},{"calls",calls},
                        {"round",round+1},{"quartet",quartet+1},{"position",position+1},{"comparison",comparison?"AB":"AA"},
                        {"label",label?"B":"A"},{"tag",tag},{"host_ms",host_ms},{"submit_ms",submit_ms},{"event_ms",event_ms},
                        {"reset_ms",reset_ms},{"verify_export_ms",elapsed(verify_start)},{"CPU_relative",error},{"output",file}});
                    result["samples"].back()["original_relative"]=relative(v,host_reference[method?1:0]);
                    result["samples"].back()["original_bitwise_equal"]=std::memcmp(v.data(),host_reference[method?1:0].data(),v.size()*sizeof(double))==0;
                }
        }
        reset();for(auto& s:scratch)if(!s.bytes.empty())balanced_cuda(cudaMemsetAsync(s.p,0xa5,s.bytes.size(),cudaStreamPerThread));
        reset();check_state();bool caught=false;
        try{execute(2,0,16);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));throw std::runtime_error("tiny probe post-Graph exception");}
        catch(const std::runtime_error& e){balanced_require(std::string(e.what())=="tiny probe post-Graph exception","Unexpected exception");caught=true;reset();}
        balanced_require(caught,"Fault not executed");check_state();guards();c["restored"]=true;c["guards_passed"]=true;
        result["cases"].push_back(c);
    }
    balanced_require(result["cases"].size()==(measure?2:6),"Wrong case count");
    balanced_require(result["samples"].size()==(measure?384:0),"Wrong sample count");
    result["passed"]=true;result["action"]=action;write_json(out/"data.json",result);return 0;
}
