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
            else space.apply(r.value,z.value,base,nullptr,method==2?Mode::FusedProjection:Mode::Reference);
            balanced_cuda(cudaGetLastError());
        };
        std::array<std::vector<double>,3> cpu={full_apply_probe::load<double>(item.at("files").at("MAS_expected"),n),
            full_apply_probe::load<double>(item.at("files").at("balanced_expected"),n),{}};cpu[2]=cpu[1];
        std::array<std::vector<double>,3> host_reference;
        std::array<std::array<std::unique_ptr<Graph>,2>,3> graphs;
        auto validate=[&](int method,const std::vector<double>& v){const double error=relative(v,cpu[method]);
            balanced_require(std::isfinite(error)&&error<=1e-3,"CPU action calibration failed");
            if(probe=="zero")for(double x:v)balanced_require(x==0,"Zero action not zero");return error;};
        Json c={{"name",name},{"probe",probe},{"graphs",Json::array()},{"warm",Json::array()},
            {"explicit_workspace_bytes",workspace_bytes},{"guard_bytes_each",guard_bytes}};
        for(int method=0;method<3;++method){reset();for(int k=0;k<16;++k)op(method);
            balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));host_reference[method]=download(z.value,n);validate(method,host_reference[method]);
            for(int mode=0;mode<2;++mode){reset();graphs[method][mode]=std::make_unique<Graph>();const int count=mode?16:1;
                graphs[method][mode]->capture([&]{for(int k=0;k<count;++k)op(method);});
                auto info=graphs[method][mode]->info;
                balanced_require(info.at("types_cuda_enum").value("10",0)==0&&info.at("types_cuda_enum").value("11",0)==0,"Graph allocation nodes remain");
                reset();const auto start=Clock::now();balanced_cuda(cudaGraphLaunch(graphs[method][mode]->exec,cudaStreamPerThread));
                balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));info["first_launch_host_ms"]=elapsed(start);
                info["method"]=methods[method];info["calls"]=count;c["graphs"].push_back(info);
                const auto v=download(z.value,n);const double error=validate(method,v),difference=relative(v,host_reference[method]);
                balanced_require(difference<=1e-3,"Host/Graph action differs");guards();
                Json warm={{"method",methods[method]},{"capture_calls",count},{"CPU_relative",error},{"host_relative",difference},
                    {"output",save(out/(name+"_warm_"+methods[method]+"_g"+std::to_string(count)+".bin"),v)}};
                if(probe=="rhs"){
                    const auto old=full_apply_probe::load<double>(item.at("historical_b").at(methods[method]),n);
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
        auto window=[&](int method,int mode,int calls,const std::string& tag,bool annotate){
            const double reset_ms=reset();if(annotate)nvtxRangePushA(tag.c_str());
            const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));execute(method,mode,calls);
            balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));const double submit_ms=elapsed(start);
            balanced_cuda(cudaEventSynchronize(events.last));const double host_ms=elapsed(start),event_ms=events.ms();
            if(annotate)nvtxRangePop();
            const auto verify_start=Clock::now();const auto values=download(z.value,n);const double error=validate(method,values);
            balanced_require(relative(values,host_reference[method])<=1e-3,"Measured action differs");guards();
            std::string filetag=tag;std::replace(filetag.begin(),filetag.end(),'/','_');
            const auto file=save(out/(filetag+".bin"),values);
            return Json{{"name",name},{"method",methods[method]},{"capture_calls",mode?16:1},{"calls",calls},{"tag",tag},
                {"host_ms",host_ms},{"submit_ms",submit_ms},{"event_ms",event_ms},{"reset_ms",reset_ms},
                {"verify_export_ms",elapsed(verify_start)},{"CPU_relative",error},{"output",file}};
        };
        if(action=="stability"){
            const int sizes[3]={128,512,2048};
            for(int round=0;round<3;++round)for(int mode_order=0;mode_order<2;++mode_order)for(int level_order=0;level_order<3;++level_order){
                const int mode=(mode_order+round)%2,level=(level_order+round)%3,calls=sizes[level];
                for(int quartet=0;quartet<4;++quartet)for(int position=0;position<4;++position){
                    // ABBA or BAAB: the GPU object/path is EXACTLY the same.
                    const bool outer=position==0||position==3;const int label=int(outer)^((round+quartet)%2);
                    const std::string tag="aa/"+name+"/g"+std::to_string(mode?16:1)+"/c"+std::to_string(calls)+
                        "/r"+std::to_string(round+1)+"/q"+std::to_string(quartet+1)+"/p"+std::to_string(position+1);
                    auto row=window(0,mode,calls,tag,false);row["round"]=round+1;row["quartet"]=quartet+1;
                    row["position"]=position+1;row["label"]=label?"B":"A";result["samples"].push_back(row);
                }
            }
        }
        if(action=="profile"||action=="trace"){
            if(action=="trace"&&!capture_started){balanced_cuda(cudaProfilerStart());capture_started=true;}
            const int sequence[3][3]={{0,1,2},{2,1,0},{1,2,0}};
            for(int repeat=0;repeat<3;++repeat)for(int order=0;order<3;++order){
                const int method=sequence[repeat][order];const std::string tag="cost/"+name+"/"+methods[method]+"/r"+std::to_string(repeat+1);
                auto row=window(method,0,32,tag,true);row["repeat"]=repeat+1;row["order"]=order+1;result["samples"].push_back(row);
            }
        }
        reset();for(auto& s:scratch)if(!s.bytes.empty())balanced_cuda(cudaMemsetAsync(s.p,0xa5,s.bytes.size(),cudaStreamPerThread));
        reset();check_state();bool caught=false;
        try{execute(2,0,16);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));throw std::runtime_error("cost probe post-Graph exception");}
        catch(const std::runtime_error& e){balanced_require(std::string(e.what())=="cost probe post-Graph exception","Unexpected exception");caught=true;reset();}
        balanced_require(caught,"Fault not executed");check_state();guards();c["restored"]=true;c["guards_passed"]=true;
        result["cases"].push_back(c);
    }
    if(capture_started)balanced_cuda(cudaProfilerStop());
    balanced_require(result["cases"].size()==(measure?2:6),"Wrong case count");
    const size_t expected=action=="contracts"?0:action=="stability"?576:18;
    balanced_require(result["samples"].size()==expected,"Wrong measured window count");
    result["passed"]=true;result["action"]=action;result["profiler_capture_used"]=capture_started;
    write_json(out/"data.json",result);return 0;
}
