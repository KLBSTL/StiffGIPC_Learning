namespace tiny_probe {
using namespace full_apply_probe;
int run(const char* manifest_path,const char* raw_output,bool measure) {
    const auto m=read_json(manifest_path);const fs::path out=raw_output;
    balanced_require(fs::is_directory(out)&&!fs::exists(out/"data.json"),"Fresh output required");
    Json result={{"passed",false},{"action",measure?"tiny_measure":"tiny"},{"device",device()},
        {"cases",Json::array()},{"samples",Json::array()}};
    for(const auto& item:m.at("tiny_cases")){
        const std::string name=item.at("name");if(measure&&name!="bunny_first"&&name!="mixed_first")continue;
        auto L=full_apply_probe::load<double>(item.at("L"),64),b=full_apply_probe::load<double>(item.at("b"),8);
        gipc::BalancedBuffer<double> dl(64+32),da(8+32),db(8);
        std::vector<double> lguard(64+32,1234567.25),aguard(8+32,-7654321.5);
        std::copy(L.begin(),L.end(),lguard.begin()+16);std::copy(b.begin(),b.end(),aguard.begin()+16);
        balanced_cuda(cudaMemcpy(dl.value,lguard.data(),lguard.size()*8,cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(da.value,aguard.data(),aguard.size()*8,cudaMemcpyHostToDevice));
        balanced_cuda(cudaMemcpy(db.value,b.data(),64,cudaMemcpyHostToDevice));
        auto op=[&](int method){
            // Every repeated solve sees the same RHS. This copy is included in both timing arms.
            balanced_cuda(cudaMemcpyAsync(da.value+16,db.value,64,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            if(method)gipc::balanced_tiny_solve_prefetch<<<1,32,0,cudaStreamPerThread>>>(dl.value+16,da.value+16);
            else gipc::balanced_tiny_solve<<<1,1,0,cudaStreamPerThread>>>(dl.value+16,da.value+16);
            balanced_cuda(cudaGetLastError());
        };
        auto guards=[&]{
            const auto x=download(dl.value,lguard.size()),y=download(da.value,aguard.size()),input=download(db.value,8);
            balanced_require(std::memcmp(x.data(),lguard.data(),lguard.size()*8)==0&&std::memcmp(input.data(),b.data(),64)==0,"Tiny readonly changed");
            balanced_require(std::memcmp(y.data(),aguard.data(),16*8)==0&&std::memcmp(y.data()+24,aguard.data()+24,16*8)==0,"Tiny a guard changed");
        };
        std::array<std::array<std::unique_ptr<Graph>,2>,2> graphs;
        std::array<std::vector<double>,2> host;
        Json c={{"name",name},{"graphs",Json::array()},{"warm",Json::array()}};
        for(int method=0;method<2;++method){for(int k=0;k<16;++k)op(method);balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
            host[method]=download(da.value+16,8);for(double v:host[method])balanced_require(std::isfinite(v),"Nonfinite tiny output");guards();
            c["warm"].push_back({{"method",method?"prefetch":"original"},{"execution","host"},
                {"output",save(out/(name+"_host_"+std::to_string(method)+".bin"),host[method])}});
            if(method)balanced_require(std::memcmp(host[0].data(),host[1].data(),64)==0,"Tiny host bytes differ");
            for(int mode=0;mode<2;++mode){const int count=mode?16:1;graphs[method][mode]=std::make_unique<Graph>();
                graphs[method][mode]->capture([&]{for(int k=0;k<count;++k)op(method);});
                auto info=graphs[method][mode]->info;info["tiny_launches"]=tiny_launch_info(*graphs[method][mode],method==1);
                balanced_require(info["tiny_launches"].size()==size_t(count)&&info["nodes"]==2*count
                    &&info["types_cuda_enum"].value("0",0)==count&&info["types_cuda_enum"].value("1",0)==count,"Wrong standalone graph");
                info["method"]=method?"prefetch":"original";info["calls"]=count;c["graphs"].push_back(info);
                balanced_cuda(cudaGraphLaunch(graphs[method][mode]->exec,cudaStreamPerThread));balanced_cuda(cudaStreamSynchronize(cudaStreamPerThread));
                const auto v=download(da.value+16,8);balanced_require(std::memcmp(v.data(),host[0].data(),64)==0,"Tiny Graph bytes differ");guards();
                c["warm"].push_back({{"method",method?"prefetch":"original"},{"execution",mode?"graph16":"graph1"},
                    {"output",save(out/(name+"_g"+std::to_string(count)+"_"+std::to_string(method)+".bin"),v)}});
            }
        }
        if(measure){Events events;const int calls=m.at("calls");balanced_require(calls==2048,"Wrong tiny batch");
            for(int round=0;round<3;++round)for(int mode_order=0;mode_order<2;++mode_order)for(int quartet=0;quartet<4;++quartet)
                for(int position=0;position<4;++position){
                    const int mode=(mode_order+round)%2;const bool outer=position==0||position==3;
                    const int method=int(outer)^((round+quartet)%2),width=mode?16:1;
                    const std::string tag=name+"_g"+std::to_string(width)+"_r"+std::to_string(round+1)+"_q"+std::to_string(quartet+1)+"_p"+std::to_string(position+1);
                    const auto start=Clock::now();balanced_cuda(cudaEventRecord(events.first,cudaStreamPerThread));
                    for(int k=0;k<calls/width;++k)balanced_cuda(cudaGraphLaunch(graphs[method][mode]->exec,cudaStreamPerThread));
                    balanced_cuda(cudaEventRecord(events.last,cudaStreamPerThread));const double submit_ms=elapsed(start);
                    balanced_cuda(cudaEventSynchronize(events.last));const double host_ms=elapsed(start),event_ms=events.ms();
                    const auto verify_start=Clock::now();const auto v=download(da.value+16,8);
                    balanced_require(std::memcmp(v.data(),host[0].data(),64)==0,"Timed tiny bytes differ");guards();
                    const auto file=save(out/(tag+".bin"),v);
                    result["samples"].push_back({{"name",name},{"method",method?"prefetch":"original"},{"label",method?"B":"A"},
                        {"comparison","AB"},{"capture_calls",width},{"calls",calls},{"round",round+1},{"quartet",quartet+1},{"position",position+1},
                        {"tag",tag},{"host_ms",host_ms},{"submit_ms",submit_ms},{"event_ms",event_ms},{"reset_ms",0},
                        {"verify_export_ms",elapsed(verify_start)},{"output",file}});
                }
        }
        guards();c["guards_passed"]=true;result["cases"].push_back(c);
    }
    balanced_require(result["cases"].size()==(measure?2:70)&&result["samples"].size()==(measure?192:0),"Tiny count differs");
    result["passed"]=true;write_json(out/"data.json",result);return 0;
}
}
