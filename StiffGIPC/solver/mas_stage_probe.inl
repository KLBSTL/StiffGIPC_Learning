// Diagnostic only: existing kernels, private R/Z, fixed upstream stage inputs.
template<class Accumulator>
__global__ void ad_ordered_restriction(const double* input,float* output,const int* starts,const int* nodes,int clusters)
{
    const int component=blockIdx.x*blockDim.x+threadIdx.x;
    if(component>=clusters*3)return;
    const int row=component/3,axis=component%3;Accumulator sum=0;
    // Identical ordered terms and FP32 input rounding in both precision arms.
    for(int j=starts[row];j<starts[row+1];++j)sum+=static_cast<Accumulator>(static_cast<float>(input[nodes[j]*3+axis]));
    output[component]=static_cast<float>(sum);
}
gipc::Json MASPreconditioner::diagnostic_stage_study(const double3* R, double3* Z,
                                                    const std::string& prefix)
{
    if(wide_apply || inverse64 || cholesky || totalNodes<1)
        throw std::runtime_error("MAS stage probe requires nonempty legacy FP32 action");
    using RBuffer=cudatool::DeviceBuffer<Eigen::Vector3f>;
    using ZBuffer=cudatool::DeviceBuffer<Precision_T3>;
    struct RestoreScratch
    {
        RBuffer& r; ZBuffer& z; RBuffer saved_r; ZBuffer saved_z;
        RestoreScratch(RBuffer& rr,ZBuffer& zz)
            :r(rr),z(zz),saved_r(std::move(rr)),saved_z(std::move(zz)){}
        ~RestoreScratch(){r=std::move(saved_r);z=std::move(saved_z);}
    } restore(d_multiLevelR,d_multiLevelZ);
    d_multiLevelR.resize_discard(totalNumberClusters);
    d_multiLevelZ.resize_discard(totalNumberClusters);
    cudatool::DeviceBuffer<double3> input,output;
    input.resize_discard(totalNodes);output.resize_discard(totalNodes);
    CUDA_SAFE_CALL(cudaMemcpy(input.data(),R,totalNodes*sizeof(double3),cudaMemcpyDeviceToDevice));
    const auto saved_signature=graph_signature(); // private diagnostic addresses
    const int repeats=std::getenv("GIPC_AD_MAS_REPEATS")?std::clamp(std::atoi(std::getenv("GIPC_AD_MAS_REPEATS")),3,20):3;
    gipc::Json result={{"scope","same legacy factors/maps, private scratch; no preparation/PCG replay"},
        {"warmups_per_stage",1},{"observations_per_stage",repeats},{"stage_replays",4*(repeats+1)},
        {"dimensions",diagnostic_dimensions()},{"stages",gipc::Json::array()}};
    auto read_bytes=[](const void* device,size_t bytes)
    {
        std::vector<unsigned char> data(bytes);
        if(bytes)CUDA_SAFE_CALL(cudaMemcpy(data.data(),device,bytes,cudaMemcpyDeviceToHost));
        return data;
    };
    auto hash=[](const std::vector<unsigned char>& bytes)
    {
        std::uint64_t value=14695981039346656037ull;
        for(auto b:bytes){value^=b;value*=1099511628211ull;}
        return value;
    };
    auto dump=[&](const std::string& name,const std::vector<unsigned char>& bytes)
    {
        const std::string path=prefix+"_mas_stage_"+name+".bin";
        std::ofstream file(path,std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        if(!file)throw std::runtime_error("Failed to write MAS stage output");
        return path;
    };
    const size_t r_bytes=totalNumberClusters*sizeof(Eigen::Vector3f);
    const size_t z_bytes=totalNumberClusters*sizeof(Precision_T3);
    const size_t out_bytes=totalNodes*sizeof(double3);
    const auto input_bytes=read_bytes(input.data(),out_bytes);
    result["input_r_file"]=dump("input_r",input_bytes);
    result["input_r_fnv1a64"]=hash(input_bytes);
    std::vector<unsigned char> frozen_r,frozen_z,full_reference;
    auto clear_restriction=[&]()
    {
        // Poison overwritten fine rows; normal production clearing of coarse rows.
        CUDA_SAFE_CALL(cudaMemset(d_multiLevelR.data(),0xff,r_bytes));
        CUDA_SAFE_CALL(cudaMemset(d_multiLevelR.data()+totalMapNodes,0,
            (totalNumberClusters-totalMapNodes)*sizeof(Eigen::Vector3f)));
    };
    for(const std::string stage:{"restrict","local","prolong","full_action"})
    {
        gipc::Json row={{"stage",stage},{"output_dtype",stage=="prolong"||stage=="full_action"?"float64":"float32"},
            {"upstream_fixed",true},{"outputs",gipc::Json::array()}};
        for(int repeat=0;repeat<=repeats;++repeat)
        {
            if(stage=="restrict" || stage=="full_action")clear_restriction();
            if(stage=="local")
                CUDA_SAFE_CALL(cudaMemcpy(d_multiLevelR.data(),frozen_r.data(),r_bytes,cudaMemcpyHostToDevice));
            if(stage=="local" || stage=="full_action")
                CUDA_SAFE_CALL(cudaMemset(d_multiLevelZ.data(),0,z_bytes));
            if(stage=="prolong")
                CUDA_SAFE_CALL(cudaMemcpy(d_multiLevelZ.data(),frozen_z.data(),z_bytes,cudaMemcpyHostToDevice));
            if(stage=="prolong" || stage=="full_action")
                CUDA_SAFE_CALL(cudaMemset(output.data(),0xff,out_bytes));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            if(stage=="restrict" || stage=="full_action")BuildMultiLevelR(input.data());
            if(stage=="local" || stage=="full_action")SchwarzLocalXSym_block3();
            if(stage=="prolong" || stage=="full_action")CollectFinalZ(output.data());
            CUDA_SAFE_CALL(cudaGetLastError());
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const void* data=stage=="restrict"?static_cast<const void*>(d_multiLevelR.data()):
                stage=="local"?static_cast<const void*>(d_multiLevelZ.data()):output.data();
            const size_t count=stage=="restrict"?r_bytes:stage=="local"?z_bytes:out_bytes;
            const auto values=read_bytes(data,count);
            bool finite=true;
            for(size_t i=0;i<count;i+=stage=="restrict"||stage=="local"?sizeof(float):sizeof(double))
            {
                if(stage=="restrict"||stage=="local")
                {float v;std::memcpy(&v,values.data()+i,sizeof(v));finite &= std::isfinite(v);}
                else {double v;std::memcpy(&v,values.data()+i,sizeof(v));finite &= std::isfinite(v);}
            }
            if(!finite)throw std::runtime_error("MAS stage output nonfinite/uncovered");
            if(repeat==0)
            {
                if(stage=="restrict")frozen_r=values;
                if(stage=="local")frozen_z=values;
                if(stage=="full_action")full_reference=values;
            }
            row["outputs"].push_back({{"repeat",repeat},{"warmup",repeat==0},{"finite",finite},
                {"fnv1a64",hash(values)},{"file",dump(stage+"_r"+std::to_string(repeat),values)}});
            if(stage=="local" && read_bytes(d_multiLevelR.data(),r_bytes)!=frozen_r)
                throw std::runtime_error("Local action changed frozen restriction input");
            if(stage=="prolong" && read_bytes(d_multiLevelZ.data(),z_bytes)!=frozen_z)
                throw std::runtime_error("Prolongation changed frozen local input");
        }
        result["stages"].push_back(row);
    }
    if(const char* control=std::getenv("GIPC_AD_RESTRICT_CONTROL");control&&std::string(control)=="1"){
        // CSR maps are private to this diagnostic and restored before signature checks.
        auto saved_starts=std::move(d_restriction_starts),saved_nodes=std::move(d_restriction_nodes);
        prepare_restriction_map();
        result["restriction_controls"]=gipc::Json::array();
        auto dot=[](const std::vector<double>& a,const std::vector<double>& b){long double s=0;for(size_t i=0;i<a.size();++i)s+=(long double)a[i]*b[i];return double(s);};
        std::vector<double> r(totalNodes*3),s(totalNodes*3),combined(totalNodes*3),half(totalNodes*3);
        std::memcpy(r.data(),input_bytes.data(),out_bytes);
        const double scale=std::sqrt(dot(r,r)/std::max<size_t>(1,r.size()));
        for(size_t i=0;i<r.size();++i){s[i]=scale*std::sin(double(i+1)*1.618033988749895);combined[i]=r[i]+s[i];half[i]=.5*r[i];}
        for(int mode=0;mode<3;++mode){
            const std::string label=mode==0?"native_fp32":mode==1?"ordered_fp32":"ordered_fp64_accum";
            auto restriction=[&](){
                if(mode==0){clear_restriction();BuildMultiLevelR(input.data());}
                else if(mode==1)ad_ordered_restriction<float><<<(totalNumberClusters*3+255)/256,256>>>(reinterpret_cast<double*>(input.data()),reinterpret_cast<float*>(d_multiLevelR.data()),d_restriction_starts.data(),d_restriction_nodes.data(),totalNumberClusters);
                else ad_ordered_restriction<double><<<(totalNumberClusters*3+255)/256,256>>>(reinterpret_cast<double*>(input.data()),reinterpret_cast<float*>(d_multiLevelR.data()),d_restriction_starts.data(),d_restriction_nodes.data(),totalNumberClusters);
            };
            auto apply=[&](const std::vector<double>& v){
                CUDA_SAFE_CALL(cudaMemcpy(input.data(),v.data(),out_bytes,cudaMemcpyHostToDevice));restriction();
                CUDA_SAFE_CALL(cudaMemset(d_multiLevelZ.data(),0,z_bytes));SchwarzLocalXSym_block3();CollectFinalZ(output.data());
                std::vector<double> z(totalNodes*3);CUDA_SAFE_CALL(cudaMemcpy(z.data(),output.data(),out_bytes,cudaMemcpyDeviceToHost));return z;
            };
            gipc::Json record={{"mode",label},{"restriction",gipc::Json::array()},{"full",gipc::Json::array()}};
            for(int rep=0;rep<=repeats;++rep){
                const auto z=apply(r);const auto rb=read_bytes(d_multiLevelR.data(),r_bytes),zb=read_bytes(output.data(),out_bytes);
                if(rep){record["restriction"].push_back({{"file",dump(label+"_restrict_r"+std::to_string(rep),rb)},{"fnv1a64",hash(rb)}});
                    record["full"].push_back({{"file",dump(label+"_full_r"+std::to_string(rep),zb)},{"fnv1a64",hash(zb)}});}
            }
            const auto zr=apply(r),zs=apply(s),zc=apply(combined),zh=apply(half);
            long double lin=0,lin_norm=0,hom=0,z_norm=0;
            for(size_t i=0;i<r.size();++i){const long double d=zc[i]-zr[i]-zs[i];lin+=d*d;lin_norm+=(long double)zc[i]*zc[i];
                const long double h=zh[i]-.5*zr[i];hom+=h*h;z_norm+=(long double)zr[i]*zr[i];}
            const double rzs=dot(r,zs),szr=dot(s,zr);
            record["r_M_r"]=dot(r,zr);record["s_M_s"]=dot(s,zs);
            record["relative_symmetry_defect"]=std::abs(rzs-szr)/std::max({1e-300,std::abs(rzs),std::abs(szr)});
            record["relative_linearity_defect"]=std::sqrt(double(lin/std::max(lin_norm,1e-300L)));
            record["relative_half_scaling_defect"]=std::sqrt(double(hom/std::max(z_norm,1e-300L)));
            result["restriction_controls"].push_back(record);
        }
        CUDA_SAFE_CALL(cudaMemcpy(input.data(),input_bytes.data(),out_bytes,cudaMemcpyHostToDevice));
        d_restriction_starts=std::move(saved_starts);d_restriction_nodes=std::move(saved_nodes);
    }
    result["input_r_unchanged"]=read_bytes(input.data(),out_bytes)==input_bytes;
    result["private_buffer_addresses_unchanged"]=graph_signature()==saved_signature;
    if(!result["input_r_unchanged"].get<bool>() || !result["private_buffer_addresses_unchanged"].get<bool>())
        throw std::runtime_error("MAS stage probe changed input or addresses");
    // Return a valid diagnostic output. The caller restores its original z.
    CUDA_SAFE_CALL(cudaMemcpy(Z,full_reference.data(),out_bytes,cudaMemcpyHostToDevice));
    return result; // guard restores original owned buffers and addresses
}
