// Included after the production inverse kernels; this fixture invokes both
// instantiations with the same immutable packed input. It loads no scene.
int gipc::run_mas_pivot_sync_fixture(const char* output)
{
    gipc::Json report={{"fixture","mas_pivot_sync"},{"arithmetic_precision_unchanged",true},
        {"matrix_dimension",BANKSIZE*3},{"production_threads_per_block",32*3},
        {"removed_barriers_per_matrix",(BANKSIZE*3)*(BANKSIZE*3-1)},
        {"batches",gipc::Json::array()},{"passed",false}};
    std::vector<__GEIGEN__::MasMatrixSymT> captured;
    const char* input_path=std::getenv("GIPC_MAS_PIVOT_SYNC_INPUT");
    if(input_path)
    {
        std::ifstream file(input_path,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("Cannot open captured MAS input");
        const std::streamoff bytes=file.tellg();
        constexpr std::streamoff item=sizeof(__GEIGEN__::MasMatrixSymT);
        if(bytes<=0 || bytes%item!=0 || bytes/item>16384)
            throw std::runtime_error("Captured MAS input ABI/count invalid");
        captured.resize(static_cast<size_t>(bytes/item));
        file.seekg(0);file.read(reinterpret_cast<char*>(captured.data()),static_cast<std::streamsize>(bytes));
        if(!file)throw std::runtime_error("Cannot read complete captured MAS input");
        report["captured_input"]=input_path;
    }
    const std::vector<int> counts=captured.empty()?std::vector<int>{1,3,32,129}:
        std::vector<int>{static_cast<int>(captured.size())};
    for(int count:counts)
    {
        constexpr int n=BANKSIZE*3;
        std::vector<__GEIGEN__::MasMatrixSymT> input=captured.empty()?
            std::vector<__GEIGEN__::MasMatrixSymT>(count):captured;
        for(int b=0;captured.empty() && b<count;++b)
        {
            Eigen::Matrix<double,n,n> matrix=Eigen::Matrix<double,n,n>::Zero();
            for(int i=0;i<n;++i)
            {
                // Include padded zero diagonals, full SPD couplings, wide
                // diagonal scales, and nonuniform small off-diagonals.
                matrix(i,i)=b%4==0 && i>=n/2?0.0:
                    b%4==2?std::pow(10.0,6.0*i/(n-1)):1.0+.01*i;
            }
            if(b%4==1 || b%4==3)
                for(int k=0;k<12;++k)
                {
                    Eigen::Matrix<double,n,1> v;
                    for(int i=0;i<n;++i) v[i]=std::sin((i+1)*(k+1)*.071+b*.13);
                    matrix+=(b%4==1?2.0:.01)*v*v.transpose();
                }
            const double scale=b%5==0?1e-5:b%5==1?1e5:1.0;
            matrix*=scale;
            for(int row=0;row<BANKSIZE;++row)
                for(int col=row;col<BANKSIZE;++col)
                {
                    const int index=BANKSIZE*row-row*(row+1)/2+col;
                    for(int r=0;r<3;++r) for(int c=0;c<3;++c)
                        input[b].M[index](r,c)=matrix(3*row+r,3*col+c);
                }
        }
        cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymT> source;
        cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymf> reference,candidate;
        source.resize_discard(count);reference.resize_discard(count);candidate.resize_discard(count);
        CUDA_SAFE_CALL(cudaMemcpy(source.data(),input.data(),count*sizeof(input[0]),cudaMemcpyHostToDevice));
        auto launch=[&](bool fast)
        {
            const int threads=32*3,blocks=(count*n+threads-1)/threads;
            if(fast) __inverse6_P96x96<true><<<blocks,threads>>>(candidate.data(),source.data(),count*n);
            else __inverse6_P96x96<false><<<blocks,threads>>>(reference.data(),source.data(),count*n);
        };
        launch(false);launch(true);
        CUDA_SAFE_CALL(cudaGetLastError());CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::vector<__GEIGEN__::MasMatrixSymf> a(count),b(count);
        CUDA_SAFE_CALL(cudaMemcpy(a.data(),reference.data(),count*sizeof(a[0]),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(b.data(),candidate.data(),count*sizeof(b[0]),cudaMemcpyDeviceToHost));
        std::size_t different=0,entries=0;double maximum=0;
        for(int block=0;block<count;++block)
            for(int tile=0;tile<BANKSIZE*(BANKSIZE+1)/2;++tile)
                for(int r=0;r<3;++r) for(int c=0;c<3;++c)
                {
                    const auto& x=a[block].M[tile](r,c);const auto& y=b[block].M[tile](r,c);
                    if(!std::isfinite(x) || !std::isfinite(y))
                        throw std::runtime_error("MAS pivot fixture nonfinite inverse");
                    different+=std::memcmp(&x,&y,sizeof(x))!=0;++entries;
                    maximum=std::max(maximum,std::abs(double(x)-double(y)));
                }
        if(different) throw std::runtime_error("MAS pivot inverse differs bitwise");
        std::vector<__GEIGEN__::MasMatrixSymT> retained(count);
        CUDA_SAFE_CALL(cudaMemcpy(retained.data(),source.data(),count*sizeof(retained[0]),cudaMemcpyDeviceToHost));
        if(std::memcmp(input.data(),retained.data(),count*sizeof(input[0])))
            throw std::runtime_error("MAS pivot fixture changed packed input");
        gipc::Json batch={{"matrices",count},{"compared_entries",entries},
            {"bitwise_different_entries",different},{"max_absolute_difference",maximum},
            {"input_unchanged_bitwise",true},{"timing_pairs",gipc::Json::array()}};
        // CUDA events enclose repeated kernels only. These micro timings are
        // diagnostic evidence; the complete solver still needs paired runs.
        cudaEvent_t begin,end;
        CUDA_SAFE_CALL(cudaEventCreate(&begin));CUDA_SAFE_CALL(cudaEventCreate(&end));
        for(int pair=0;pair<3;++pair)
            for(int arm=0;arm<2;++arm)
            {
                const bool fast=(pair%2?1-arm:arm)!=0;
                CUDA_SAFE_CALL(cudaEventRecord(begin));
                for(int repeat=0;repeat<8;++repeat) launch(fast);
                CUDA_SAFE_CALL(cudaEventRecord(end));CUDA_SAFE_CALL(cudaEventSynchronize(end));
                float ms=0;CUDA_SAFE_CALL(cudaEventElapsedTime(&ms,begin,end));
                batch["timing_pairs"].push_back({{"pair",pair+1},{"pivot_only_sync",fast},
                    {"replays",8},{"gpu_ms_per_launch",ms/8}});
            }
        CUDA_SAFE_CALL(cudaEventDestroy(begin));CUDA_SAFE_CALL(cudaEventDestroy(end));
        report["batches"].push_back(std::move(batch));
    }
    report["passed"]=true;
    std::ofstream file(output);file<<report.dump(2);file.close();
    if(!file) throw std::runtime_error("Cannot save MAS pivot fixture evidence");
    return 0;
}
