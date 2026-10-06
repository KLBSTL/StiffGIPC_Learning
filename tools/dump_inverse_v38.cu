// Export the exact typed inversion kernel used by v38, without another solve.
#include <solver/MASPreconditioner.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <fstream>
#include <vector>
#include <iostream>
#include "../sources/mas_replay_v38/native_kernels.cuh"
int main(int argc,char** argv)
{
    if(argc!=3)return 2;
    using Mat=__GEIGEN__::MasMatrixSymT;
    std::ifstream in(argv[1],std::ios::binary|std::ios::ate);if(!in)return 3;
    size_t bytes=in.tellg();if(bytes%sizeof(Mat))return 4;
    std::vector<Mat> data(bytes/sizeof(Mat));in.seekg(0);in.read((char*)data.data(),bytes);
    Mat *input,*output;
    if(cudaMalloc(&input,bytes)!=cudaSuccess||cudaMalloc(&output,bytes)!=cudaSuccess)return 5;
    cudaMemcpy(input,data.data(),bytes,cudaMemcpyHostToDevice);
    const int numbers=data.size()*BANKSIZE*3;
    __inverse6_P96x96<<<(numbers+95)/96,96>>>(output,input,numbers);
    if(cudaDeviceSynchronize()!=cudaSuccess)return 6;
    cudaMemcpy(data.data(),output,bytes,cudaMemcpyDeviceToHost);
    std::ofstream out(argv[2],std::ios::binary);out.write((const char*)data.data(),bytes);
    cudaFree(input);cudaFree(output);return out?0:7;
}
