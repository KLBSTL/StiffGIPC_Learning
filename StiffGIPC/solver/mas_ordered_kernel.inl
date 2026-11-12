template<class Accumulator>
__global__ void ad_ordered_restriction(const double* input,float* output,const int* starts,const int* nodes,int clusters)
{
    const int component=blockIdx.x*blockDim.x+threadIdx.x;
    if(component>=clusters*3)return;
    const int row=component/3,axis=component%3;Accumulator sum=0;
    for(int j=starts[row];j<starts[row+1];++j)
        sum+=static_cast<Accumulator>(static_cast<float>(input[nodes[j]*3+axis]));
    output[component]=static_cast<float>(sum);
}
