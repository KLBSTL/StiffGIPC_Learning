// F and its probes remain byte-for-byte unchanged. W uses the same sorted leaf
// input order, two-child DFS order, x-endpoint body keys and original classifier.
struct P3QueueState {unsigned long long count;uint32_t stack_error,reserved;};
static_assert(sizeof(P3QueueState)==16,"P3 queue header ABI");

__global__ void p3_ee_traverse(const int* body,const int* boundary,const uint2* edges,
    const AABB* boxes,const Node* nodes,double dHat,int number,uint2* queue,
    uint32_t capacity,P3QueueState* state)
{
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=number)return;
    const int leaf=slot+number-1;const auto self=nodes[leaf].element_idx;
    const auto a=edges[self];const auto box=boxes[leaf];const double gap=sqrt(dHat);
    uint32_t stack[65];uint32_t* top=stack;*top++=0;
    do
    {
        const auto id=*--top;const auto left=nodes[id].left_idx,right=nodes[id].right_idx;
        #pragma unroll
        for(int side=0;side<2;++side)
        {
            const auto child=side?right:left;
            if(!overlap(box,boxes[child],gap))continue;
            const auto other=nodes[child].element_idx;
            if(other==0xffffffffu)
            {
                if(top==stack+65){atomicExch(&state->stack_error,1u);return;}
                *top++=child;continue;
            }
            if(self==other)continue;
            const auto b=edges[other];
            if(!((body[a.x]!=body[b.x])||(body[a.x]==-1)))continue;
            if(a.x==b.x||a.x==b.y||a.y==b.x||a.y==b.y||other<self)continue;
            if(boundary[a.x]>=2&&boundary[a.y]>=2&&boundary[b.x]>=2&&boundary[b.y]>=2)continue;
            const auto index=atomicAdd(&state->count,1ull);
            if(index<capacity)queue[index]=make_uint2(self,other);
        }
    }while(stack<top);
}

__global__ void p3_ee_classify(const uint2* queue,uint32_t count,const uint2* edges,
    const double3* x,const double3* rest,double dHat,int number,int4* pairs,int4* ccd,
    uint32_t* counts,int* matrix,uint32_t capacity)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto ids=queue[i],a=edges[ids.x],b=edges[ids.y];
    _checkEEintersection(x,rest,a.x,a.y,b.x,b.y,ids.y,dHat,counts,matrix,pairs,ccd,number,capacity);
}
