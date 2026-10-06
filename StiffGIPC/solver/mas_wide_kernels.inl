// Derived from frozen v38 typed kernels; upstream MAS MPL provenance retained.
template<class Z>
__global__ void __collectFinalZ_new_wide(double3*                  _Z,
                                    const Z*       d_multiLevelZ,
                                    const __GEIGEN__::itable* _coarseTable,
                                    int*                      _real_map_partId,
                                    int                       levelnum,
                                    int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Z cz;  // = d_multiLevelZ[idx];
    int          rdx            = _real_map_partId[idx];
    cz.x                        = d_multiLevelZ[rdx].x;
    cz.y                        = d_multiLevelZ[rdx].y;
    cz.z                        = d_multiLevelZ[rdx].z;
    __GEIGEN__::itable table    = _coarseTable[idx];
    int*               tablePtr = table.index;
    for(int i = 1; i < levelnum; i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}

template<class S>
__global__ void __buildMultiLevelR_optimized_new_wide(const double3*   _R,
                                                 Eigen::Matrix<S,3,1>* _multiLR,
                                                 int*             _goingNext,
                                                 int*             _prefixOrigin,
                                                 unsigned int* _fineConnectMsk,
                                                 int*          _partId_map_real,
                                                 int           levelNum,
                                                 int           numbers)
{
    int          pdx         = blockIdx.x * blockDim.x + threadIdx.x;
    bool         inRange     = pdx < numbers;
    unsigned int inRangeMask = __ballot_sync(0xffffffffU, inRange);
    if(!inRange)
        return;

    Eigen::Matrix<S,3,1> r;
    int             idx = _partId_map_real[pdx];
    if(idx >= 0)
    {

        r[0] = _R[idx].x;
        r[1] = _R[idx].y;
        r[2] = _R[idx].z;
    }
    else
    {
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
    }

    int laneId      = threadIdx.x % BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int gwarpId     = pdx / BANKSIZE;
    int level       = 0;
    //int rdx         = _real_map_partId[idx];
    _multiLR[pdx] = r;

    __shared__ S c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    __shared__ int prefixSum[DEFAULT_WARPNUM];

    if(laneId == 0)
    {
        prefixSum[localWarpId] = _prefixOrigin[gwarpId];
    }

    __syncwarp(inRangeMask);

    bool useDirectReduction = idx >= 0 && prefixSum[localWarpId] == 1;
    unsigned int directReductionMask = __ballot_sync(inRangeMask, useDirectReduction);
    unsigned int indirectReductionMask =
        __ballot_sync(inRangeMask, idx >= 0 && !useDirectReduction);

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        if(useDirectReduction)
        {
            auto mask_val  = directReductionMask;
            int  warpId    = threadIdx.x & 0x1f;
            bool maskHead  = warpId == 0 || !(mask_val & (1U << (warpId - 1)));
            bool bBoundary = (laneId == 0) || maskHead;

            unsigned int boundaryMask = __ballot_sync(mask_val, bBoundary);
            unsigned int interval = _segmentInterval(mask_val, boundaryMask, warpId);


            for(int iter = 1; iter < BANKSIZE; iter <<= 1)
            {
                S tmpx = __shfl_down_sync(mask_val, r[0], iter);
                S tmpy = __shfl_down_sync(mask_val, r[1], iter);
                S tmpz = __shfl_down_sync(mask_val, r[2], iter);
                if(interval >= iter)
                {
                    r[0] += tmpx;
                    r[1] += tmpy;
                    r[2] += tmpz;
                }
            }
            //int level = 0;

            if(bBoundary)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd(&(_multiLR[idx][0]), r[0]);
                    atomicAdd(&(_multiLR[idx][1]), r[1]);
                    atomicAdd(&(_multiLR[idx][2]), r[2]);
                }
            }
            return;
        }
        else
        {
            int elected_lane = __ffs(connectMsk) - 1;

            c_sumResidual[threadIdx.x]                         = 0;
            c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = 0;
            c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
            __syncwarp(indirectReductionMask);
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane, r[0]);
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE,
                      r[1]);
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                      r[2]);
            __syncwarp(indirectReductionMask);

            unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
            if(electedPrefix == 0)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd(&(_multiLR[idx][0]), c_sumResidual[threadIdx.x]);
                    atomicAdd(&(_multiLR[idx][1]),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                    atomicAdd(&(_multiLR[idx][2]),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
                }
            }
        }
    }
}

template<class Mat,class R,class C,class Z>
__global__ void _schwarzLocalXSym6_wide(const Mat* Pred,
                                   const Eigen::Matrix<R,3,1>*           mR,
                                   Z*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * BANKSIZE);

    int Hid   = idx / hessianSize;
    int lvrid = (idx % hessianSize) / (BANKSIZE);
    int lvcid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + lvrid;
    int vcid = Hid * BANKSIZE + lvcid;

    Eigen::Matrix<C,3,1> rdata;
    //rdata.setZero();

    __shared__ Eigen::Matrix<C,3,1> smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid].template cast<C>();
    }
    __syncthreads();

    if(vcid >= vrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index].template cast<C>() * smR[lvcid];
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].template cast<C>().transpose() * smR[lvcid];
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        C tmpx = __shfl_down_sync(0xffffffff, rdata[0], iter);
        C tmpy = __shfl_down_sync(0xffffffff, rdata[1], iter);
        C tmpz = __shfl_down_sync(0xffffffff, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x)), rdata[0]);
        atomicAdd((&(mZ[vrid].y)), rdata[1]);
        atomicAdd((&(mZ[vrid].z)), rdata[2]);
    }
}