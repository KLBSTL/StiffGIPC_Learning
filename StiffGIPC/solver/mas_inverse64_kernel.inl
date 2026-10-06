// Frozen v40 typed inverse; upstream MAS MPL provenance retained.
template<class Mat>
__global__ void __inverse6_P96x96_typed(Mat* _preMatrix,
                                  __GEIGEN__::MasMatrixSymT* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId       = idx / (BANKSIZE * 3);
    int i           = idx % (BANKSIZE * 3);
    int block_matId = threadIdx.x / (BANKSIZE * 3);

    __shared__ double sPMas[32 / BANKSIZE][BANKSIZE * 3][BANKSIZE * 3];
    __shared__ double colm[32 / BANKSIZE][BANKSIZE * 3];

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](j % 3, i % 3);
        }
        else
        {
            index = BANKSIZE * colId - colId * (colId + 1) / 2 + rowId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](i % 3, j % 3);
        }
        if(i == j)
        {
            if(sPMas[block_matId][j][i] == 0)
            {
                sPMas[block_matId][j][i] = 1;
            }
        }
    }

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();

        rt = sPMas[block_matId][j][j];

        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        if(i == j)
        {

            sPMas[block_matId][i][j] = 1;
        }
        else
        {
            sPMas[block_matId][i][j] = 0;
        }
        __syncthreads();
        sPMas[block_matId][j][i] /= rt;

        __syncthreads();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }

        j++;
    }
    __syncthreads();
    if(i % 3 < 2)
        sPMas[block_matId][i + 1][i] = sPMas[block_matId][i][i + 1];
    else
        sPMas[block_matId][i][i - 2] = sPMas[block_matId][i - 2][i];
    __syncthreads();
    //__threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            _preMatrix[matId].M[index](j % 3, i % 3) = sPMas[block_matId][j][i];
        }
    }
}
