#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernParallelScan(int n, int pow2d, int *odata, const int *idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k >= n) {
                return;
            }

            if (k >= pow2d){ 
                odata[k] = idata[k - pow2d] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }  
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_A;
            int* dev_B;

			cudaMalloc((void**)&dev_A, n * sizeof(int));
            cudaMalloc((void**)&dev_B, n * sizeof(int));


			cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d < ilog2ceil(n); ++d)
            {
				kernParallelScan <<<fullBlocksPerGrid, BLOCK_SIZE >>> (n, pow(2, d), dev_B, dev_A);
                std::swap(dev_A, dev_B);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_A, (n - 1) * sizeof(int), cudaMemcpyDefault);

			cudaFree(dev_A);
			cudaFree(dev_B);

        }
    }
}
