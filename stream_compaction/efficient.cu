#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernWorkEfficientUpSweep(int paddedN, int n, int d, int* idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

			int halfStride = 1 << (d);
            int fullStride = halfStride * 2;

            if (k >= n || ((k + 1) % fullStride != 0)) {
                return;
            }

            // up-sweep
            idata[k] += idata[k - halfStride];
        }

        __global__ void setLastElementToZero(int paddedN, int* idata)
        {
            idata[paddedN - 1] = 0;
        }

        __global__ void kernWorkEfficientDownSweep(int paddedN, int n, int d, int* idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int halfStride = 1 << (d);
            int fullStride = halfStride * 2;

            if (k >= paddedN || ((k + 1) % fullStride != 0)) {
                return;
            }

            // down-sweep
			int originalLeftChildValue = idata[k - halfStride];
			int parentValue = idata[k];

			idata[k - halfStride] = parentValue ;
			idata[k] = parentValue + originalLeftChildValue;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int paddedN = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid((paddedN + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            int* dev_idata; 

			cudaMalloc((void**)&dev_idata, paddedN * sizeof(int));
			cudaMemset(dev_idata, 0, paddedN * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
                kernWorkEfficientUpSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (paddedN, n, d, dev_idata);
            }

            setLastElementToZero << <1, 1 >> > (paddedN, dev_idata);

            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernWorkEfficientDownSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (paddedN, n, d, dev_idata);
            }

			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
			cudaFree(dev_idata);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
