#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void sort(int n, int *odata, const int *idata) {

			dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_A;
            int* dev_B;

			cudaMalloc((void**)&dev_A, n * sizeof(int));
            cudaMalloc((void**)&dev_B, n * sizeof(int));


			cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d < ilog2ceil(n); ++d)
            {
				//kernParallelScan <<<fullBlocksPerGrid, BLOCK_SIZE >>> (n, pow(2, d), dev_B, dev_A);
                std::swap(dev_A, dev_B);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_A, (n - 1) * sizeof(int), cudaMemcpyDefault);

			cudaFree(dev_A);
			cudaFree(dev_B);

        }
    }
}
