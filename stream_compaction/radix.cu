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

        __global__ void kernRadixComputeE(int n, int d, int* eArr, const int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n) return;

            eArr[k] = ((data[k] >> d) & 1) == 0 ? 1 : 0;
        }

        __global__ void kernRadixScatter(int n, int d, int totalFalses, int* odata, const int* idata, const int* fArr) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n) return;

            int element = idata[k];
            int bit_is_false = ((element >> d) & 1) == 0; // 1 if bit is 0, 0 if bit is 1

            int new_idx;
            if (bit_is_false) {
                // If the bit is 0 (false), its new position is given by the
                // exclusive scan result 'fArr[k]'. This corresponds to f[i] in your image.
                new_idx = fArr[k];
            }
            else {
                // If the bit is 1 (true), this calculates its destination address.
                // This corresponds to t[i] = i - f[i] + totalFalses from your image.
                int ones_offset = k - fArr[k];
                new_idx = totalFalses + ones_offset;
            }

            odata[new_idx] = element;
        }

        void sort(int n, int *odata, const int *idata) {

			dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_bufferA;
            int* dev_bufferB; 
            int* dev_eArr;         
            int* dev_fArr;         
            
            cudaMalloc((void**)&dev_bufferA, n * sizeof(int));
            cudaMalloc((void**)&dev_bufferB, n * sizeof(int));
            cudaMalloc((void**)&dev_eArr, n * sizeof(int));
            cudaMalloc((void**)&dev_fArr, n * sizeof(int));


            cudaMemcpy(dev_bufferA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_input = dev_bufferA;
            int* dev_output = dev_bufferB;

            timer().startGpuTimer();

            for (int d = 0; d < sizeof(int) * 8; ++d) {
                // e array
                kernRadixComputeE << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, dev_eArr, dev_input);

				//  exclusive scan to get f array
                StreamCompaction::Efficient::scan_device(n, dev_fArr, dev_eArr);

                // total count of zero
                int totalFalses = 0;
                if (n > 0) {
                    int last_idx, last_flag;
                    cudaMemcpy(&last_idx, &dev_fArr[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&last_flag, &dev_eArr[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
                    totalFalses = last_idx + last_flag;
                }

                kernRadixScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, d, totalFalses, dev_output, dev_input, dev_fArr);

                std::swap(dev_input, dev_output);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_input, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
            cudaFree(dev_eArr);
            cudaFree(dev_fArr);
        }
    }
}
