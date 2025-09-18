#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256

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

            idata[k - halfStride] = parentValue;
            idata[k] = parentValue + originalLeftChildValue;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            // k is the ID of the k-th active thread (dense)
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int fullStride = 1 << (d + 1);
            int halfStride = 1 << d;

            int global_idx = (k + 1) * fullStride - 1;

            if (global_idx >= n) {
                return;
            }

            data[global_idx] += data[global_idx - halfStride];
        }

        __global__ void kernUpSweep_DEBUG(int n, int d, int* data, int numActiveThreads) {
            // k is the ID of the k-th active thread (dense)
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            // This check is important: only let threads that are part of the
            // compact grid do the printing.
            if (k >= numActiveThreads) {
                return;
            }

            int fullStride = 1 << (d + 1);
            int halfStride = 1 << d;

            int global_idx = (k + 1) * fullStride - 1;

            // --- Let's print the state of the first and last active threads ---
            if (d == 0 && (k == 0 || k == numActiveThreads - 1)) {
                printf("d=%d, active_thread_k=%d, global_idx=%d, n=%d\n",
                    d, k, global_idx, n);
            }

            if (global_idx >= n) {
                return;
            }

            // Let's also check the values we are about to add
            if (d == 0 && (k == 0 || k == numActiveThreads - 1)) {
                int read_addr = global_idx - halfStride;
                printf("  -> Thread %d will add data[%d] (value=%d) to data[%d] (value=%d)\n",
                    k, read_addr, data[read_addr], global_idx, data[global_idx]);
            }

            data[global_idx] += data[global_idx - halfStride];
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int fullStride = 1 << (d + 1);
            int halfStride = 1 << d;

            int global_idx = (k + 1) * fullStride - 1;

            if (global_idx >= n) {
                return;
            }

            // Standard down-sweep logic using the calculated global index
            int originalLeftChildValue = data[global_idx - halfStride];
            int parentValue = data[global_idx];


            data[global_idx - halfStride] = parentValue;
            data[global_idx] = parentValue + originalLeftChildValue;
        }

        /// <summary>
        /// return an excluisve scan
        /// </summary>
        /// <param name="n">is the actual number of elements</param>
        /// <param name="dev_odata">be sure that the length of this should be n</param>
        /// <param name="dev_idata">be sure that the length of this should be n</param>
        void scan_device(int n, int* dev_odata, const int* dev_idata) {
            int paddedN = 1 << ilog2ceil(n);

            int* dev_temp;

            cudaMalloc((void**)&dev_temp, paddedN * sizeof(int));
            cudaMemset(dev_temp, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_temp, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);


            for (int d = 0; d < ilog2ceil(paddedN); d++) {
                // Calculate how many threads are actually needed for this pass
                int numActiveThreads = paddedN / (1 << (d + 1));
                int blockSize = std::min(BLOCK_SIZE, numActiveThreads);


                dim3 gridDim((numActiveThreads + blockSize - 1) / blockSize);
                dim3 blockDim(blockSize);

                kernUpSweep << <gridDim, blockDim >> > (paddedN, d, dev_temp);

                cudaDeviceSynchronize();
            }

            setLastElementToZero << <1, 1 >> > (paddedN, dev_temp);

            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numActiveThreads = paddedN / (1 << (d + 1));
                int blockSize = std::min(BLOCK_SIZE, numActiveThreads);


                dim3 gridDim((numActiveThreads + blockSize - 1) / blockSize);
                dim3 blockDim(blockSize);

                kernDownSweep << <gridDim, blockDim >> > (paddedN, d, dev_temp);

                cudaDeviceSynchronize();
            }

            cudaMemcpy(dev_odata, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(dev_temp);
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            scan_device(n, dev_odata, dev_idata);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int paddedN = 1 << ilog2ceil(n);

            //
			int t = ilog2ceil(9);

            dim3 fullBlocksPerGrid((paddedN + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_temp; // Use a temporary buffer for the in-place algorithm
            int* dev_odata;
            cudaMalloc((void**)&dev_temp, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_odata, paddedN * sizeof(int));
            cudaMemset(dev_temp, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            scan_device(paddedN, dev_odata, dev_temp);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_temp);
            cudaFree(dev_odata);
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

            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_idata;
            int* dev_flags;
            int* dev_scanResult;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_flags, n * sizeof(int));
            cudaMalloc((void**)&dev_scanResult, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            //compute the temporary array
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_flags, dev_idata);

			//compute the scan of the temporary array
            int paddedN = 1 << ilog2ceil(n);
            scan_device(n, dev_scanResult, dev_flags);

			//compute the scatter
            int lastElementOfScan = 0;
            int lastElementOfFlags = 0;
            cudaMemcpy(&lastElementOfScan, &dev_scanResult[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastElementOfFlags, &dev_flags[n - 1], sizeof(int), cudaMemcpyDeviceToHost);


            int totalCount = lastElementOfScan + lastElementOfFlags;
            cudaMalloc((void**)&dev_odata, totalCount * sizeof(int));
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_odata, dev_idata, dev_flags, dev_scanResult);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, totalCount * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_flags);
            cudaFree(dev_scanResult);
            cudaFree(dev_odata);

            return totalCount;
        }
    }
}
