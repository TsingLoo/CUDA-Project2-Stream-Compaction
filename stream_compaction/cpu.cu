#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            int sum = 0;
            for (int i = 0; i < n; ++i)
            {
                odata[i] = sum;
                sum += idata[i];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            scanWithoutTimer(n, odata, idata);

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            int curr = 0;
            for (int i = 0; i < n; ++i)
            {
                curr = idata[i];

                if (curr != 0) {
                    odata[count] = curr;
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            int* tempArray = new int[n];

            for (int i = 0; i < n; ++i)
            {
                tempArray[i] = (idata[i] != 0) ? 1 : 0;
            }

            int* scanResult = new int[n];
            scanWithoutTimer(n, scanResult, tempArray);

            //Scatter
            int finalLength = scanResult[n - 1];

            for (int i = 0; i < n; ++i)
            {
                if (tempArray[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                }
            }


            delete[] tempArray;
            delete[] scanResult;


            timer().endCpuTimer();
            return finalLength;
        }
    }
}
