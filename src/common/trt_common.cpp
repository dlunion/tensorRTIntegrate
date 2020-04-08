

#include "trt_common.hpp"
#include <math.h>

dim3 gridDims(int numJobs) {
	int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
	return dim3(ceil(numJobs / (float)numBlockThreads));
}

dim3 blockDims(int numJobs) {
	return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}