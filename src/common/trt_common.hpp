
#ifndef TRT_COMMON_HPP
#define TRT_COMMON_HPP

#include <cuda_runtime.h>

#define GPU_BLOCK_THREADS  512
#define KERNEL_POSITION											\
	int position = (blockDim.x * blockIdx.x + threadIdx.x);		\
	if (position >= (edge)) return;

dim3 gridDims(int numJobs);
dim3 blockDims(int numJobs);

#endif //TRT_COMMON_HPP