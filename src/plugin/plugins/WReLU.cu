

#include "WReLU.hpp"

using namespace Plugin;

__global__ void relu(float* input, float* output, int edge) {

	KERNEL_POSITION;
	output[position] = (input[position] < 0 ? 0 : input[position]) + 1.3;
}

int WReLU::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = gridDims(count);
	auto block = blockDims(count);

	float* ptr = inputs[0].ptr();
	float* output = outputs[0].ptr();
	relu <<<grid, block>>> (ptr, output, count);
	return 0;
}

RegisterPlugin(WReLU);