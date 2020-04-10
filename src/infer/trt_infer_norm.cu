

#include "trt_infer.hpp"
#include <common/trt_common.hpp>

namespace TRTInfer {
	static __global__ void ImageNormMeanStd_forwardGPU_impl(float* d0, float* d1, float* d2, float m0, float m1, float m2, float std0, float std1, float std2, unsigned char* src, int edge) {

		KERNEL_POSITION;

		float* pptr_dest[3] = {d0, d1, d2};
		float mean[3] = {m0, m1, m2};
		float std[3] = {std0, std1, std2};

		int pos_real = position * 3;
		unsigned char* src_real = src + pos_real;
		for (int c = 0; c < 3; ++c) {
			unsigned char value = src_real[c];
			float* dest_ptr = pptr_dest[c];
			dest_ptr[position] = (value / 255.0f - mean[c]) / std[c];
		}
	}

	void ImageNormMeanStd_forwardGPU(float* d0, float* d1, float* d2, float mean[3], float std[3], unsigned char* src, int nump) {

		auto grid = gridDims(nump);
		auto block = blockDims(nump);
		ImageNormMeanStd_forwardGPU_impl << <grid, block >> >(d0, d1, d2, mean[0], mean[1], mean[2], std[0], std[1], std[2], src, nump);
	}
};