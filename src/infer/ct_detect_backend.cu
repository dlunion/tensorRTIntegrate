

#include "ct_detect_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	CTDetectBackend::CTDetectBackend(int width, int height, int channels, int stride, float threshold, int maxobjs) {

		this->count_ = 1 * channels * width * height;
		this->width_ = width;
		this->height_ = height;
		this->channels_ = channels;
		this->stride_ = stride;
		this->threshold_ = threshold;
		this->maxobjs_ = maxobjs;

		this->memSize_ = maxobjs * sizeof(ccutil::BBox) + sizeof(int);  // num
		auto code = cudaMalloc(&gpuMemory_, this->memSize_);
		Assert(code == cudaSuccess);

		cpuMemory_ = malloc(this->memSize_);
	}

	CTDetectBackend::~CTDetectBackend() {
		if (gpuMemory_) {
			cudaFree(gpuMemory_);
			gpuMemory_ = nullptr;
		}

		if (cpuMemory_) {
			free(cpuMemory_);
			cpuMemory_ = nullptr;
		}
	}

	__global__ void CTDetectBackend_forwardGPU(float* hm, float* hmpool, float* wh, float* reg, int* countptr, ccutil::BBox* boxptr, int width, int height, int w_x_h, int channels, int stride, float threshold,
		int maxobjs, int imageWidth, int imageHeight, float scale, int edge) {

		KERNEL_POSITION;

		float confidence = hm[position];
		if (confidence != hmpool[position] || confidence < threshold)
			return;

		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		int channel_index = position / w_x_h;
		int classes = channel_index;
		int offsetChannel0 = position - channel_index * w_x_h;
		int offsetChannel1 = offsetChannel0 + w_x_h;

		int cx = offsetChannel0 % width;
		int cy = offsetChannel0 / width;

		ccutil::BBox* ptr = boxptr + index;
		float dx = reg[offsetChannel0];
		float dy = reg[offsetChannel1];
		float dw = wh[offsetChannel0];
		float dh = wh[offsetChannel1];

		ptr->x = ((cx + dx - dw * 0.5 - width * 0.5) * stride) / scale + imageWidth * 0.5;
		ptr->y = ((cy + dy - dh * 0.5 - height * 0.5) * stride) / scale + imageHeight * 0.5;
		ptr->r = ((cx + dx + dw * 0.5 - width * 0.5) * stride) / scale + imageWidth * 0.5;
		ptr->b = ((cy + dy + dh * 0.5 - height * 0.5) * stride) / scale + imageHeight * 0.5;
		ptr->score = confidence;
		ptr->label = classes;
	}

	std::vector<ccutil::BBox> CTDetectBackend::forwardGPU(float* hm, float* hmpool, float* wh, float* reg, int imageWidth, int imageHeight) {

		auto grid = gridDims(this->count_);
		auto block = blockDims(this->count_);

		float scale = 0;
		if (imageWidth >= imageHeight)
			scale = this->width_ * this->stride_ / (float)imageWidth;
		else
			scale = this->height_ * this->stride_ / (float)imageHeight;

		int* countptr = (int*)gpuMemory_;
		ccutil::BBox* boxptr = (ccutil::BBox*)((char*)gpuMemory_ + sizeof(int));

		// clean
		cudaMemset(gpuMemory_, 0, memSize_);
		CTDetectBackend_forwardGPU <<< grid, block >>> (hm, hmpool, wh, reg, countptr, boxptr,
			width_, height_, width_ * height_, channels_, stride_, threshold_, maxobjs_, imageWidth, imageHeight, scale, count_);

		cudaMemcpy(cpuMemory_, gpuMemory_, this->memSize_, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		int num = *((int*)cpuMemory_);
		num = std::min(num, this->maxobjs_);

		if (num == 0)
			return std::vector<ccutil::BBox>();

		ccutil::BBox* ptr = (ccutil::BBox*)((char*)cpuMemory_ + sizeof(int));
		return std::vector<ccutil::BBox>(ptr, ptr + num);
	}
};