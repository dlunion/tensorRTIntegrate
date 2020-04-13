

#include "ct_detect_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	CTDetectBackend::CTDetectBackend(CUStream stream) :Backend(stream){}

	static __global__ void CTDetectBackend_forwardGPU(float* hm, float* hmpool, float* wh, float* reg, int* countptr, ccutil::BBox* boxptr, int width, int height, int w_x_h, int channels, int stride, float threshold,
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

	const std::vector<std::vector<ccutil::BBox>>& CTDetectBackend::forwardGPU(std::shared_ptr<Tensor> hm, std::shared_ptr<Tensor> hmpool, std::shared_ptr<Tensor> wh, 
		std::shared_ptr<Tensor> reg, const std::vector<cv::Size>& imageSize, float threshold, int maxobjs) {

		int count = hm->count(1);  // w * h * c
		int width = hm->width();
		int height = hm->height();
		int batchSize = hm->num();
		int channels = hm->channel();
		int stride = 4;
		auto grid = gridDims(count);
		auto block = blockDims(count);

		size_t objsStoreSize = maxobjs * sizeof(ccutil::BBox) + sizeof(int);
		int heatmapArea = width * height;
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();

		for (int i = 0; i < batchSize; ++i) {

			auto& imsize = imageSize[i];
			float sw = width * stride / (float)imsize.width;
			float sh = height * stride / (float)imsize.height;
			float scale = std::min(sw, sh);

			float* hm_ptr = hm->gpu<float>(i);
			float* hm_pool_ptr = hmpool->gpu<float>(i);
			float* wh_ptr = wh->gpu<float>(i);
			float* reg_ptr = reg->gpu<float>(i);

			int* countPtr = (int*)gpuPtrInput;
			ccutil::BBox* boxPtr = (ccutil::BBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			CTDetectBackend_forwardGPU <<< grid, block, 0, stream >>> (hm_ptr, hm_pool_ptr, wh_ptr, reg_ptr, countPtr, boxPtr,
				width, height, heatmapArea, channels, stride, threshold, maxobjs, imsize.width, imsize.height, scale, count);

			cudaMemcpyAsync(cpuPtrInput, gpuPtrInput, objsStoreSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
			
			cpuPtrInput += objsStoreSize;
			gpuPtrInput += objsStoreSize;
		}
		cudaStreamSynchronize(stream);

		cpuPtrInput = (char*)cpuPtr;
		outputs_.resize(batchSize);

		for (int i = 0; i < batchSize; ++i, cpuPtrInput += objsStoreSize) {
			auto& output = outputs_[i];
			output.clear();

			int num = *((int*)cpuPtrInput);
			num = std::min(num, maxobjs);

			if (num == 0) 
				continue;

			ccutil::BBox* ptr = (ccutil::BBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
		return outputs_;
	}
};