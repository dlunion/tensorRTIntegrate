

#include "dbface_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	DBFaceBackend::DBFaceBackend(CUStream stream):Backend(stream){}

	static __device__ float commonExp(float value) {

		float gate = 1.0f;
		if (fabs(value) < gate)
			return value * exp(gate);

		if (value > 0)
			return exp(value);
		else
			return -exp(-value);
	}

	static __global__ void DBFaceBackend_forwardGPU(float* hm, float* hmpool, float* tlrb, float* landmark, int* countptr, FaceBox* boxptr,
		int width, int height, int w_x_h, int stride, float threshold,
		int maxobjs, int edge) {

		KERNEL_POSITION;

		float confidence = hm[position];
		if (confidence != hmpool[position] || confidence < threshold)
			return;

		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		int cx = position % width;
		int cy = position / width;
		int oc0 = position;
		int oc1 = position + w_x_h;
		int oc2 = position + w_x_h * 2;
		int oc3 = position + w_x_h * 3;

		FaceBox* ptr = boxptr + index;
		float dx = tlrb[oc0];
		float dy = tlrb[oc1];
		float dr = tlrb[oc2];
		float db = tlrb[oc3];

		ptr->x = (cx - dx) * stride;
		ptr->y = (cy - dy) * stride;
		ptr->r = (cx + dr) * stride;
		ptr->b = (cy + db) * stride;
		ptr->score = confidence;
		ptr->label = 0;

		for (int k = 0; k < 5; ++k) {
			// xxxxx yyyyy
			float landmark_x = landmark[position + w_x_h * k] * 4;
			float landmark_y = landmark[position + w_x_h * (k + 5)] * 4;

			cv::Point2f& point = ptr->landmark[k];
			point.x = (commonExp(landmark_x) + cx) * stride;
			point.y = (commonExp(landmark_y) + cy) * stride;
		}
	}

	const std::vector<std::vector<FaceBox>>& DBFaceBackend::forwardGPU(std::shared_ptr<Tensor> hm, std::shared_ptr<Tensor> hmpool, std::shared_ptr<Tensor> tlrb, 
		std::shared_ptr<Tensor> landmark, float threshold, int maxobjs) {

		int width = hm->width();
		int height = hm->height();
		int batchSize = hm->num();
		int count = hm->count(1);	// c * h * w
		auto grid = gridDims(count);
		auto block = blockDims(count);

		size_t objsStoreSize = maxobjs * sizeof(FaceBox) + sizeof(int);
		int heatmapArea = width * height;
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		int stride = 4;
		auto stream = getStream();

		for (int i = 0; i < batchSize; ++i) {

			float* hm_ptr = hm->gpu<float>(i);
			float* hm_pool_ptr = hmpool->gpu<float>(i);
			float* tlrb_ptr = tlrb->gpu<float>(i);
			float* landmark_ptr = landmark->gpu<float>(i);

			int* countPtr = (int*)gpuPtrInput;
			FaceBox* boxPtr = (FaceBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			DBFaceBackend_forwardGPU <<< grid, block, 0, stream >>> (hm_ptr, hm_pool_ptr, tlrb_ptr, landmark_ptr, countPtr, boxPtr,
				width, height, heatmapArea, stride, threshold, maxobjs, count);

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

			FaceBox* ptr = (FaceBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
		return outputs_;
	}
};