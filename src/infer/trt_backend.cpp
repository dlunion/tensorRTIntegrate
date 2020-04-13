

#include "trt_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

#define cuCheck(op)	 Assert((op) == cudaSuccess)

namespace TRTInfer {

	Backend::Backend(CUStream stream) {

		this->stream_ = stream;
		this->ownStream_ = false;

		if (stream == nullptr) {
			cuCheck(cudaStreamCreate(&stream_));
			ownStream_ = true;
		}
	}

	void* Backend::getCPUMemory(size_t size) {
		if (cpuMemSize_ >= size)
			return cpuMemory_;

		releaseCPUMemory();
		cpuMemSize_ = size;
		cpuMemory_ = malloc(size);
		Assert(cpuMemory_ != nullptr);
		return cpuMemory_;
	}

	CUStream Backend::getStream() const {
		return stream_;
	}

	void Backend::releaseCPUMemory() {
		if (cpuMemory_) {
			free(cpuMemory_);
			cpuMemory_ = nullptr;
			cpuMemSize_ = 0;
		}
	}

	void* Backend::getGPUMemory(size_t size) {
		if (gpuMemSize_ >= size)
			return gpuMemory_;

		releaseGPUMemory();
		gpuMemSize_ = size;

		cuCheck(cudaMalloc(&gpuMemory_, gpuMemSize_));
		return gpuMemory_;
	}

	void Backend::releaseGPUMemory() {
		if (gpuMemory_) {
			cudaFree(gpuMemory_);
			gpuMemory_ = nullptr;
			gpuMemSize_ = 0;
		}
	}

	Backend::~Backend() {
		releaseGPUMemory();
		releaseCPUMemory();

		if (ownStream_) {
			cudaStreamDestroy(stream_);
		}
		stream_ = nullptr;
	}
};