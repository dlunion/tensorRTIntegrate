
#ifndef TRT_BACKEND_HPP
#define TRT_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_infer.hpp"

namespace TRTInfer {

	class Backend {
	public:
		Backend(CUStream stream = nullptr);
		virtual ~Backend();

	protected:
		void* getCPUMemory(size_t size);
		void releaseCPUMemory();

		void* getGPUMemory(size_t size);
		void releaseGPUMemory();

		CUStream getStream() const;

	private:
		void* cpuMemory_ = nullptr;
		size_t cpuMemSize_ = 0;
		void* gpuMemory_ = nullptr;
		size_t gpuMemSize_ = 0;

		CUStream stream_ = nullptr;
		bool ownStream_ = false;
	};
};

#endif  // TRT_BACKEND_HPP