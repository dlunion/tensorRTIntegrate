
#ifndef CT_DETECT_BACKEND_HPP
#define CT_DETECT_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace TRTInfer {

	class CTDetectBackend {
	public:
		CTDetectBackend(int width, int height, int channels, int stride, float threshold, int maxobjs = 100);
		virtual ~CTDetectBackend();

		std::vector<ccutil::BBox> forwardGPU(float* hm, float* hmpool, float* wh, float* reg, int imageWidth, int imageHeight);

	private:
		void* cpuMemory_ = nullptr;
		void* gpuMemory_ = nullptr;
		size_t count_;
		int width_, height_, channels_, stride_, maxobjs_;
		float threshold_;
		size_t memSize_;
	};
};

#endif  // CT_DETECT_BACKEND_HPP