
#ifndef CT_DETECT_BACKEND_HPP
#define CT_DETECT_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

namespace TRTInfer {

	class CTDetectBackend : public Backend{
	public:
		CTDetectBackend(CUStream stream = nullptr);

		const std::vector<std::vector<ccutil::BBox>>& forwardGPU(std::shared_ptr<Tensor> hm, std::shared_ptr<Tensor> hmpool, std::shared_ptr<Tensor> wh, std::shared_ptr<Tensor> reg,
			const std::vector<cv::Size>& imageSize, float threshold, int maxobjs);

	private:
		std::vector<std::vector<ccutil::BBox>> outputs_;
	};
};

#endif  // CT_DETECT_BACKEND_HPP