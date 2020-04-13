
#ifndef DBFACE_BACKEND_HPP
#define DBFACE_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

namespace TRTInfer {

	struct FaceBox : ccutil::BBox {
		cv::Point2f landmark[5];

		FaceBox() {}
		FaceBox(const ccutil::BBox& other):ccutil::BBox(other) {}
	};

	class DBFaceBackend : public Backend{
	public:
		DBFaceBackend(CUStream stream = nullptr);

		const std::vector<std::vector<FaceBox>>& forwardGPU(
			std::shared_ptr<Tensor> hm, std::shared_ptr<Tensor> hmpool, std::shared_ptr<Tensor> tlrb, std::shared_ptr<Tensor> landmark,
			float threshold, int maxobjs = 100);

	private:
		std::vector<std::vector<FaceBox>> outputs_;
	};
};

#endif  // DBFACE_BACKEND_HPP