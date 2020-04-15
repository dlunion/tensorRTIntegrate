
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include "infer/dbface_backend.hpp"

using namespace cv;
using namespace std;

namespace examples {

	static float commonExp(float value) {

		float gate = 1;
		float base = exp(gate);
		if (fabs(value) < gate)
			return value * base;

		if (value > 0) {
			return exp(value);
		}
		else {
			return -exp(-value);
		}
	}

	static vector<TRTInfer::FaceBox> detectDBFace(const shared_ptr<TRTInfer::Engine>& dbfaceDetect_, const Mat& image, float threshold = 0.3) {

		Assert(image.cols % 32 == 0 || image.rows % 32 == 0);

		float mean[3] = {0.408, 0.447, 0.47};
		float std[3] = {0.289, 0.274, 0.278};

		//dbfaceDetect_->input()->setNormMat(0, image, mean, std);  // 20 ms
		dbfaceDetect_->input()->setNormMatGPU(0, image, mean, std);		// 5 ms
		dbfaceDetect_->forward();
		auto outHM = dbfaceDetect_->tensor("hm");
		auto outHMPool = dbfaceDetect_->tensor("pool_hm");
		auto outTLRB = dbfaceDetect_->tensor("tlrb");
		auto outLandmark = dbfaceDetect_->tensor("landmark");
		const int stride = 4;

		vector<TRTInfer::FaceBox> bboxs;
		Size inputSize = dbfaceDetect_->input()->size();
		float sx = image.cols / (float)inputSize.width * stride;
		float sy = image.rows / (float)inputSize.height * stride;

		for (int class_ = 0; class_ < outHM->channel(); ++class_) {
			for (int i = 0; i < outHM->height(); ++i) {
				float* ohmptr = outHM->cpu<float>(0, class_, i);
				float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
				for (int j = 0; j < outHM->width(); ++j) {
					if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {

						float dx = outTLRB->at<float>(0, 0, i, j);
						float dy = outTLRB->at<float>(0, 1, i, j);
						float dr = outTLRB->at<float>(0, 2, i, j);
						float db = outTLRB->at<float>(0, 3, i, j);
						float cx = j;
						float cy = i;
						float x = (cx - dx) * stride;
						float y = (cy - dy) * stride;
						float r = (cx + dr) * stride;
						float b = (cy + db) * stride;

						TRTInfer::FaceBox box(ccutil::BBox(x, y, r, b, *ohmptr, class_));
						if (box.area() > 0) {

							for (int k = 0; k < 5; ++k) {
								float landmark_x = outLandmark->at<float>(0, k, i, j) * 4;
								float landmark_y = outLandmark->at<float>(0, k + 5, i, j) * 4;
								landmark_x = (commonExp(landmark_x) + cx) * stride;
								landmark_y = (commonExp(landmark_y) + cy) * stride;
								box.landmark[k] = Point2f(landmark_x, landmark_y);
							}
							bboxs.push_back(box);
						}
					}
					++ohmptr;
					++ohmpoolptr;
				}
			}
		}
		return bboxs;
	}

	static vector<TRTInfer::FaceBox> detectDBFaceOptim(const shared_ptr<TRTInfer::Engine>& dbfaceDetect_, const Mat& image, float threshold, TRTInfer::DBFaceBackend* backend) {

		Assert(image.cols % 32 == 0 || image.rows % 32 == 0);

		float mean[3] = {0.408, 0.447, 0.47};
		float std[3] = {0.289, 0.274, 0.278};

		dbfaceDetect_->input()->setNormMatGPU(0, image, mean, std);		// 5 ms
		dbfaceDetect_->forward(false);
		auto outHM = dbfaceDetect_->tensor("hm");
		auto outHMPool = dbfaceDetect_->tensor("pool_hm");
		auto outTLRB = dbfaceDetect_->tensor("tlrb");
		auto outLandmark = dbfaceDetect_->tensor("landmark");
		const int stride = 4;
		return backend->forwardGPU(outHM, outHMPool, outTLRB, outLandmark, threshold, 1000)[0]; // 0.25 ms
	}

	static Mat padImage(const Mat& image, int stride = 32) {

		int w = image.cols;
		if (image.cols % stride != 0)
			w = image.cols + (stride - (image.cols % stride));

		int h = image.rows;
		if (image.rows % stride != 0)
			h = image.rows + (stride - (image.rows % stride));

		if (Size(w, h) == image.size())
			return image;

		Mat output(h, w, image.type(), Scalar(0));
		image.copyTo(output(Rect(0, 0, image.cols, image.rows)));
		return output;
	}

	void dbface() {

		Mat image = imread("imgs/selfie.jpg");
		if (image.empty()) {
			INFOW("image load fail");
			return;
		}

		Mat padimage = padImage(image);
		int maxBatchSize = 1;
		string modelPath = ccutil::format("models/dbface.%dx%d.fp32.b%d.trtmodel", padimage.cols, padimage.rows, maxBatchSize);

		if (!ccutil::exists(modelPath)) {

			if (!ccutil::exists("models/dbface.onnx")) {
				INFOW(
					"models/dbface.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dbface.onnx"
				);
				return;
			}

			TRTBuilder::compileTRT(
				TRTBuilder::TRTMode_FP32, {}, maxBatchSize,
				TRTBuilder::ModelSource("models/dbface.onnx"),
				modelPath, 
				{TRTBuilder::InputDims(3, padimage.rows, padimage.cols)}
			);
		}

		INFO("load model: %s", modelPath.c_str());
		auto engine = TRTInfer::loadEngine(modelPath);
		if (!engine) {
			INFO("can not load model: %s", modelPath.c_str());
			return;
		}

		INFO("forward...");
		TRTInfer::DBFaceBackend backend(engine->getCUStream());
		auto objs = detectDBFaceOptim(engine, image, 0.25, &backend);
		
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(image, obj, ccutil::DrawType::Empty);

			for (int k = 0; k < 5; ++k) {
				cv::circle(image, obj.landmark[k], 3, Scalar(0, 0, 255), -1, 16);
			}
		}

		imwrite("results/selfie.draw.jpg", image);

#ifdef _WIN32
		cv::imshow("dbface selfie detect", image);
		cv::waitKey();
		cv::destroyAllWindows();
#endif
		INFO("done.");
	}
};