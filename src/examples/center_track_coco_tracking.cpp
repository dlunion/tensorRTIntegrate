
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

namespace examples {

	static void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
		cv::Scalar color, int thickness, int lineType)
	{
		const double PI = 3.1415926;
		Point arrow;
		double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
		line(img, pStart, pEnd, color, thickness, lineType);

		arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
		arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
		line(img, pEnd, arrow, color, thickness, lineType);
		arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
		arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
		line(img, pEnd, arrow, color, thickness, lineType);
	}

	static Rect restoreCenterNetBox(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, Size netSize, Size imageSize) {

		float scale = 0;
		if (imageSize.width >= imageSize.height)
			scale = netSize.width / (float)imageSize.width;
		else
			scale = netSize.height / (float)imageSize.height;

		float x = ((cellx + dx - dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
		float y = ((celly + dy - dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
		float r = ((cellx + dx + dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
		float b = ((celly + dy + dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
		return Rect(Point(x, y), Point(r + 1, b + 1));
	}

	static Scalar restoreCenterTracking(float ox, float oy, float cellx, float celly, int stride, Size netSize, Size imageSize) {

		float scale = 0;
		if (imageSize.width >= imageSize.height)
			scale = netSize.width / (float)imageSize.width;
		else
			scale = netSize.height / (float)imageSize.height;

		float x = ((cellx + ox) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
		float y = ((celly + oy) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
		float x0 = ((cellx)* stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
		float y0 = ((celly)* stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
		return Scalar(x0, y0, x, y);
	}

	static void preprocessCenterNetImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

		int outH = tensor->height();
		int outW = tensor->width();
		float sw = outW / (float)image.cols;
		float sh = outH / (float)image.rows;
		float scale = std::min(sw, sh);

		Mat matrix = getRotationMatrix2D(Point2f(image.cols*0.5, image.rows*0.5), 0, scale);
		matrix.at<double>(0, 2) -= image.cols*0.5 - outW * 0.5;
		matrix.at<double>(1, 2) -= image.rows*0.5 - outH * 0.5;

		float mean[3] = {0.40789654, 0.44719302, 0.47026115};
		float std[3] = {0.28863828, 0.27408164, 0.27809835};

		Mat outimage;
		cv::warpAffine(image, outimage, matrix, Size(outW, outH));
		tensor->setNormMatGPU(numIndex, outimage, mean, std);
	}

	static vector<tuple<ccutil::BBox, Scalar>> detectBoundingboxAndTracking(const shared_ptr<TRTInfer::Engine>& boundingboxAndTrackingDetect_, const Mat& image, const Mat& prevImage, float threshold = 0.3) {

		if (boundingboxAndTrackingDetect_ == nullptr) {
			INFO("detectBoundingbox failure call, model is nullptr");
			return vector<tuple<ccutil::BBox, Scalar>>();
		}

		preprocessCenterNetImageToTensor(image, 0, boundingboxAndTrackingDetect_->input(0));
		preprocessCenterNetImageToTensor(prevImage, 0, boundingboxAndTrackingDetect_->input(1));
		boundingboxAndTrackingDetect_->forward();
		auto outHM = boundingboxAndTrackingDetect_->tensor("hm");
		auto outHMPool = boundingboxAndTrackingDetect_->tensor("hm_pool");
		auto outWH = boundingboxAndTrackingDetect_->tensor("wh");
		auto outXY = boundingboxAndTrackingDetect_->tensor("reg");
		auto outTracking = boundingboxAndTrackingDetect_->tensor("tracking");
		const int stride = 4;

		vector<tuple<ccutil::BBox, Scalar>> bboxs;
		Size inputSize = boundingboxAndTrackingDetect_->input()->size();
		float sx = image.cols / (float)inputSize.width * stride;
		float sy = image.rows / (float)inputSize.height * stride;

		for (int class_ = 0; class_ < outHM->channel(); ++class_) {
			for (int i = 0; i < outHM->height(); ++i) {
				float* ohmptr = outHM->cpu<float>(0, class_, i);
				float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
				for (int j = 0; j < outHM->width(); ++j) {
					if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {

						float dx = outXY->at<float>(0, 0, i, j);
						float dy = outXY->at<float>(0, 1, i, j);
						float dw = outWH->at<float>(0, 0, i, j);
						float dh = outWH->at<float>(0, 1, i, j);
						float ox = outTracking->at<float>(0, 0, i, j);
						float oy = outTracking->at<float>(0, 1, i, j);
						ccutil::BBox box = restoreCenterNetBox(dx, dy, dw, dh, j, i, stride, inputSize, image.size());
						auto offset = restoreCenterTracking(ox, oy, j, i, stride, inputSize, image.size());
						box = box.box() & Rect(0, 0, image.cols, image.rows);
						box.label = class_;
						box.score = *ohmptr;

						if (box.area() > 0)
							bboxs.push_back(make_tuple(box, offset));
					}
					++ohmptr;
					++ohmpoolptr;
				}
			}
		}
		return bboxs;
	}

	void center_track_coco_tracking() {
		INFOW("onnx to trtmodel...");

		if (!ccutil::exists("models/coco_tracking.fp32.trtmodel")) {

			if (!ccutil::exists("models/coco_tracking.onnx")) {

				INFOW(
					"models/coco_tracking.onnx not found, download url: http://zifuture.com:1000/fs/public_models/coco_tracking.onnx"
				);
				return;
			}

			TRTBuilder::compileTRT(
				TRTBuilder::TRTMode_FP32, {}, 1,
				TRTBuilder::ModelSource("models/coco_tracking.onnx"),
				"models/coco_tracking.fp32.trtmodel", 
				{TRTBuilder::InputDims(3, 512, 512), TRTBuilder::InputDims(3, 512, 512)}
			);
		}

		INFO("load model: models/coco_tracking.fp32.trtmodel");
		auto engine = TRTInfer::loadEngine("models/coco_tracking.fp32.trtmodel");
		if (!engine) {
			INFO("can not load model.");
			return;
		}

		INFO("forward...");
		Mat prevImage = imread("imgs/000020.jpg");
		Mat image = imread("imgs/000023.jpg");

		auto objs = detectBoundingboxAndTracking(engine, image, prevImage, 0.35);

		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			auto& box = get<0>(obj);
			auto& offset = get<1>(obj);
			ccutil::drawbbox(image, box, ccutil::DrawType::NoName);
			drawArrow(image, Point(offset[0], offset[1]), Point(offset[2], offset[3]), 10, 35, Scalar(0, 255, 0), 2, 16);

			ccutil::drawbbox(prevImage, box, ccutil::DrawType::NoName);
			drawArrow(prevImage, Point(offset[0], offset[1]), Point(offset[2], offset[3]), 10, 35, Scalar(0, 255, 0), 2, 16);
		}

		imwrite("results/coco.tracking.jpg", image);

#ifdef _WIN32
		cv::imshow("coco.tracking.current", image);
		cv::imshow("coco.tracking.prev", prevImage);
		cv::waitKey();
		cv::destroyAllWindows();
#endif
		INFO("done.");
	}
};