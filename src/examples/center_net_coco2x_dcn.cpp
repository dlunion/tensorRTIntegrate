
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include "infer/ct_detect_backend.hpp"

using namespace cv;
using namespace std;

namespace examples {

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

	static vector<ccutil::BBox> detectBoundingbox(const shared_ptr<TRTInfer::Engine>& boundingboxDetect_, const Mat& image, float threshold = 0.3) {

		if (boundingboxDetect_ == nullptr) {
			INFO("detectBoundingbox failure call, model is nullptr");
			return vector<ccutil::BBox>();
		}

		preprocessCenterNetImageToTensor(image, 0, boundingboxDetect_->input());
		boundingboxDetect_->forward();

		auto outHM = boundingboxDetect_->tensor("hm");
		auto outHMPool = boundingboxDetect_->tensor("hm_pool");
		auto outWH = boundingboxDetect_->tensor("wh");
		auto outXY = boundingboxDetect_->tensor("reg");
		const int stride = 4;

		vector<ccutil::BBox> bboxs;
		Size inputSize = boundingboxDetect_->input()->size();
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
						ccutil::BBox box = restoreCenterNetBox(dx, dy, dw, dh, j, i, stride, inputSize, image.size());
						box = box.box() & Rect(0, 0, image.cols, image.rows);
						box.label = class_;
						box.score = *ohmptr;

						if (box.area() > 0)
							bboxs.push_back(box);
					}
					++ohmptr;
					++ohmpoolptr;
				}
			}
		}
		return bboxs;
	}

	static vector<vector<ccutil::BBox>> detectBoundingboxOptim(const shared_ptr<TRTInfer::Engine>& boundingboxDetect_, const vector<Mat>& images,
		float threshold, int maxobjs, TRTInfer::CTDetectBackend* detectBackend) {

		if (boundingboxDetect_ == nullptr) {
			INFO("detectBoundingbox failure call, model is nullptr");
			return vector<vector<ccutil::BBox>>();
		}
		boundingboxDetect_->input()->resize(images.size());

		vector<Size> imsize;
		for (int i = 0; i < images.size(); ++i) {
			preprocessCenterNetImageToTensor(images[i], i, boundingboxDetect_->input());  //1.0 ms
			imsize.emplace_back(images[i].size());
		}

		boundingboxDetect_->forward(false);  //41.5 ms
		auto outHM = boundingboxDetect_->tensor("hm");
		auto outHMPool = boundingboxDetect_->tensor("hm_pool");
		auto outWH = boundingboxDetect_->tensor("wh");
		auto outXY = boundingboxDetect_->tensor("reg");
		return detectBackend->forwardGPU(outHM, outHMPool, outWH, outXY, imsize, threshold, maxobjs); // 0.25 ms
	}

	void center_net_coco2x_dcn() {

		INFOW("onnx to trtmodel...");

		// tensorRT 7.0 + OnnX:  Must be an explicit batchsize, that is, batchsize must be specified at compile time.
		int batchSize = 2;
		auto modelFile = ccutil::format("models/dladcnv2.fp32.b%d.trtmodel", batchSize);
		if (!ccutil::exists(modelFile)) {

			if (!ccutil::exists("models/dladcnv2.onnx")) {
				INFOW(
					"models/dladcnv2.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dladcnv2.onnx "
					"or use centerNetDLADCNOnnX/dladcn_export_onnx.py to generate"
				);
				return;
			}

			TRTBuilder::compileTRT(
				TRTBuilder::TRTMode_FP32, {}, batchSize,
				TRTBuilder::ModelSource("models/dladcnv2.onnx"),
				modelFile, {TRTBuilder::InputDims(3, 512, 512)}
			);
		}

		INFO("load model: %s", modelFile.c_str());
		auto engine = TRTInfer::loadEngine(modelFile);
		if (!engine) {
			INFO("can not load model.");
			return;
		}

		INFO("forward...");
		vector<Mat> images{
			imread("imgs/www.jpg"),
			imread("imgs/17790319373_bd19b24cfc_k.jpg")
		};

		TRTInfer::CTDetectBackend backend(engine->getCUStream());
		auto imobjs = detectBoundingboxOptim(engine, images, 0.3, 100, &backend);  // 43.86 ms
		
		for (int j = 0; j < images.size(); ++j) {
			auto& objs = imobjs[j];
			objs = ccutil::nms(objs, 0.5);

			INFO("objs.length = %d", objs.size());
			for (int i = 0; i < objs.size(); ++i) {
				auto& obj = objs[i];
				ccutil::drawbbox(images[j], obj);
			}
			imwrite(ccutil::format("results/%d.centernet.coco2x.dcn.jpg", j), images[j]);
		}

#ifdef _WIN32
		cv::imshow("dla dcn detect 1", images[0]);
		cv::imshow("dla dcn detect 2", images[1]);
		cv::waitKey();
		cv::destroyAllWindows();
#endif
		INFO("done.");
	}
};