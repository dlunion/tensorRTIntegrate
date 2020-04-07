
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

#define GPUID		0

struct FaceBox : ccutil::BBox {
	vector<Point2f> landmark;

	FaceBox() {}
	FaceBox(const ccutil::BBox& other):ccutil::BBox(other) {}
};

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
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
	float x0 = ((cellx) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y0 = ((celly) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return Scalar(x0, y0, x, y);
}

static void preprocessCenterNetImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	float scale = 0;
	int outH = tensor->height();
	int outW = tensor->width();
	if (image.cols >= image.rows)
		scale = outW / (float)image.cols;
	else
		scale = outH / (float)image.rows;

	Mat matrix = getRotationMatrix2D(Point2f(image.cols*0.5, image.rows*0.5), 0, scale);
	matrix.at<double>(0, 2) -= image.cols*0.5 - outW * 0.5;
	matrix.at<double>(1, 2) -= image.rows*0.5 - outH * 0.5;

	Mat outImage;
	cv::warpAffine(image, outImage, matrix, Size(outW, outH));

	vector<Mat> ms(image.channels());
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = Mat(tensor->height(), tensor->width(), CV_32F, tensor->cpu<float>(numIndex, i));

	outImage.convertTo(outImage, CV_32F, 1 / 255.0);
	split(outImage, ms);

	float mean[3] = {0.40789654, 0.44719302, 0.47026115};
	float std[3] = {0.28863828, 0.27408164, 0.27809835};
	for (int i = 0; i < 3; ++i)
		ms[i] = (ms[i] - mean[i]) / std[i];
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

static vector<FaceBox> detectDBFace(const shared_ptr<TRTInfer::Engine>& dbfaceDetect_, const Mat& image, float threshold = 0.3) {

	if (dbfaceDetect_ == nullptr) {
		INFO("detectDBFace failure call, model is nullptr");
		return vector<FaceBox>();
	}

	Assert(image.cols % 32 == 0 || image.rows % 32 == 0);

	float mean[3] = {0.408, 0.447, 0.47};
	float std[3] = {0.289, 0.274, 0.278};

	dbfaceDetect_->input()->setNormMat(0, image, mean, std);
	dbfaceDetect_->forward();
	auto outHM = dbfaceDetect_->tensor("hm");
	auto outHMPool = dbfaceDetect_->tensor("pool_hm");
	auto outTLRB = dbfaceDetect_->tensor("tlrb");
	auto outLandmark = dbfaceDetect_->tensor("landmark");
	const int stride = 4;

	vector<FaceBox> bboxs;
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
					
					FaceBox box(ccutil::BBox(x, y, r, b, *ohmptr, class_));
					if (box.area() > 0) {

						box.landmark.resize(5);
						for (int k = 0; k < box.landmark.size(); ++k) {
							float landmark_x = outLandmark->at<float>(0, k, i, j) * 4;
							float landmark_y = outLandmark->at<float>(0, k+5, i, j) * 4;
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

Mat padImage(const Mat& image, int stride = 32) {

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

void demoOnnx() {

	if (!ccutil::exists("models/demo.onnx")) {
		INFOE("models/demo.onnx not exists, run< python plugin_onnx_export.py > generate demo.onnx.");
		return;
	}

	INFOW("onnx to trtmodel...");
	TRTBuilder::compileTRT(
		TRTBuilder::TRTMode_FP32, {}, 4,
		TRTBuilder::ModelSource("models/demo.onnx"),
		"models/demo.fp32.trtmodel", nullptr, "", "",
		{TRTBuilder::InputDims(3, 5, 5), TRTBuilder::InputDims(3, 5, 5)}
	);
	INFO("done.");

	INFO("load model: models/demo.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/demo.fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	INFO("forward...");

	engine->input(0)->setTo(0.25);
	engine->input(1)->setTo(0);
	engine->forward();
	auto output = engine->output(0);
	output->print();
	INFO("done.");
}

void dladcnOnnx(){

	INFOW("onnx to trtmodel...");

	if (!ccutil::exists("models/dladcnv2.fp32.trtmodel")) {

		if(!ccutil::exists("models/dladcnv2.onnx")){
			INFOW(
				"models/dladcnv2.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dladcnv2.onnx "
				"or use centerNetDLADCNOnnX/dladcn_export_onnx.py to generate"
			);
			return;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource("models/dladcnv2.onnx"),
			"models/dladcnv2.fp32.trtmodel", nullptr, "", "",
			{TRTBuilder::InputDims(3, 512, 512)}
		);
	}

	INFO("load model: models/dladcnv2.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/dladcnv2.fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	INFO("forward...");
	Mat image = imread("www.jpg");
	
	auto objs = detectBoundingbox(engine, image, 0.3);
	objs = ccutil::nms(objs, 0.5);

	INFO("objs.length = %d", objs.size());
	for(int i = 0; i < objs.size(); ++i){
		auto& obj = objs[i];
		ccutil::drawbbox(image, obj);
	}

	imwrite("www.dla.draw.jpg", image);

#ifdef _WIN32
	cv::imshow("dla dcn detect", image);
	cv::waitKey();
	cv::destroyAllWindows();
#endif
	INFO("done.");
}

void centerTrack_coco_tracking() {
	INFOW("onnx to trtmodel...");

	if (!ccutil::exists("models/coco_tracking.fp32.trtmodel")) {

		if (!ccutil::exists("models/coco_tracking.onnx")) {

			//  nuScenes_3Dtracking.onnx

			INFOW(
				"models/coco_tracking.onnx not found, download url: http://zifuture.com:1000/fs/public_models/coco_tracking.onnx"
			);
			return;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource("models/coco_tracking.onnx"),
			"models/coco_tracking.fp32.trtmodel", nullptr, "", "",
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
	Mat prevImage = imread("000020.jpg");
	Mat image = imread("000023.jpg");

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

	imwrite("coco.tracking.jpg", image);

#ifdef _WIN32
	bool showCurrent = true;
	while (true) {

		if(showCurrent)
			cv::imshow("coco.tracking", image);
		else
			cv::imshow("coco.tracking", prevImage);

		int key = cv::waitKey();
		if (key == 'q')
			break;

		showCurrent = !showCurrent;
	}
	cv::destroyAllWindows();
#endif
	INFO("done.");
}

void dbfaceOnnx() {

	Mat image = imread("selfie.jpg");
	Mat padimage = padImage(image);
	string modelPath = ccutil::format("models/dbface.%dx%d.fp32.trtmodel", padimage.cols, padimage.rows);

	if (!ccutil::exists(modelPath)) {

		if (!ccutil::exists("models/dbface.onnx")) {
			INFOW(
				"models/dbface.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dbface.onnx"
			);
			return;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource("models/dbface.onnx"),
			modelPath, nullptr, "", "",
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
	auto objs = detectDBFace(engine, image, 0.25);
	 
	INFO("objs.length = %d", objs.size());
	for (int i = 0; i < objs.size(); ++i) {
		auto& obj = objs[i];
		ccutil::drawbbox(image, obj, ccutil::DrawType::Empty);

		for (int k = 0; k < obj.landmark.size(); ++k) {
			cv::circle(image, obj.landmark[k], 3, Scalar(0, 0, 255), -1, 16);
		}
	}
	 
	imwrite("selfie.draw.jpg", image);

#ifdef _WIN32
	cv::imshow("dbface selfie detect", image);
	cv::waitKey();
	cv::destroyAllWindows();
#endif
	INFO("done.");
}

int main() {
	//log保存为文件
	ccutil::setLoggerSaveDirectory("logs");
	TRTBuilder::setDevice(GPUID);

	dbfaceOnnx();
	demoOnnx();
	dladcnOnnx();
	centerTrack_coco_tracking();
	return 0;
}