
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include <infer/task_pool.hpp>

using namespace cv;
using namespace std;

static void compileTRT(){

	if(!ccutil::exists("models/centernet.fp32.b4.trtmodel")){
		if(!ccutil::exists("models/centernet.prototxt") || !ccutil::exists("models/centernet.caffemodel")){
			INFO("models/centernet not exists, goto download from: http://zifuture.com:1000/fs/25.shared/tensorRT_demo_model_centerNet_and_openPose.zip");
			exit(0);
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {"hm", "hm_pool", "xy", "wh"},
			4, TRTBuilder::ModelSource("models/centernet.prototxt", "models/centernet.caffemodel"), "models/centernet.fp32.b4.trtmodel"
		);
	}else{
		INFO("models/centernet.fp32.b4.trtmodel is exists, ignore compile");
	}

	if(!ccutil::exists("models/alphaPose.fp32.b32.trtmodel")){
		if(!ccutil::exists("models/alphaPose.prototxt") || !ccutil::exists("models/alphaPose.caffemodel")){
			INFO("models/alphaPose not exists, goto download from: http://zifuture.com:1000/fs/25.shared/tensorRT_demo_model_centerNet_and_openPose.zip");
			exit(0);
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {"outputHM"},
			32, TRTBuilder::ModelSource("models/alphaPose.prototxt", "models/alphaPose.caffemodel"), "models/alphaPose.fp32.b32.trtmodel"
		);
	}else{
		INFO("models/alphaPose.fp32.b32.trtmodel is exists, ignore compile");
	}
}

static void preprocessAlphaPoseImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	Size imageSize = image.size();
	float rate = 1.3;
	Size padImageSize(imageSize.width * rate, imageSize.height * rate);
	Mat padImage(padImageSize, CV_8UC3, Scalar::all(127));
	int marginx = imageSize.width * (rate - 1) * 0.5;
	int marginy = imageSize.height * (rate - 1) * 0.5;
	image.copyTo(padImage(Rect(marginx, marginy, imageSize.width, imageSize.height)));

	float scale = 0;
	Size netSize = tensor->size();
	if (netSize.width / (float)padImageSize.width <= netSize.height / (float)padImageSize.height)
		scale = netSize.width / (float)padImageSize.width;
	else
		scale = netSize.height / (float)padImageSize.height;

	Mat matrix = getRotationMatrix2D(Point2f(padImageSize.width*0.5, padImageSize.height*0.5), 0, scale);
	matrix.at<double>(0, 2) -= (padImageSize.width - netSize.width) * 0.5;
	matrix.at<double>(1, 2) -= (padImageSize.height - netSize.height) * 0.5;

	cv::warpAffine(padImage, padImage, matrix, netSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(127));
	cvtColor(padImage, padImage, COLOR_BGR2RGB);
	float mean[3] = {0.406, 0.457, 0.480};
	padImage.convertTo(padImage, CV_32F, 1 / 256.f);

	vector<Mat> ms(tensor->channel());
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = Mat(tensor->height(), tensor->width(), CV_32F, tensor->cpu<float>(numIndex, i));

	split(padImage, ms);
	for (int i = 0; i < ms.size(); ++i)
		ms[i] -= mean[i];
}

//humans.size must be less engine.maxBatchSize()
static vector<vector<Point3f>> detectHumanKeypoint(const shared_ptr<TRTInfer::Engine> keypointDetect_, const vector<Mat>& humans) {

	Assert(humans.size() <= keypointDetect_->maxBatchSize());

	auto inputTensor = keypointDetect_->input();
	Size netSize = inputTensor->size();
	inputTensor->resize(humans.size());

	for (int i = 0; i < humans.size(); ++i)
		preprocessAlphaPoseImageToTensor(humans[i], i, inputTensor);

	keypointDetect_->forward();
	auto outHM = keypointDetect_->tensor("outputHM");
	const int stride = 4;
	vector<vector<Point3f>> keypoints_humans(humans.size());

	for (int ibatch = 0; ibatch < humans.size(); ++ibatch) {
		auto& human = humans[ibatch];
		Size imageSize = human.size();
		float rate = 1.3;
		int marginx = imageSize.width * (rate - 1) * 0.5;
		int marginy = imageSize.height * (rate - 1) * 0.5;
		vector<Point3f>& keypoints = keypoints_humans[ibatch];

		Size padImageSize(imageSize.width * rate, imageSize.height * rate);
		float scale = 0;
		if (netSize.width / (float)padImageSize.width <= netSize.height / (float)padImageSize.height)
			scale = netSize.width / (float)padImageSize.width;
		else
			scale = netSize.height / (float)padImageSize.height;

		float sx = imageSize.width * rate / (float)netSize.width * stride / scale;
		float sy = imageSize.height * rate / (float)netSize.height * stride / scale;
		int selectChannels = 17;   //  reference workspace/pose.png
		for (int i = selectChannels; i < outHM->channel(); ++i) {
			//for (int i = 0; i < selectChannels; ++i) {
			float* ptr = outHM->cpu<float>(ibatch, i);
			int n = outHM->count(2);
			int maxpos = std::max_element(ptr, ptr + n) - ptr;
			float confidence = ptr[maxpos];
			float x = ((maxpos % outHM->width()) * stride - (float)netSize.width * 0.5) / scale + padImageSize.width * 0.5 - marginx;
			float y = ((maxpos / outHM->width()) * stride - (float)netSize.height * 0.5) / scale + padImageSize.height * 0.5 - marginy;
			keypoints.push_back(Point3f(x, y, confidence));
		}
	}
	return keypoints_humans;
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

	float mean[3] = {0.406, 0.447, 0.470};
	float std[3] = {0.289, 0.274, 0.278};
	for (int i = 0; i < 3; ++i)
		ms[i] = (ms[i] - mean[i]) / std[i];
}

static vector<ccutil::BBox> detectBoundingbox(const shared_ptr<TRTInfer::Engine>& boundingboxDetect_, const Mat& image, float threshold = 0.3){

	if (boundingboxDetect_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return vector<ccutil::BBox>();
	}

	preprocessCenterNetImageToTensor(image, 0, boundingboxDetect_->input());
	boundingboxDetect_->forward();
	auto outHM = boundingboxDetect_->tensor("hm");
	auto outHMPool = boundingboxDetect_->tensor("hm_pool");
	auto outWH = boundingboxDetect_->tensor("wh");
	auto outXY = boundingboxDetect_->tensor("xy");
	const int stride = 4;

	vector<ccutil::BBox> bboxs;
	Size inputSize = boundingboxDetect_->input()->size();
	float sx = image.cols / (float)inputSize.width * stride;
	float sy = image.rows / (float)inputSize.height * stride;

	int c = 0;
	for (int i = 0; i < outHM->height(); ++i) {
		float* ohmptr = outHM->cpu<float>(0, c, i);
		float* ohmpoolptr = outHMPool->cpu<float>(0, c, i);
		for (int j = 0; j < outHM->width(); ++j) {
			if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {

				float dx = outXY->at<float>(0, 0, i, j);
				float dy = outXY->at<float>(0, 1, i, j);
				float dw = outWH->at<float>(0, 0, i, j);
				float dh = outWH->at<float>(0, 1, i, j);
				Rect box = restoreCenterNetBox(dx, dy, dw, dh, j, i, stride, inputSize, image.size());
				box = box & Rect(0, 0, image.cols, image.rows);

				if(box.area() > 0)
					bboxs.push_back(box);
			}
			++ohmptr;
			++ohmpoolptr;
		}
	}
	return bboxs;
}

static void drawHumanKeypointAndBoundingbox(Mat& image, const Rect& box, const vector<Point3f>& keypoints){

	int connectLines[][4] = {{0, 1, 2, 6}, {5, 4, 3, 6}, {6, 7, 8, 9}, {10, 11, 12, 7}, {15, 14, 13, 7}};
	int numLines = sizeof(connectLines) / sizeof(connectLines[0]);
	rectangle(image, box, Scalar(0, 255), 2, 16);
	
	Point baseTL = box.tl();
	int lineID = 0;
	for(int j = 0; j < numLines; ++j){
		int* line = connectLines[j];
		for(int k = 0; k < 3; ++k){
			int p = line[k];
			int np = line[k+1];
			cv::line(image, 
				Point(keypoints[p].x, keypoints[p].y) + baseTL,
				Point(keypoints[np].x, keypoints[np].y) + baseTL,
				ccutil::randColor(lineID++), 3, 16
			);
		}
	}

	for(int j = 0; j < keypoints.size(); ++j){
		circle(image, Point(keypoints[j].x, keypoints[j].y) + box.tl(), 5, ccutil::randColor(j), -1, 16);
	}
}

static Rect restoreFaceBox(float dx, float dy, float dr, float db, float cellx, float celly, int stride, Size netSize, Size imageSize) {

	float sx = netSize.width / (float)imageSize.width;
	float sy = netSize.height / (float)imageSize.height;

	float x = ((cellx - dx) * stride) / sx;
	float y = ((celly - dy) * stride) / sy;
	float r = ((cellx + dr) * stride) / sx;
	float b = ((celly + db) * stride) / sy;
	return Rect(Point(x, y), Point(r + 1, b + 1));
}

static vector<ccutil::BBox> detectFaceBoundingbox(const shared_ptr<TRTInfer::Engine>& faceEngine, const Mat& image, float threshold = 0.3){

	if (faceEngine == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return vector<ccutil::BBox>();
	}

	float mean[] = {0.408, 0.447, 0.47};
	float std[] = {0.289, 0.274, 0.278};
	faceEngine->input()->setNormMat(0, image, mean, std);
	faceEngine->forward();

	auto hm_pool = faceEngine->tensor("934");
	auto hm = faceEngine->tensor("932");
	auto box_delta = faceEngine->tensor("933");
	auto landmark = faceEngine->tensor("931");
	const int stride = 4;

	vector<ccutil::BBox> bboxs;
	Size inputSize = faceEngine->input()->size();
	float sx = image.cols / (float)inputSize.width * stride;
	float sy = image.rows / (float)inputSize.height * stride;

	int c = 0;
	for (int i = 0; i < hm->height(); ++i) {
		float* ohmptr = hm->cpu<float>(0, c, i);
		float* ohmpoolptr = hm_pool->cpu<float>(0, c, i);
		for (int j = 0; j < hm->width(); ++j) {
			if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {

				float dx = box_delta->at<float>(0, 0, i, j);
				float dy = box_delta->at<float>(0, 1, i, j);
				float dr = box_delta->at<float>(0, 2, i, j);
				float db = box_delta->at<float>(0, 3, i, j);
				Rect box = restoreFaceBox(dx, dy, dr, db, j, i, stride, inputSize, image.size());
				box = box & Rect(0, 0, image.cols, image.rows);

				if(box.area() > 0)
					bboxs.push_back(box);
			}
			++ohmptr;
			++ohmpoolptr;
		}
	}
	return bboxs;
}

void testmbv3(){

	if(!ccutil::exists("models/mbv3.fp32.b1.trtmodel")){
		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {},
			1, TRTBuilder::ModelSource("models/mbv3.onnx"), "models/mbv3.fp32.b1.trtmodel"
		);
	}

	// Mat image = imread("cimage.jpg");
	// //auto engine = TRTInfer::loadEngine("models/centerface.fp32.b1.trtmodel");
	// auto engine = TRTInfer::loadEngine("models/mbv3.fp32.b1.trtmodel");
	// engine->input()->resize(1);

	// float mean[] = {0.408, 0.447, 0.47};
	// float std[] = {0.289, 0.274, 0.278};
	// resize(image, image, Size(480, 352));
	// engine->input()->setNormMat(0, image, mean, std);

	// ccutil::Timer t;
	// for(int i = 0; i < 1000; ++i){
	// 	engine->forward();
	// }
	// INFO("fee only inference: %.2f ms / pic", t.end() / 1000);

	// t.begin();
	// for(int i = 0; i < 1000; ++i){
	// 	auto objs = detectFaceBoundingbox(engine, image, 0.3);
	// 	//engine->forward();
	// }
	// INFO("fee full: %.2f ms / pic", t.end() / 1000);

	auto engine = TRTInfer::loadEngine("models/mbv3.fp32.b1.trtmodel");
	Mat image = imread("cimage.jpg");
	// int neww = image.cols;
	// int newh = neww / (float)engine->input()->width() * engine->input()->height();

	// Mat timage(newh, neww, CV_8UC3, Scalar(0));
	// image.copyTo(timage(Rect(0, 0, image.cols, image.rows)));
	
	auto objs = detectFaceBoundingbox(engine, image, 0.3);

	for(auto& obj : objs){
		rectangle(image, obj, Scalar(0, 255, 255), 2, 16);
	}
	imwrite("image.jpg", image);
	exit(0);
}

int main_old() {

	//save log to file
	ccutil::setLoggerSaveDirectory("logs");
	INFO("startup.");

	TRTInfer::setDevice(0);
	TRTInfer::initNVPlugins();
	//testmbv3();
	//TRTInfer::initNVPlugins();    if use ssd or yolo
	compileTRT();

	INFO("load engine.");
	auto boxEngine = TRTInfer::loadEngine("models/centernet.fp32.b4.trtmodel");
	auto poseEngine = TRTInfer::loadEngine("models/alphaPose.fp32.b32.trtmodel");

	if(!boxEngine){
		INFO("boxEngine is nullptr");
		return 0;
	}

	if(!poseEngine){
		INFO("poseEngine is nullptr");
		return 0;
	}

	INFO("detect ....");
	Mat image = imread("person.jpg");
	auto objs = detectBoundingbox(boxEngine, image);

	vector<Mat> crops;
	for(int i = 0; i < objs.size(); ++i){
		auto& obj = objs[i];
		crops.push_back(image(obj.box()));
	}

	auto keys = detectHumanKeypoint(poseEngine, crops);
	for(int i = 0; i < keys.size(); ++i)
		drawHumanKeypointAndBoundingbox(image, objs[i], keys[i]);
	
	INFO("save result");
	cv::imwrite("person_draw.jpg", image);
	INFO("done.");
	return 0;
}