
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

#define GPUID		0

void softmax(float* ptr, int count) {

	float total = 0;
	float* p = ptr;
	for (int i = 0; i < count; ++i)
		total += exp(*p++);

	p = ptr;
	for (int i = 0; i < count; ++i, ++p)
		*p = exp(*p) / total;
}

int argmax(float* ptr, int count, float* confidence) {

	auto ind = std::max_element(ptr, ptr + count) - ptr;
	if (confidence) *confidence = ptr[ind];
	return ind;
}

vector<tuple<int, float>> topRank(float* ptr, int count, int ntop=5) {

	vector<tuple<int, float>> result;
	for (int i = 0; i < count; ++i) 
		result.push_back(make_tuple(i, ptr[i]));
	
	std::sort(result.begin(), result.end(), [](tuple<int, float>& a, tuple<int, float>& b) {
		return get<1>(a) > get<1>(b);
	});

	int n = min(ntop, (int)result.size());
	result.erase(result.begin() + n, result.end());
	return result;
}

void testOnnxFP32() {

	INFO("onnx to trtmodel...");
	TRTBuilder::compileTRT(
		TRTBuilder::TRTMode_FP32, {}, 4,
		TRTBuilder::ModelSource("models/efficientnet-b0.onnx"),
		"models/efficientnet-b0.fp32.trtmodel"
	);
	INFO("done.");

	INFO("load model: models/efficientnet-b0.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/efficientnet-b0.fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	INFO("forward...");

	auto labelName = ccutil::loadList("labels_map_lines.txt");
	float mean[3] = {0.485, 0.456, 0.406};
	float std[3] = {0.229, 0.224, 0.225};
	Mat image = imread("img.jpg");

	//对于自己归一化的时候，用setMat函数，要求类型是CV32F并且通道一致
	//engine->input()->setMat(0, image);
	engine->input()->setNormMat(0, image, mean, std);
	engine->forward();

	auto output = engine->output(0);
	//float conf = 0;
	//int label = argmax(output->cpu(), output->count(), &conf);
	//INFO("label: %d, conf = %f", label, conf);

	auto rank5 = topRank(output->cpu<float>(), output->count());
	for (int i = 0; i < rank5.size(); ++i) {
		int label = get<0>(rank5[i]);
		float confidence = get<1>(rank5[i]);
		string name = labelName[label].substr(4);		//000=abc

		INFO("top %d: %.2f %%, %s", i, confidence * 100, name.c_str());
	}

	INFO("done.");
}

void demoOnnx(){

	if(!ccutil::exists("models/demo.onnx")){
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

	float mean[3] = {0.408, 0.447, 0.470};
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
	auto outXY = boundingboxDetect_->tensor("reg");
	const int stride = 4;

	vector<ccutil::BBox> bboxs;
	Size inputSize = boundingboxDetect_->input()->size();
	float sx = image.cols / (float)inputSize.width * stride;
	float sy = image.rows / (float)inputSize.height * stride;

	for(int class_ = 0; class_ < outHM->channel(); ++class_){
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

					if(box.area() > 0)
						bboxs.push_back(box);
				}
				++ohmptr;
				++ohmpoolptr;
			}
		}
	}
	return bboxs;
}

void dladcnOnnx(){

	INFOW("onnx to trtmodel...");

	if (!ccutil::exists("models/dladcnv2.fp32.trtmodel")) {

		if(!ccutil::exists("models/dladcnv2.onnx")){
			INFOE(
				"models/dladcnv2.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dladcnv2.onnx\n"
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
#endif
	INFO("done.");
}

int main() {
	//log保存为文件
	ccutil::setLoggerSaveDirectory("logs");
	TRTBuilder::setDevice(GPUID);

	demoOnnx();
	dladcnOnnx();
	return 0;
}