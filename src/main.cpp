
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include <infer/task_pool.hpp>
#include <test_common.hpp>

using namespace cv;
using namespace std;

void testInt8Caffe() {

	TRTBuilder::setDevice(0);
	auto preprocess = [](int current, int count, cv::Mat& inputOutput) {

		INFO("process: %d / %d", current, count);
		inputOutput.convertTo(inputOutput, CV_32F, 1 / 255.0f, -0.5f);
	};
	
	INFO("Int8 量化");
	TRTBuilder::compileTRT(
		TRTBuilder::TRTMode_INT8, 
		{"fc_blob1"}, 4,
		TRTBuilder::ModelSource("models/handmodel_resnet18.prototxt", "models/handmodel_resnet18.caffemodel"),
		"models/handmodel_resnet18.int8.trtmodel", 
		preprocess, 
		"models/int8data", 
		"models/handmodel_resnet18.calibrator.txt"
	);
	INFO("Int8 Done");
}

void testOnnxFP32() {

	TRTBuilder::setDevice(0);

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

void testOnnxINT8() {

	//int8是有损失的，而且是看得见的损失，精度要求不是特别严格的时候可以用，更加建议是fp16
	TRTBuilder::setDevice(0);

	auto preprocess = [](int current, int count, cv::Mat& inputOutput) {

		INFO("process: %d / %d", current, count);
		inputOutput.convertTo(inputOutput, CV_32F, 1 / 255.0f);

		float mean[3] = {0.485, 0.456, 0.406};
		float std[3] = {0.229, 0.224, 0.225};

		Mat ms[3];
		split(inputOutput, ms);
		for (int i = 0; i < 3; ++i) {
			ms[i] = (ms[i] - mean[i]) / std[i];
		}
		merge(ms, 3, inputOutput);
	};

	INFO("onnx to trtmodel...");
	TRTBuilder::compileTRT(
		TRTBuilder::TRTMode_INT8, {}, 4,
		TRTBuilder::ModelSource("models/efficientnet-b0.onnx"),
		"models/efficientnet-b0.int8.trtmodel", 
		preprocess,
		"models/int8data",
		"models/efficientnet-b0.calibrator.txt"
	);
	INFO("done.");

	INFO("load model: models/efficientnet-b0.int8.trtmodel");
	auto engine = TRTInfer::loadEngine("models/efficientnet-b0.int8.trtmodel");
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

void testPlugin() {

	TRTBuilder::setDevice(0);

	//对于fp16，参照这里的文章：https://blog.csdn.net/yutianzuijin/article/details/90521252
	//和这里的计算能力向导：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
	//如果显卡不支持fp16加速，那么性能会比fp32差（因为涉及到转换数据类型耗时）
	//如果显卡支持fp16加速，那么，fp16的性能会比fp32快1倍以上，（多batch时更明显）。fp16的效果也接近fp32（几乎看不到损失），int8则是不可取（损失太重）
	//INFO("Plugin FP16 转换");
	//TRTBuilder::compileTRT(
	//	TRTBuilder::TRTMode_FP16, {"myImage"}, 4, 
	//	TRTBuilder::ModelSource("models/demo.prototxt", "models/demo.caffemodel"), 
	//	"models/demo.fp16.trtmodel");
	//INFO("Plugin FP16 Done");

	INFO("Plugin FP32 转换");
	TRTBuilder::compileTRT(
		TRTBuilder::TRTMode_FP32, {"myImage"}, 4,
		TRTBuilder::ModelSource("models/demo.prototxt", "models/demo.caffemodel"),
		"models/demo.fp32.trtmodel");
	INFO("Plugin FP32 Done");

	INFO("加载模型 models/demo.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/demo.fp32.trtmodel");

	if (engine == nullptr) {
		INFO("模型加载失败.");
		return;
	}

	INFO("Inference ...");
	engine->input(0)->setTo(-0.5);
	engine->forward();

	//这里应该输出1.3，因为WReLU随便写的cuda核，
	//output[position] = (input[position] < 0 ? 0 : input[position]) + 1.3;
	engine->output(0)->print();
	INFO("done");
}

int main() {
	//log保存为文件
	ccutil::setLoggerSaveDirectory(".");

	testInt8Caffe();
	testOnnxFP32();
	testOnnxINT8();
	testPlugin();
	return 0;
}