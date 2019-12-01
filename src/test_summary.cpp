
#if 0

#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include <infer/task_pool.hpp>
#include <test_common.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

void testInt8Caffe() {

	auto preprocess = [](int current, int count, cv::Mat& inputOutput) {

		INFO("process: %d / %d", current, count);
		inputOutput.convertTo(inputOutput, CV_32F, 1 / 255.0f, -0.5f);
	};

	TRTBuilder::caffeToTRT_INT8(
		"models/handmodel_resnet18.prototxt",
		"models/handmodel_resnet18.caffemodel",
		{"fc_blob1"}, 4, "models/resnet18_int8.trtmodel", "E:/globaldata/int8data_part", 
		preprocess
	);
}

void testResNet50() {

	TRTInfer::setDevice(1);
	TRTBuilder::caffeToTRTFP32OrFP16(
		"models/classification_resnet50.prototxt",
		"models/classification_resnet50.caffemodel",
		{"prob"}, 4, "models/resnet50.trtmodel"
	);
	INFO("done");
	auto engine = TRTInfer::loadEngine("models/resnet50.trtmodel");
	Mat image = imread("img.jpg");

	ccutil::Timer t;
	for (int i = 0; i < 1000; ++i) {
		float mean[3] = {0.485, 0.456, 0.406};
		float std[3] = {0.229, 0.224, 0.225};
		Mat tmp = image.clone();
		tmp.convertTo(tmp, CV_32F);
		engine->input(0)->setNormMat(0, tmp, mean, std);
		engine->forward();
		engine->output(0)->gpu();
	}
	INFO("ticks resnet50: %.2f ms", t.end());
}

void testOnnxInfer() {
	TRTBuilder::setDevice(1);
	//TRTBuilder::onnxToTRTFP32OrFP16(
	//	"models/efficientnet-b0.onnx",
	//	4, "models/efficientnet-b0.onnx.trtmodel"
	//);

	auto engine = TRTInfer::loadEngine("models/efficientnet-b0.onnx.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	float mean[3] = {0.485, 0.456, 0.406};
	float std[3] = {0.229, 0.224, 0.225};
	Mat image = imread("img.jpg");
	engine->input()->setNormMat(0, image, mean, std);
	engine->forward();

	auto output = engine->output(0);
	float conf = 0;
	int label = argmax(output->cpu(), output->count(), &conf);
	INFO("label: %d, conf = %f", label, conf);
	INFO("done.");
}

struct ClassifierData {
	cv::Mat image;
	shared_ptr<TRTInfer::Tensor> imageCache;
	shared_ptr<TRTInfer::Tensor> dla_wh;
	shared_ptr<TRTInfer::Tensor> dla_xy;
	shared_ptr<TRTInfer::Tensor> dla_hm;
	shared_ptr<TRTInfer::Tensor> prob_pool;
	int label = -1;
	float conf = 0;

	ClassifierData() {}
	ClassifierData(const cv::Mat& image) {
		this->image = image;
		imageCache.reset(new TRTInfer::Tensor());
		dla_wh.reset(new TRTInfer::Tensor());
		dla_xy.reset(new TRTInfer::Tensor());
		dla_hm.reset(new TRTInfer::Tensor());
		prob_pool.reset(new TRTInfer::Tensor());
	}
};

vector<Point3f> nms_one_channel(float* ptr, int f_h, int f_w, float* referance_ptr, float minScore, float mindist) {

	vector<Point3f> out_channel;
	int w = f_w;
	int h = f_h;

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			float value = *ptr++;
			float ref = *referance_ptr++;

			if (value == ref) {
				if (value > minScore)
					out_channel.emplace_back(Point3f(x, y, value));
			}
		}//x
	}//y
	return out_channel;
}

vector<ccutil::BBox> centerNetDetect(
	Size inputSize,
	TRTInfer::Tensor* dla_hm,
	TRTInfer::Tensor* dla_wh,
	TRTInfer::Tensor* dla_xy,
	TRTInfer::Tensor* prob_pool, 
	Size imageSize) {

	float threshold = 0.3;
	float scalex = inputSize.width / (float)imageSize.width;
	float scaley = inputSize.height / (float)imageSize.height;

	auto hm = dla_hm;
	auto wh = dla_wh;
	auto xy = dla_xy;

	Mat heatmap(hm->height(), hm->width(), CV_32F, hm->cpu());
	auto pts = nms_one_channel(hm->cpu(), hm->height(), hm->width(), prob_pool->cpu(), threshold, 0);
	int width = inputSize.width;
	int height = inputSize.height;
	const int Stride = 4;
	vector<ccutil::BBox> objs;
	for (int i = 0; i < pts.size(); ++i) {
		auto& p = pts[i];
		int fx = p.x;
		int fy = p.y;
		fx = std::min(width / Stride - 1, std::max(0, fx));
		fy = std::min(height / Stride - 1, std::max(0, fy));

		float ox = xy->at(0, 0, fy, fx) * Stride;
		float oy = xy->at(0, 1, fy, fx) * Stride;
		float dw = wh->at(0, 0, fy, fx) * Stride;
		float dh = wh->at(0, 1, fy, fx) * Stride;

		ccutil::BBox box;
		box.x = (fx * Stride + ox - dw / 2) / scalex;
		box.y = (fy * Stride + oy - dh / 2) / scaley;
		box.r = box.x + dw / scalex - 1;
		box.b = box.y + dh / scaley - 1;
		box.score = p.z;
		objs.push_back(box);
	}
	objs = ccutil::nms(objs, 0.8);
	return objs;
}

typedef Taskpool<ClassifierData, TRTInfer::Engine> TestPool;
void testTaskPool() {

	TRTInfer::setDevice(1);
	//TRTBuilder::caffeToTRTFP32OrFP16(
	//	"models/CenterNetBigBlue.prototxt",
	//	"models/CenterNetBigBlue.caffemodel",
	//	{"dla_hm", "prob_pool", "dla_xy", "dla_wh"}, 4, "models/CenterNetBigBlue.trtmodel"
	//);
	//INFO("gen done.");
	auto engine = TRTInfer::loadEngine("models/CenterNetBigBlue.trtmodel");
	double preprocessTime = 0;

	auto preprocess = [&](const vector<int>& inputDims, TestPool::InOutFormat& input) {

		ccutil::Timer t;
		input.imageCache->resize(1, inputDims[1], inputDims[2], inputDims[3]);

		float mean[3] = {0.408f, 0.447f, 0.47f};
		float std[3] = {0.289f, 0.274f, 0.278f};
		input.imageCache->setNormMat(0, input.image, mean, std);

#pragma omp critical
		{
			preprocessTime += t.end();
		}
	};

	double forwardTime = 0;
	auto batchForward = [&](TRTInfer::Engine* net, TestPool::Batch& batch) {

		ccutil::Timer t;
		TRTInfer::setDevice(1);
		auto inputTensor = net->input(0);
		inputTensor->to_cpu(false);
		inputTensor->resize(batch.size());

		for (int i = 0; i < batch.size(); ++i) {
			auto& cacheTensor = batch[i]->imageCache;

			Assert(inputTensor->count(1) == cacheTensor->count());
			memcpy(inputTensor->cpu(i), cacheTensor->cpu(), cacheTensor->bytes());
		}
		net->forward();

		auto dla_hm = net->tensor("dla_hm");
		auto prob_pool = net->tensor("prob_pool");
		auto dla_xy = net->tensor("dla_xy");
		auto dla_wh = net->tensor("dla_wh");

		for (int i = 0; i < batch.size(); ++i) {
			auto& output = *batch[i];
			output.dla_hm->from(dla_hm->cpu(i), 1, dla_hm->channel(), dla_hm->height(), dla_hm->width());
			output.prob_pool->from(prob_pool->cpu(i), 1, prob_pool->channel(), prob_pool->height(), prob_pool->width());
			output.dla_xy->from(dla_xy->cpu(i), 1, dla_xy->channel(), dla_xy->height(), dla_xy->width());
			output.dla_wh->from(dla_wh->cpu(i), 1, dla_wh->channel(), dla_wh->height(), dla_wh->width());
		}

#pragma omp critical
		{
			forwardTime += t.end();
		}
	};

	double backProcessTime = 0;
	auto backProcess = [&](TestPool::InOutFormat& output) {

		ccutil::Timer t;
		auto objs = centerNetDetect(
			output.imageCache->size(),
			output.dla_hm.get(), output.dla_wh.get(), output.dla_xy.get(), output.prob_pool.get(), output.image.size());

#pragma omp critical
		{
			backProcessTime += t.end();

			//INFO("objs.size() = %d", objs.size());
			//Mat im = output.image.clone();
			//for (int i = 0; i < objs.size(); ++i) {
			//	rectangle(im, objs[i], Scalar(0, 255), 2);
			//}
			//imshow(format("%d-im", omp_get_thread_num()), im);
			//waitKey(1);

		}
		//INFO("objs.size() = %d", objs.size());
	};

	auto inputTensor = engine->input(0);
	shared_ptr<TestPool> pool(new TestPool(
		engine,
		preprocess, 
		batchForward, 
		backProcess, 
		inputTensor->dims(),
		engine->maxBatchSize()));
	
	Mat image = imread("6-58-48.avi_top_left2018-11-27-16-58-48.avi_01582.jpg");
	int numThreads = 10;
	vector<ClassifierData> datas;
	for (int i = 0; i < numThreads; ++i)
		datas.push_back(ClassifierData(image));

	for (int i = 0; i < numThreads; ++i) {
		int ind = omp_get_thread_num();
		pool->forward(datas[ind]);
	}

	preprocessTime = 0;
	forwardTime = 0;
	backProcessTime = 0;

	ccutil::Timer totalTick;

#pragma omp parallel for num_threads(numThreads)
	for (int i = 0; i < 1000; ++i) {
		int ind = omp_get_thread_num();
		pool->forward(datas[ind]);
	}
	INFO("taskPool predict 1000 pic, threads: %d, total time: %.2f ms\n"
		"preprocessTime: %f ms\n"
		"forwardTime: %f ms\n"
		"backProcessTime: %f ms"
		,numThreads, totalTick.end(), preprocessTime, forwardTime, backProcessTime);
	//INFO("predict: %d, conf: %f", data.label, data.conf);
}

void testSEQ() {

	TRTInfer::setDevice(1);
	auto engine = TRTInfer::loadEngine("models/CenterNetBigBlue.trtmodel");
	Mat image = imread("6-58-48.avi_top_left2018-11-27-16-58-48.avi_01582.jpg");
	ClassifierData data(image);

	double preprocessTime = 0;
	double forwardTime = 0;
	double backProcessTime = 0;

	for (int i = 0; i < 10; ++i) {
		int ind = omp_get_thread_num();
		float mean[3] = {0.408f, 0.447f, 0.47f};
		float std[3] = {0.289f, 0.274f, 0.278f};
		engine->input(0)->setNormMat(0, image, mean, std);

		engine->forward();
		auto dla_hm = engine->tensor("dla_hm");
		auto prob_pool = engine->tensor("prob_pool");
		auto dla_xy = engine->tensor("dla_xy");
		auto dla_wh = engine->tensor("dla_wh");
		data.dla_hm->from(dla_hm->cpu(), 1, dla_hm->channel(), dla_hm->height(), dla_hm->width());
		data.prob_pool->from(prob_pool->cpu(), 1, prob_pool->channel(), prob_pool->height(), prob_pool->width());
		data.dla_xy->from(dla_xy->cpu(), 1, dla_xy->channel(), dla_xy->height(), dla_xy->width());
		data.dla_wh->from(dla_wh->cpu(), 1, dla_wh->channel(), dla_wh->height(), dla_wh->width());

		centerNetDetect(
			data.imageCache->size(),
			data.dla_hm.get(), data.dla_wh.get(), data.dla_xy.get(), data.prob_pool.get(), data.image.size());
	}

	ccutil::Timer totalTick;
	for (int i = 0; i < 1000; ++i) {
		ccutil::Timer t0;
		int ind = omp_get_thread_num();
		float mean[3] = {0.408f, 0.447f, 0.47f};
		float std[3] = {0.289f, 0.274f, 0.278f};
		engine->input(0)->setNormMat(0, image, mean, std);
		preprocessTime += t0.end();

		ccutil::Timer t1;
		engine->forward();
		auto dla_hm = engine->tensor("dla_hm");
		auto prob_pool = engine->tensor("prob_pool");
		auto dla_xy = engine->tensor("dla_xy");
		auto dla_wh = engine->tensor("dla_wh");
		data.dla_hm->from(dla_hm->cpu(), 1, dla_hm->channel(), dla_hm->height(), dla_hm->width());
		data.prob_pool->from(prob_pool->cpu(), 1, prob_pool->channel(), prob_pool->height(), prob_pool->width());
		data.dla_xy->from(dla_xy->cpu(), 1, dla_xy->channel(), dla_xy->height(), dla_xy->width());
		data.dla_wh->from(dla_wh->cpu(), 1, dla_wh->channel(), dla_wh->height(), dla_wh->width());
		forwardTime += t1.end();

		ccutil::Timer t2;
		centerNetDetect(
			data.imageCache->size(),
			data.dla_hm.get(), data.dla_wh.get(), data.dla_xy.get(), data.prob_pool.get(), data.image.size());
		backProcessTime += t2.end();
	}
	INFO("SEQ predict 1000 pic, total time: %.2f ms\n"
		"preprocessTime: %f ms\n"
		"forwardTime: %f ms\n"
		"backProcessTime: %f ms"
		, totalTick.end(), preprocessTime, forwardTime, backProcessTime);

}

void testPlugin() {


	auto preprocess = [](int current, int count, cv::Mat& inputOutput) {
		//INFO("process: %d / %d", current, count);
		inputOutput.convertTo(inputOutput, CV_32F, 1 / 255.0f, -0.5f);
	};

	//TRTBuilder::caffeToTRTFP32OrFP16("models/demo.prototxt", "models/handmodel_resnet18.caffemodel", {"myImage"}, 4, "models/demo.trtmodel");
	//TRTBuilder::caffeToTRT_INT8(
	//	"models/demo.prototxt", 
	//	"models/handmodel_resnet18.caffemodel",
	//	{"myImage"}, 4, 
	//	"models/demo.trtmodel", "E:/globaldata/int8data_part",
	//	preprocess);
	//
	auto engine = TRTInfer::loadEngine("models/demo.trtmodel");
	engine->input(0)->setTo(-0.5);
	engine->forward();
	engine->output(0)->print();
}

void testBinIO() {

	ccutil::BinIO bin;
	bin.openMemoryWrite();
	bin << "test";

	string outputData = bin.writedMemory();
	bin.close();

	ccutil::BinIO read(outputData.c_str(), outputData.size());
	string value;
	read >> value;
	printf("");
}

int main() {

	//testBinIO();
	testPlugin();
	//testSEQ();
	//testTaskPool();
	//testTaskPool();
	//testOnnxInfer();
	//testInt8Caffe();	
	return 0;
}
#endif