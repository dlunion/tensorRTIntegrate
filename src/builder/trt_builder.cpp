
#include "trt_builder.hpp"

#include <cc_util.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include <NvOnnxParser.h>
#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <plugin/plugin.hpp>
#include <infer/trt_infer.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

#define cuCheck(op)	 Assert((op) == cudaSuccess)

static class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) {

		if (severity == Severity::kINTERNAL_ERROR) {
			INFO("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}else if (severity == Severity::kERROR) {
			INFO("NVInfer ERROR: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFO("NVInfer WARNING: %s", msg);
		}
		else {
			//INFO("%s", msg);
		}
	}
}gLogger;

namespace TRTBuilder {

	template<typename _T>
	static void destroyNV(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	bool caffeToTRTFP32OrFP16(const std::string& deployFile,                    // name for caffe prototxt
		const std::string& modelFile,                     // name for model
		const std::vector<std::string>& outputs,          // network outputs
		unsigned int maxBatchSize,                        // batch size - NB must be at least as large as the batch we want to run with)
		const string& savepath,
		const string& mode)
	{
		// create the builder
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFO("Can not create builder.");
			return false;
		}

		if (mode == "FP16") {
			if (!builder->platformHasFastFp16()) {
				INFO("Platform not have fast fp16 support, when mode=='%s'", mode.c_str());
				return false;
			}
		}

		// parse the caffe model to populate the network, then set the outputs
		shared_ptr<INetworkDefinition> network(builder->createNetwork(), destroyNV<INetworkDefinition>);
		if (!network) {
			INFO("Can not create network.");
			return false;
		}

		shared_ptr<ICaffeParser> parser(createCaffeParser(), destroyNV<ICaffeParser>);
		if (!parser) {
			INFO("Can not create caffe parser.");
			return false;
		}

		auto factory = Plugin::createPluginFactoryForBuildPhase();
		if (factory) {
			INFO("Using plugin factory for build phase.");
			parser->setPluginFactoryExt(factory.get());
		}

		auto blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT);
		if (blobNameToTensor == nullptr) {
			INFO("parse network fail.");
			return false;
		}

		for (auto& output : outputs) {
			auto blobMarked = blobNameToTensor->find(output.c_str());
			if (blobMarked == nullptr) {
				INFO("Can not found marked output '%s' in network.", output.c_str());
				return false;
			}

			auto dims = blobMarked->getDimensions();
			INFO("Marked output blob '%s'.(%d,%d,%d)", output.c_str(), dims.d[0], dims.d[1], dims.d[2]);
			network->markOutput(*blobMarked);
		}

		INFO("Set max batch size: %d.", maxBatchSize);
		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1 << 30);

		if (mode == "FP16") {
			INFO("Set FP16 mode.");
			builder->setFp16Mode(true);
		}

		INFO("Build engine");
		shared_ptr<ICudaEngine> engine(builder->buildCudaEngine(*network), destroyNV<ICudaEngine>);
		if (!engine) {
			INFO("build cuda engine fail.");
			return false;
		}

		shared_ptr<IHostMemory> seridata(engine->serialize(), destroyNV<IHostMemory>);
		return ccutil::savefile(savepath, seridata->data(), seridata->size());
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////INT8 Generator////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class Int8EntropyCalibrator : public IInt8EntropyCalibrator
	{
	public:

		Int8EntropyCalibrator(
			const vector<string>& imagefiles, cv::Size size, int channels, const Int8Process& preprocess) {

			Assert(channels == 1 || channels == 3);
			Assert(preprocess != nullptr);
			this->size_ = size;
			this->channels_ = channels;
			this->allimgs_ = imagefiles;
			this->preprocess_ = preprocess;
		}

		int getBatchSize() const {
			return 1;
		}

		bool next() {
			if (cursor_ >= allimgs_.size())
				return false;

			cv::Mat im = cv::imread(allimgs_[cursor_++], channels_ == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
			if (im.empty())
				return false;

			resize(im, im, size_);
			im.convertTo(batchOutput_, CV_32F);
			preprocess_(cursor_, allimgs_.size(), batchOutput_);

			//copy from image
			tensor_.from(batchOutput_.ptr<float>(0), 1, batchOutput_.rows, batchOutput_.cols, batchOutput_.channels());

			//transpose NHWC to NCHW format
			tensor_.transposeInplace(0, 3, 1, 2);
			return true;
		}

		bool getBatch(void* bindings[], const char* names[], int nbBindings) {
			if (!next()) return false;
			bindings[0] = tensor_.gpu();
			return true;
		}
		
		const void* readCalibrationCache(size_t& length) {length = 0; return nullptr;}
		virtual void writeCalibrationCache(const void* cache, size_t length) {}

	private:
		cv::Mat batchOutput_;
		int channels_ = 0;
		Int8Process preprocess_;
		vector<string> allimgs_;
		size_t batchCudaSize_ = 0;
		int cursor_ = 0;
		cv::Size size_;
		TRTInfer::Tensor tensor_;
	};

	bool caffeToTRT_INT8(
		const std::string& deployFile,
		const std::string& modelFile,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const std::string& savepath,
		const std::string& imageDirectory,
		const Int8Process& preprocess) {

		auto entropyCalibratorFiles = ccutil::findFiles(imageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
		if (entropyCalibratorFiles.empty()) {
			INFO("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", imageDirectory.c_str());
			return false;
		}

		// create the builder
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFO("Can not create builder.");
			return false;
		}

		if (!builder->platformHasFastInt8()) {
			INFO("Platform not have fast int8 support");
			return false;
		}

		// parse the caffe model to populate the network, then set the outputs
		shared_ptr<INetworkDefinition> network(builder->createNetwork(), destroyNV<INetworkDefinition>);
		if (!network) {
			INFO("Can not create network.");
			return false;
		}

		shared_ptr<ICaffeParser> parser(createCaffeParser(), destroyNV<ICaffeParser>);
		if (!parser) {
			INFO("Can not create caffe parser.");
			return false;
		}

		auto factory = Plugin::createPluginFactoryForBuildPhase();
		if (factory) {
			INFO("Using plugin factory for build phase.");
			parser->setPluginFactoryExt(factory.get());
		}

		auto blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT);
		if (blobNameToTensor == nullptr) {
			INFO("parse network fail.");
			return false;
		}

		for (auto& output : outputs) {
			auto blobMarked = blobNameToTensor->find(output.c_str());
			if (blobMarked == nullptr) {
				INFO("Can not found marked output '%s' in network.", output.c_str());
				return false;
			}

			INFO("Marked output blob '%s'.", output.c_str());
			network->markOutput(*blobMarked);
		}

		if (network->getNbInputs() > 1) {
			INFO("Warning: network has %d input, maybe have errors", network->getNbInputs());
		}

		auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();
		Assert(inputDims.nbDims == 3);

		int channel = inputDims.d[0];
		int height = inputDims.d[1];
		int width = inputDims.d[2];

		INFO("Set max batch size: %d.", maxBatchSize);
		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1 << 30);
		builder->setInt8Mode(true);

		shared_ptr<Int8EntropyCalibrator> int8Calibrator(new Int8EntropyCalibrator(
			entropyCalibratorFiles, cv::Size(width, height), channel, preprocess
		));
		builder->setInt8Calibrator(int8Calibrator.get());

		INFO("Build engine");
		shared_ptr<ICudaEngine> engine(builder->buildCudaEngine(*network), destroyNV<ICudaEngine>);
		if (!engine) {
			INFO("build cuda engine fail.");
			return false;
		}

		shared_ptr<IHostMemory> seridata(engine->serialize(), destroyNV<IHostMemory>);
		return ccutil::savefile(savepath, seridata->data(), seridata->size());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////ONNX////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool onnxToTRTFP32OrFP16(
		const std::string& modelFile,                     // name for model
		unsigned int maxBatchSize,                        // batch size - NB must be at least as large as the batch we want to run with)
		const string& savepath,
		const string& mode)
	{
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFO("Can not create builder.");
			return false;
		}

		if (mode == "FP16") {
			if (!builder->platformHasFastFp16()) {
				INFO("Platform not have fast fp16 support, when mode=='%s'", mode.c_str());
				return false;
			}
		}

		// parse the caffe model to populate the network, then set the outputs
		shared_ptr<INetworkDefinition> network(builder->createNetwork(), destroyNV<INetworkDefinition>);
		if (network == nullptr) {
			INFO("Can not create Network.");
			return false;
		}

		shared_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger), destroyNV<nvonnxparser::IParser>);
		if (parser == nullptr) {
			INFO("Can not create parser.");
			return false;
		}

		if (!parser->parseFromFile(modelFile.c_str(), 1)) {
			INFO("Can not parse OnnX file: %s", modelFile.c_str());
			return false;
		}

		INFO("Set max batch size: %d.", maxBatchSize);
		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1 << 30);

		if (mode == "FP16") {
			INFO("Set FP16 mode.");
			builder->setFp16Mode(true);
		}

		INFO("Build engine");
		shared_ptr<ICudaEngine> engine(builder->buildCudaEngine(*network), destroyNV<ICudaEngine>);
		if (engine == nullptr) {
			INFO("engine is nullptr");
			return false;
		}

		// serialize the engine, then close everything down
		shared_ptr<IHostMemory> seridata(engine->serialize(), destroyNV<IHostMemory>);
		return ccutil::savefile(savepath, seridata->data(), seridata->size());
	}

	void setDevice(int device_id) {
		cudaSetDevice(device_id);
	}
}; //namespace TRTBuilder