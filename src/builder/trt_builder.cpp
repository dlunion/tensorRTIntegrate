
#include "trt_builder.hpp"

#include <cc_util.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include <onnx_parser/NvOnnxParser.h>
#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <caffeplugin/caffeplugin.hpp>
#include <infer/trt_infer.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

#define cuCheck(op)	 Assert((op) == cudaSuccess)

static class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) {

		if (severity == Severity::kINTERNAL_ERROR) {
			INFOE("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}else if (severity == Severity::kERROR) {
			INFOE("NVInfer ERROR: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFOW("NVInfer WARNING: %s", msg);
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

	const char* modeString(TRTMode type) {
		switch (type) {
		case TRTMode_FP32:
			return "FP32";
		case TRTMode_FP16:
			return "FP16";
		default:
			return "UnknowTRTMode";
		}
	}

	InputDims::InputDims(int channels, int height, int width){
		this->channels_ = channels;
		this->height_ = height;
		this->width_ = width;
	}

	int InputDims::channels() const	{return this->channels_;}
	int InputDims::height() const	{return this->height_;}
	int InputDims::width() const	{return this->width_;}

	ModelSource::ModelSource(const std::string& prototxt, const std::string& caffemodel) {
		this->type_ = ModelSourceType_FromCaffe;
		this->prototxt_ = prototxt;
		this->caffemodel_ = caffemodel;
	}

	ModelSource::ModelSource(const std::string& onnxmodel) {
		this->type_ = ModelSourceType_FromONNX;
		this->onnxmodel_ = onnxmodel;
	}

	std::string ModelSource::prototxt() const { return this->prototxt_; }
	std::string ModelSource::caffemodel() const { return this->caffemodel_; }
	std::string ModelSource::onnxmodel() const { return this->onnxmodel_; }
	ModelSourceType ModelSource::type() const { return this->type_; }

	/////////////////////////////////////////////////////////////////////////////////////////
	bool compileTRT(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		std::vector<InputDims> inputsDimsSetup) {

		INFOW("Build %s trtmodel.", modeString(mode));
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFOE("Can not create builder.");
			return false;
		}

		shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroyNV<IBuilderConfig>);
		if (mode == TRTMode_FP16) {
			if (!builder->platformHasFastFp16()) {
				INFOW("Platform not have fast fp16 support");
			}
			config->setFlag(BuilderFlag::kFP16);
		}

		shared_ptr<INetworkDefinition> network;
		shared_ptr<ICaffeParser> caffeParser;
		shared_ptr<nvonnxparser::IParser> onnxParser;

		std::shared_ptr<nvcaffeparser1::IPluginFactoryExt> pluginFactory;
		if (source.type() == ModelSourceType_FromCaffe) {

			network = shared_ptr<INetworkDefinition>(builder->createNetwork(), destroyNV<INetworkDefinition>);
			caffeParser.reset(createCaffeParser(), destroyNV<ICaffeParser>);
			if (!caffeParser) {
				INFOW("Can not create caffe parser.");
				return false;
			}

			pluginFactory = Plugin::createPluginFactoryForBuildPhase();
			if (pluginFactory) {
				INFO("Using plugin factory for build phase.");
				caffeParser->setPluginFactoryExt(pluginFactory.get());
			}

			auto blobNameToTensor = caffeParser->parse(source.prototxt().c_str(), source.caffemodel().c_str(), *network, nvinfer1::DataType::kFLOAT);
			if (blobNameToTensor == nullptr) {
				INFO("parse network fail, prototxt: %s, caffemodel: %s", source.prototxt().c_str(), source.caffemodel().c_str());
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
		}
		else if(source.type() == ModelSourceType_FromONNX){

			//const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
			//network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroyNV<INetworkDefinition>);
			network = shared_ptr<INetworkDefinition>(builder->createNetwork(), destroyNV<INetworkDefinition>);

			vector<nvinfer1::Dims> dimsSetup;
			for(int i = 0; i < inputsDimsSetup.size(); ++i){
				auto& item = inputsDimsSetup[i];
				dimsSetup.push_back(nvinfer1::Dims3(item.channels(), item.height(), item.width()));
			}
			
			//from onnx is not markOutput
			onnxParser.reset(nvonnxparser::createParser(*network, dimsSetup, gLogger), destroyNV<nvonnxparser::IParser>);
			if (onnxParser == nullptr) {
				INFO("Can not create parser.");
				return false;
			}

			if (!onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)) {
				INFO("Can not parse OnnX file: %s", source.onnxmodel().c_str());
				return false;
			}
		}
		else {
			INFO("not implementation source type: %d", source.type());
			Assert(false);
		}

		auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();
		int channel = 0;
		int height = 0;
		int width = 0;

		if(inputDims.nbDims == 3){
			channel = inputDims.d[0];
			height = inputDims.d[1];
			width = inputDims.d[2];
		}else if(inputDims.nbDims == 4){
			channel = inputDims.d[1];
			height = inputDims.d[2];
			width = inputDims.d[3];
		}else{
			LOG(LFATAL) << "unsupport inputDims.nbDims " << inputDims.nbDims;
		}

		size_t _1_GB = 1 << 30;
		INFO("input shape: %d x %d x %d", channel, height, width);
		INFOW("Set max batch size: %d", maxBatchSize);
		
		builder->setMaxBatchSize(maxBatchSize);
		config->setMaxWorkspaceSize(_1_GB);

		INFOW("Build engine");
		shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroyNV<ICudaEngine>);
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