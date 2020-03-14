
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

	const char* modeString(TRTMode type) {
		switch (type) {
		case TRTMode_FP32:
			return "FP32";
		case TRTMode_FP16:
			return "FP16";
		case TRTMode_INT8:
			return "INT8";
		default:
			return "UnknowTRTMode";
		}
	}

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
	class Int8EntropyCalibrator : public IInt8EntropyCalibrator
	{
	public:
		Int8EntropyCalibrator(const vector<string>& imagefiles, cv::Size size, int channels, const Int8Process& preprocess) {

			Assert(channels == 1 || channels == 3);
			Assert(preprocess != nullptr);
			this->size_ = size;
			this->channels_ = channels;
			this->allimgs_ = imagefiles;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = false;
		}

		Int8EntropyCalibrator(const string& entropyCalibratorData, cv::Size size, int channels, const Int8Process& preprocess) {

			Assert(channels == 1 || channels == 3);
			Assert(preprocess != nullptr);
			this->size_ = size;
			this->channels_ = channels;
			this->entropyCalibratorData_ = entropyCalibratorData;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = true;
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

		const string& getEntropyCalibratorData() {
			return entropyCalibratorData_;
		}

		const void* readCalibrationCache(size_t& length) {
			if (fromCalibratorData_) {
				length = this->entropyCalibratorData_.size();
				return this->entropyCalibratorData_.data();
			}

			length = 0;
			return nullptr;
		}

		virtual void writeCalibrationCache(const void* cache, size_t length) {
			entropyCalibratorData_.assign((char*)cache, length);
		}

	private:
		cv::Mat batchOutput_;
		int channels_ = 0;
		Int8Process preprocess_;
		vector<string> allimgs_;
		size_t batchCudaSize_ = 0;
		int cursor_ = 0;
		cv::Size size_;
		TRTInfer::Tensor tensor_;
		string entropyCalibratorData_;
		bool fromCalibratorData_ = false;
	};

	/////////////////////////////////////////////////////////////////////////////////////////
	bool compileTRT(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		Int8Process int8process,
		const std::string& int8ImageDirectory,
		const std::string& int8EntropyCalibratorFile) {

		bool hasEntropyCalibrator = false;
		string entropyCalibratorData;
		vector<string> entropyCalibratorFiles;

		INFO("Build %s trtmodel.", modeString(mode));

		if (!int8EntropyCalibratorFile.empty()) {
			if (ccutil::exists(int8EntropyCalibratorFile)) {
				entropyCalibratorData = ccutil::loadfile(int8EntropyCalibratorFile);
				if (entropyCalibratorData.empty()) {
					INFO("entropyCalibratorFile is set as: %s, but we read is empty.", int8EntropyCalibratorFile.c_str());
					return false;
				}
				hasEntropyCalibrator = true;
			}
		}
		
		if (mode == TRTMode_INT8) {
			if (hasEntropyCalibrator) {
				if (!int8ImageDirectory.empty()) {
					INFO("imageDirectory is ignore, when entropyCalibratorFile is set");
				}
			}
			else {
				if (int8process == nullptr) {
					INFO("int8process must be set. when Mode is '%s'", modeString(mode));
					return false;
				}

				entropyCalibratorFiles = ccutil::findFiles(int8ImageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
				if (entropyCalibratorFiles.empty()) {
					INFO("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", int8ImageDirectory.c_str());
					return false;
				}
			}
		}
		else {
			if (hasEntropyCalibrator) {
				INFO("int8EntropyCalibratorFile is ignore, when Mode is '%s'", modeString(mode));
			}
		}

		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFO("Can not create builder.");
			return false;
		}

		if (mode == TRTMode_INT8) {
			if (!builder->platformHasFastInt8()) {
				INFO("Platform not have fast int8 support");
				return false;
			}
			builder->setInt8Mode(true);
		}
		else if (mode == TRTMode_FP16) {
			if (!builder->platformHasFastFp16()) {
				INFO("Platform not have fast fp16 support");
				//return false;
			}
			builder->setFp16Mode(true);
		}

		shared_ptr<INetworkDefinition> network(builder->createNetwork(), destroyNV<INetworkDefinition>);
		if (!network) {
			INFO("Can not create network.");
			return false;
		}

		shared_ptr<ICaffeParser> caffeParser;
		shared_ptr<nvonnxparser::IParser> onnxParser;

		std::shared_ptr<nvcaffeparser1::IPluginFactoryExt> pluginFactory;
		if (source.type() == ModelSourceType_FromCaffe) {

			caffeParser.reset(createCaffeParser(), destroyNV<ICaffeParser>);
			if (!caffeParser) {
				INFO("Can not create caffe parser.");
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

			INFO("Warning: ONNX has no pluginFactory support");

			//from onnx is not markOutput
			onnxParser.reset(nvonnxparser::createParser(*network, gLogger), destroyNV<nvonnxparser::IParser>);
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
		Assert(inputDims.nbDims == 3);

		int channel = inputDims.d[0];
		int height = inputDims.d[1];
		int width = inputDims.d[2];

		INFO("Set max batch size: %d.", maxBatchSize);
		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1 << 30);
		
		shared_ptr<Int8EntropyCalibrator> int8Calibrator;
		if (mode == TRTMode_INT8) {
			if (hasEntropyCalibrator) {
				INFO("Using exist entropy calibrator data[%d bytes]: %s", entropyCalibratorData.size(), int8EntropyCalibratorFile.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorData, cv::Size(width, height), channel, int8process
				));
			}
			else {
				INFO("Using image list[%d files]: %s", entropyCalibratorFiles.size(), int8ImageDirectory.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorFiles, cv::Size(width, height), channel, int8process
				));
			}
			builder->setInt8Calibrator(int8Calibrator.get());
		}

		INFO("Build engine");
		shared_ptr<ICudaEngine> engine(builder->buildCudaEngine(*network), destroyNV<ICudaEngine>);
		if (engine == nullptr) {
			INFO("engine is nullptr");
			return false;
		}

		if (mode == TRTMode_INT8) {
			if (!hasEntropyCalibrator) {
				if (!int8EntropyCalibratorFile.empty()) {
					INFO("Save calibrator to: %s", int8EntropyCalibratorFile.c_str());
					ccutil::savefile(int8EntropyCalibratorFile, int8Calibrator->getEntropyCalibratorData());
				}
				else {
					INFO("No set entropyCalibratorFile, and entropyCalibrator will not save.");
				}
			}
		}

		// serialize the engine, then close everything down
		shared_ptr<IHostMemory> seridata(engine->serialize(), destroyNV<IHostMemory>);
		return ccutil::savefile(savepath, seridata->data(), seridata->size());
	}

	void setDevice(int device_id) {
		cudaSetDevice(device_id);
	}
}; //namespace TRTBuilder