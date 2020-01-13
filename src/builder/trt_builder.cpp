
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

	/////////////////////////////////////////////////////////////////////////////
	//定义释放函数模版，因为nv的东西全都是ptr->destroy的方式实现的，每个手动调用有点烦
	template<typename _T>
	static void destroyNV(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	//定义mode的string形式
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

	//定义模型的来源构造函数，不同的构造表示了不同的来源
	ModelSource::ModelSource(const std::string& prototxt, const std::string& caffemodel) {
		this->type_ = ModelSourceType_FromCaffe;
		this->prototxt_ = prototxt;
		this->caffemodel_ = caffemodel;
	}

	ModelSource::ModelSource(const std::string& onnxmodel) {
		this->type_ = ModelSourceType_FromONNX;
		this->onnxmodel_ = onnxmodel;
	}

	//定义属性获取的函数
	std::string ModelSource::prototxt() const { return this->prototxt_; }
	std::string ModelSource::caffemodel() const { return this->caffemodel_; }
	std::string ModelSource::onnxmodel() const { return this->onnxmodel_; }
	ModelSourceType ModelSource::type() const { return this->type_; }

	/////////////////////////////////////////////////////////////////////////////////////////
	////int8模式下的标定类
	class Int8EntropyCalibrator : public IInt8EntropyCalibrator
	{
	public:

		//如果是从图像列表中来标定，则以这个构造，提供图像列表
		//其中的preprocess是每个图的处理方法函数，调用该函数时，图像已经被归一化（resize，转换到CV_32F）
		Int8EntropyCalibrator(const vector<string>& imagefiles, cv::Size size, int channels, const Int8Process& preprocess) {

			//这里支持1或者3通道，其他通道暂时没考虑
			Assert(channels == 1 || channels == 3);
			Assert(preprocess != nullptr);
			this->size_ = size;
			this->channels_ = channels;
			this->allimgs_ = imagefiles;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = false;
		}

		//如果是从已经标定的数据中来，则以这个构造，例如从服务器上生成好calibrator文件，发送给嵌入式机器生成trtmodel时不需要图像
		//其中的preprocess是每个图的处理方法函数，调用该函数时，图像已经被归一化（resize，转换到CV_32F）
		Int8EntropyCalibrator(const string& entropyCalibratorData, cv::Size size, int channels, const Int8Process& preprocess) {

			Assert(channels == 1 || channels == 3);
			Assert(preprocess != nullptr);
			this->size_ = size;
			this->channels_ = channels;
			this->entropyCalibratorData_ = entropyCalibratorData;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = true;
		}

		//这里的batchsize是标定时指定的batch size，目前测试下来是没有区别，所以给默认值1，如果要修改，请保证tensor_的维度和getBatch的返回需要修改匹配
		//这个函数是继承自IInt8EntropyCalibrator，为了保证兼容，所以没有加关键字override
		int getBatchSize() const {
			return 1;
		}

		bool next() {
			if (cursor_ >= allimgs_.size())
				return false;

			cv::Mat im = cv::imread(allimgs_[cursor_++], channels_ == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
			if (im.empty())
				return false;

			//对图像归一化后，调用自定义的预处理函数
			resize(im, im, size_);
			im.convertTo(batchOutput_, CV_32F);
			preprocess_(cursor_, allimgs_.size(), batchOutput_);

			//copy from image
			tensor_.from(batchOutput_.ptr<float>(0), 1, batchOutput_.rows, batchOutput_.cols, batchOutput_.channels());

			//transpose NHWC to NCHW format
			tensor_.transposeInplace(0, 3, 1, 2);
			return true;
		}

		//这个函数是继承自IInt8EntropyCalibrator
		bool getBatch(void* bindings[], const char* names[], int nbBindings) {
			if (!next()) return false;
			bindings[0] = tensor_.gpu();
			return true;
		}

		const string& getEntropyCalibratorData() {
			return entropyCalibratorData_;
		}

		//这个函数是继承自IInt8EntropyCalibrator
		const void* readCalibrationCache(size_t& length) {
			if (fromCalibratorData_) {
				length = this->entropyCalibratorData_.size();
				return this->entropyCalibratorData_.data();
			}

			length = 0;
			return nullptr;
		}

		//这个函数是继承自IInt8EntropyCalibrator
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
	//编译trt模型，一个函数解决所有问题，便于测试和使用
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

		//如果提供了标定文件路径
		if (!int8EntropyCalibratorFile.empty()) {

			//如果标定文件存在
			if (ccutil::exists(int8EntropyCalibratorFile)) {

				//加载标定文件数据
				entropyCalibratorData = ccutil::loadfile(int8EntropyCalibratorFile);

				//如果加载失败或者为空则属于错误
				if (entropyCalibratorData.empty()) {
					INFO("entropyCalibratorFile is set as: %s, but we read is empty.", int8EntropyCalibratorFile.c_str());
					return false;
				}
				hasEntropyCalibrator = true;
			}
		}
		
		//对于int8模型的判断
		if (mode == TRTMode_INT8) {

			//如果提供了标定数据并且也提供了图像目录，则提示
			if (hasEntropyCalibrator) {
				if (!int8ImageDirectory.empty()) {
					INFO("imageDirectory is ignore, when entropyCalibratorFile is set");
				}
			}
			else {
				//如果没有提供标定数据，那么需要生成，则int8处理的函数必须提供，否则无法量化
				if (int8process == nullptr) {
					INFO("int8process must be set. when Mode is '%s'", modeString(mode));
					return false;
				}

				//如果查找到的图像数据为空也无法完成量化，所以必须提供
				entropyCalibratorFiles = ccutil::findFiles(int8ImageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
				if (entropyCalibratorFiles.empty()) {
					INFO("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", int8ImageDirectory.c_str());
					return false;
				}
			}
		}
		else {
			//如果不是int8还提供了标定数据，就提示他，是忽略的
			if (hasEntropyCalibrator) {
				INFO("int8EntropyCalibratorFile is ignore, when Mode is '%s'", modeString(mode));
			}
		}

		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroyNV<IBuilder>);
		if (builder == nullptr) {
			INFO("Can not create builder.");
			return false;
		}

		//对于是否支持做判断和提示
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

		//构建网络
		shared_ptr<INetworkDefinition> network(builder->createNetwork(), destroyNV<INetworkDefinition>);
		if (!network) {
			INFO("Can not create network.");
			return false;
		}

		//如果是源自caffe的，则使用caffe的解析器，这里parser必须在engine build的时候依然存在，否则会报nvinfer.dll内部错误
		//因此我们没有把caffeParser放到if内
		shared_ptr<ICaffeParser> caffeParser;
		shared_ptr<nvonnxparser::IParser> onnxParser;

		std::shared_ptr<nvcaffeparser1::IPluginFactoryExt> pluginFactory;
		if (source.type() == ModelSourceType_FromCaffe) {

			caffeParser.reset(createCaffeParser(), destroyNV<ICaffeParser>);
			if (!caffeParser) {
				INFO("Can not create caffe parser.");
				return false;
			}

			//对于是否存在插件，如果有则设置
			pluginFactory = Plugin::createPluginFactoryForBuildPhase();
			if (pluginFactory) {
				INFO("Using plugin factory for build phase.");
				caffeParser->setPluginFactoryExt(pluginFactory.get());
			}

			//解析网络，得到blob到tensor的映射
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

			//onnx没有设置markoutput的接口
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

		//如果是int8模式
		if (mode == TRTMode_INT8) {
			
			//如果没有提供标定数据
			if (!hasEntropyCalibrator) {

				//如果指定了标定文件，则认为是存储到指定文件
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