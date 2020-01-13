
#include "plugin.hpp"
#include <string>
#include "plugin.hpp"

using namespace nvinfer1;
using namespace std;

namespace Plugin {

	GTensor::GTensor(float* ptr, int n, int c, int h, int w) {
		this->ptr_ = ptr;
		this->num_ = n;
		this->channel_ = c;
		this->height_ = h;
		this->width_ = w;
		this->dtType_ = TRTInfer::DataType::dtFloat;
	}

	GTensor::GTensor(TRTInfer::halfloat* ptr, int n, int c, int h, int w) {
		this->ptr_ = ptr;
		this->num_ = n;
		this->channel_ = c;
		this->height_ = h;
		this->width_ = w;
		this->dtType_ = TRTInfer::DataType::dtHalfloat;
	}

	GTensor::GTensor(const TRTInfer::Tensor& tensor) {
		this->ptr_ = (float*)tensor.gpu();
		this->num_ = tensor.num();
		this->channel_ = tensor.channel();
		this->height_ = tensor.height();
		this->width_ = tensor.width();
		this->dtType_ = TRTInfer::DataType::dtFloat;
	}

	int GTensor::count(int start_axis) const {
		int dims[] = {num_, channel_, height_, width_};
		start_axis = std::max(0, start_axis);
		start_axis = std::min(3, start_axis);

		int size = 1;
		for (int i = start_axis; i < 4; ++i)
			size *= dims[i];
		return size;
	}

	///////////////////////////////////
	LayerConfig::LayerConfig() {
		supportDataType_ = {nvinfer1::DataType::kFLOAT};
		supportPluginFormat_ = {nvinfer1::PluginFormat::kNCHW};
		configDataType_ = TRTInfer::DataType::dtFloat;
		configPluginFormat_ = nvinfer1::PluginFormat::kNCHW;
	}

	void LayerConfig::serialCopyTo(void* buffer) {
		if (!serializeData_.empty())
			memcpy(buffer, &serializeData_[0], serializeData_.size());
	}

	int LayerConfig::serialize() {

		ccutil::BinIO out;
		out << input;
		out << output;
		out << workspaceSize_;
		out << configDataType_;
		out << configPluginFormat_;
		out << configMaxbatchSize_;

		out << (int)weights_.size();
		for (int i = 0; i < weights_.size(); ++i) {

			if (configDataType_ == TRTInfer::DataType::dtFloat) {
				weights_[i]->toFloat();
			}
			else if (configDataType_ == TRTInfer::DataType::dtHalfloat) {
				weights_[i]->toHalf();
			}

			out << weights_[i]->dims();
			out << weights_[i]->type();
			out.write((char*)weights_[i]->cpu(), weights_[i]->bytes());
		}

		seril(out);
		serializeData_ = out.writedMemory();
		return serializeData_.size();
	}

	void LayerConfig::deserialize(const void* ptr, size_t length) {

		ccutil::BinIO in(ptr, length);
		in >> input;
		in >> output;
		in >> workspaceSize_;
		in >> configDataType_;
		in >> configPluginFormat_;
		in >> configMaxbatchSize_;

		int nbWeights = 0;
		in >> nbWeights;

		weights_.resize(nbWeights);
		for (int i = 0; i < nbWeights; ++i) {
			std::vector<int> dims;
			in >> dims;

			TRTInfer::DataType dt;
			in >> dt;

			weights_[i].reset(new TRTInfer::Tensor(dims, dt));
			in.read(weights_[i]->cpu(), weights_[i]->bytes());
			weights_[i]->gpu();
		}
		deseril(in);
	}

	void LayerConfig::loadWeights(const nvinfer1::Weights* weights, int nbWeights) {

		Assert(nbWeights == weights_.size());
		for (int i = 0; i < nbWeights; ++i) {
			//weights应该在config的时候就已经指定了，这里要求数量是一致的
			Assert(weights[i].type == nvinfer1::DataType::kFLOAT);	//这里要求权重类型必须是fp32，默认定义的
			Assert(weights_[i] != nullptr && weights_[i]->count() == weights[i].count);
			memcpy(weights_[i]->cpu(), weights[i].values, weights_[i]->bytes());
		}
	}

	///////////////////////////////////////////////////////////////////////////////////

	TRTPlugin::~TRTPlugin() {
	}

	void TRTPlugin::pluginInit(const std::string& name, const nvinfer1::Weights* weights, int nbWeights) {
		phase_ = CompilePhase;
		layerName_ = name;
		config_ = config(name);
		Assert(config_ != nullptr);
		config_->output.resize(config_->nbOutput_);
		config_->loadWeights(weights, nbWeights);
	}

	void TRTPlugin::pluginInit(const std::string& name, const void* serialData, size_t serialLength) {
		phase_ = InferencePhase;
		layerName_ = name;
		config_ = config(name);
		Assert(config_ != nullptr);
		config_->deserialize(serialData, serialLength);
	}

	std::shared_ptr<LayerConfig> TRTPlugin::config(const std::string& layerName) {
		return std::shared_ptr<LayerConfig>(new LayerConfig());
	}

	bool TRTPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
		INFO("supportsFormat %d, %d", type, format);
		return config_->supportDataType_.find(type) != config_->supportDataType_.end() &&
			config_->supportPluginFormat_.find(format) != config_->supportPluginFormat_.end();
	}

	void TRTPlugin::configureWithFormat(
		const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
		int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) {

		INFO("configureWithFormat: type: %d", type);
		if (type == nvinfer1::DataType::kFLOAT) {
			this->config_->configDataType_ = TRTInfer::DataType::dtFloat;
		}
		else if (type == nvinfer1::DataType::kHALF) {
			this->config_->configDataType_ = TRTInfer::DataType::dtHalfloat;
		}
		else {
			LOG(LFATAL) << "unsuport datatype: " << (int)type;
		}
		this->config_->configPluginFormat_ = format;
		this->config_->configMaxbatchSize_ = maxBatchSize;
	}

	int TRTPlugin::getNbOutputs() const {
		return config_->nbOutput_;
	}

	nvinfer1::Dims TRTPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {

		if (config_->input.empty()) {
			for (int i = 0; i < nbInputDims; ++i)
				config_->input.push_back(inputs[i]);
		}

		auto dims = outputDims(index, inputs, nbInputDims);
		config_->output[index] = dims;
		return dims;
	}

	void TRTPlugin::configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize) {
		//INFO("configure ");
	}

	int TRTPlugin::initialize() {
		return 0;
	}

	void TRTPlugin::terminate() {
	}

	size_t TRTPlugin::getWorkspaceSize(int maxBatchSize) const {
		return config_->workspaceSize_;
	}

	void TRTPlugin::mappingToGTensor() {
		if (inputTensors_.empty()) {
			inputTensors_.resize(config_->input.size());
			outputTensors_.resize(config_->output.size());
			weightTensors_.resize(config_->weights_.size());
			for (int i = 0; i < inputTensors_.size(); ++i) {
				auto& dims = config_->input[i];
				inputTensors_[i].num_ = 1;
				inputTensors_[i].channel_ = dims.nbDims > 0 ? dims.d[0] : 1;
				inputTensors_[i].height_ = dims.nbDims > 1 ? dims.d[1] : 1;
				inputTensors_[i].width_ = dims.nbDims > 2 ? dims.d[2] : 1;
			}

			for (int i = 0; i < outputTensors_.size(); ++i) {
				auto& dims = config_->output[i];
				outputTensors_[i].num_ = 1;
				outputTensors_[i].channel_ = dims.nbDims > 0 ? dims.d[0] : 1;
				outputTensors_[i].height_ = dims.nbDims > 1 ? dims.d[1] : 1;
				outputTensors_[i].width_ = dims.nbDims > 2 ? dims.d[2] : 1;
			}

			for (int i = 0; i < weightTensors_.size(); ++i) {
				auto& w = config_->weights_[i];
				weightTensors_[i].num_ = w->num();
				weightTensors_[i].channel_ = w->channel();
				weightTensors_[i].height_ = w->height();
				weightTensors_[i].width_ = w->width();
				weightTensors_[i].ptr_ = w->gpu();
				weightTensors_[i].dtType_ = w->type();
			}
		}
	}

	int TRTPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
		//INFO("enqueue");
		mappingToGTensor();

		for (int i = 0; i < inputTensors_.size(); ++i) {
			inputTensors_[i].num_ = batchSize;
			inputTensors_[i].ptr_ = (void*)inputs[i];
			inputTensors_[i].dtType_ = config_->configDataType_;
		}

		for (int i = 0; i < outputTensors_.size(); ++i) {
			outputTensors_[i].num_ = batchSize;
			outputTensors_[i].ptr_ = outputs[i];
			inputTensors_[i].dtType_ = config_->configDataType_;
		}
		return enqueue(inputTensors_, outputTensors_, weightTensors_, workspace, stream);
	}

	size_t TRTPlugin::getSerializationSize() {
		return config_->serialize();
	}

	void TRTPlugin::serialize(void* buffer) {
		config_->serialCopyTo(buffer);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class PluginRegistryImpl : public PluginRegistry {
	public:
		virtual void addPlugin(PluginCreater creater, const std::string& pattern) {
			plugins_.push_back({creater, pattern});
		}

		virtual PluginInfo* findPlugin(const std::string& layerName) {
			for (int i = 0; i < plugins_.size(); ++i) {
				if (ccutil::patternMatch(layerName.c_str(), plugins_[i].pattern.c_str()))
					return &plugins_[i];
			}
			return nullptr;
		}

	private:
		vector<PluginInfo> plugins_;
	};

	PluginRegistry* getPluginRegistry() {
		static shared_ptr<PluginRegistryImpl> instance;
		if (instance) return instance.get();
		
		instance.reset(new PluginRegistryImpl());
		return instance.get();
	}

	PluginRegister::PluginRegister(PluginCreater creater, const std::string& pattern) {
		getPluginRegistry()->addPlugin(creater, pattern);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//如果能找到插件，则说明支持他
	bool TRTBuilderPluginFactory::support(const std::string& layerName) {
		return getPluginRegistry()->findPlugin(layerName) != nullptr;
	}

	//根据layerName创建插件
	shared_ptr<TRTPlugin> TRTBuilderPluginFactory::createPlugin(const std::string& layerName) {

		//通过名字查找支持的插件信息
		auto pluginInfo = getPluginRegistry()->findPlugin(layerName);

		//如果找不到就报错，因为support的返回跟这个函数用的同一个，不应该找不到
		if (pluginInfo == nullptr) {
			INFO("layer '%s' unsupport.", layerName.c_str());
		}
		Assert(pluginInfo != nullptr);

		//执行创建并添加到记录表中去
		auto pluginInstance = pluginInfo->creater();
		plugins_.push_back(pluginInstance);
		return pluginInstance;
	}

	IPlugin* TRTBuilderPluginFactory::builderCreate(const std::string& layerName, const Weights* weights, int nbWeights) {
		INFO("builderCreate %s", layerName.c_str());

		auto instance = createPlugin(layerName);
		instance->pluginInit(layerName, weights, nbWeights);
		return instance.get();
	}

	IPlugin* TRTBuilderPluginFactory::inferCreate(const std::string& layerName, const void* serialData, size_t serialLength) {
		//INFO("inferCreate %s", layerName.c_str());

		auto instance = createPlugin(layerName);
		instance->pluginInit(layerName, serialData, serialLength);
		return instance.get();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	bool TRTBuilderPluginFactory::isPluginExt(const char* layerName) {
		return support(layerName);
	}

	bool TRTBuilderPluginFactory::isPlugin(const char* layerName) {
		return support(layerName);
	}

	IPlugin* TRTBuilderPluginFactory::createPlugin(const char* layerName, const Weights* weights, int nbWeights) {
		return builderCreate(layerName, weights, nbWeights);
	}

	IPlugin* TRTBuilderPluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
		return inferCreate(layerName, serialData, serialLength);
	}

	////////////////////////////////////////////////////////////////////////////////////////
	dim3 gridDims(int numJobs) {
		int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
		return dim3(ceil(numJobs / (float)numBlockThreads));
	}

	dim3 blockDims(int numJobs) {
		return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	std::shared_ptr<nvcaffeparser1::IPluginFactoryExt> createPluginFactoryForBuildPhase() {
		return std::shared_ptr<nvcaffeparser1::IPluginFactoryExt>(new TRTBuilderPluginFactory());
	}

	std::shared_ptr<nvinfer1::IPluginFactory> createPluginFactoryForInferPhase() {
		return std::shared_ptr<nvinfer1::IPluginFactory>(new TRTBuilderPluginFactory());
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	//编译FP16模型时：
	//isPlugin: conv1
	//isPlugin: batch_norm1
	//isPlugin: bn_scale1
	//...
	//isPlugin: MyReLU -> true
	//createPlugin: MyReLU
	//builderCreate: MyReLU
	//getNbOutputs
	//getNbOutputs
	//getNbOutputs
	//getOutputDimensions
	//Marked output blob 'myImage'.(3,1,1)
	//Set max batch size: 4.
	//Build engine
	//supportsFormat 0, 0
	//supportsFormat 1, 0
	//supportsFormat 1, 1
	//supportsFormat 1, 2
	//如果存在half和float，那么engine会执行enqueue来选择合适的（高效的）格式。例如1080Ti上没有
	//half的支持，因此他选择了float，调用了enqueue
	//configureWithFormat: type: 0
	//supportsFormat 0, 0
	//enqueue
	//enqueue
	//configureWithFormat : type: 1
	//supportsFormat 1, 0
	//enqueue
	//enqueue
	//configureWithFormat : type: 0
	//getWorkspaceSize
	//getWorkspaceSize
	//initialize
	//getSerializationSize
	//serialize
	//terminate
	//destroy TRTBuilderPluginFactory
	//destroy manager
	//destroy TRTPlugin


	//编译FP32模型时：
	//isPlugin: conv1
	//isPlugin: batch_norm1
	//isPlugin: bn_scale1
	//...
	//isPlugin: MyReLU -> true
	//createPlugin: MyReLU
	//builderCreate: MyReLU
	//getNbOutputs
	//getNbOutputs
	//getNbOutputs
	//getOutputDimensions
	//Marked output blob 'myImage'.(3,1,1)
	//Set max batch size: 4.
	//Build engine
	//supportsFormat 0, 0
	//supportsFormat 1, 0
	//supportsFormat 1, 1
	//supportsFormat 1, 2
	//configureWithFormat
	//getWorkspaceSize
	//getWorkspaceSize
	//initialize
	//getSerializationSize
	//serialize
	//terminate
	//destroy TRTBuilderPluginFactory
	//destroy manager
	//destroy TRTPlugin


	//编译INT8模型时：
	//isPlugin: conv1
	//isPlugin: batch_norm1
	//isPlugin: bn_scale1
	//...
	//isPlugin: MyReLU -> true
	//createPlugin: MyReLU
	//builderCreate: MyReLU
	//getNbOutputs
	//getNbOutputs
	//getNbOutputs
	//Marked output blob 'myImage'.(3,1,1)
	//Set max batch size: 4.
	//Build engine
	//getOutputDimensions
	//supportsFormat 0, 0
	//supportsFormat 1, 0
	//supportsFormat 1, 1
	//supportsFormat 1, 2
	//configureWithFormat
	//getWorkspaceSize
	//getWorkspaceSize
	//initialize
	//enqueue
	//enqueue
	//...
	//enqueue
	//enqueue
	//terminate
	//supportsFormat 0, 0
	//supportsFormat 1, 0
	//supportsFormat 1, 1
	//supportsFormat 1, 2
	//getWorkspaceSize
	//getWorkspaceSize
	//initialize
	//getSerializationSize
	//serialize
	//terminate
	//destroy TRTBuilderPluginFactory
	//destroy manager
	//destroy TRTPlugin


	//Inference的时候
	//createPlugin: MyReLU
	//inferCreate: MyReLU
	//initialize
	//enqueue
	//terminate
	//destroy TRTBuilderPluginFactory
	//destroy manager
	//destroy TRTPlugin
};// namespace Plugin