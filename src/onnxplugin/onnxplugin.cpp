
#include "onnxplugin.hpp"
#include <string>

using namespace nvinfer1;
using namespace std;

namespace ONNXPlugin {

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
		out << info_;

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
		in >> info_;

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

	void LayerConfig::setup(const std::string& info, const std::vector<std::shared_ptr<TRTInfer::Tensor>>& weights) {

		this->info_ = info;
		this->weights_ = weights;
	}

	///////////////////////////////////////////////////////////////////////////////////

	TRTPlugin::~TRTPlugin() {
	}

	void TRTPlugin::pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<TRTInfer::Tensor>>& weights) {
		phase_ = CompilePhase;
		layerName_ = name;
		config_ = this->config(name);
		Assert(config_ != nullptr);
		config_->output.resize(config_->nbOutput_);
		config_->setup(info, weights);
		config_->init();
	}

	void TRTPlugin::pluginInit(const std::string& name, const void* serialData, size_t serialLength) {
		phase_ = InferencePhase;
		layerName_ = name;
		config_ = this->config(name);
		Assert(config_ != nullptr);
		config_->deserialize(serialData, serialLength);
		config_->init();
	}

	std::shared_ptr<LayerConfig> TRTPlugin::config(const std::string& layerName) {
		return std::shared_ptr<LayerConfig>(new LayerConfig());
	}

	bool TRTPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
		//INFO("supportsFormat %d, %d", type, format);
		return config_->supportDataType_.find(type) != config_->supportDataType_.end() &&
			config_->supportPluginFormat_.find(format) != config_->supportPluginFormat_.end();
	}

	void TRTPlugin::configureWithFormat(
		const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
		int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) {

		//INFO("configureWithFormat: type: %d", type);
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

	size_t TRTPlugin::getSerializationSize() const{
		return config_->serialize();
	}

	void TRTPlugin::serialize(void* buffer) const {
		config_->serialCopyTo(buffer);
	}
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


	//createPlugin: MyReLU
	//inferCreate: MyReLU
	//initialize
	//enqueue
	//terminate
	//destroy TRTBuilderPluginFactory
	//destroy manager
	//destroy TRTPlugin
};// namespace Plugin