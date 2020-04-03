

#include "MReLU.hpp"
#include <json.hpp>

typedef TRTInfer::halfloat halfloat;

template<typename _T>
__global__ void MReLUKernel(_T* input, _T* output, _T bias, int edge);


template<>
__global__ void MReLUKernel(float* input, float* output, float bias, int edge) {

    KERNEL_POSITION;
    float x = input[position];
    float a = x > 0 ? x : 0;
	output[position] = a + bias;
}

template<>
__global__ void MReLUKernel(halfloat* input, halfloat* output, halfloat bias, int edge) {

	KERNEL_POSITION;

	halfloat x = input[position];
    halfloat _zero = 0.0f;
    x = x > _zero ? x : _zero;
	output[position] = x + bias;
}

void MReLUConfig::init(){
    INFO("init MReLU config: %s", info_.c_str());
	INFO("MReLU weights = %d[%s]", this->weights_.size(), this->weights_[0]->shapeString());
	
	Json::Value value;
	if(Json::Reader().parse(info_, value)){
		INFO("MReLU kernel_size: %d", value["kernel_size"].asInt());
		INFO("MReLU eps: %g", value["eps"].asFloat());
		INFO("MReLU other: %s", value["other"].asCString());
	}
}

nvinfer1::Dims MReLU::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
	return inputDims[0];
}

std::shared_ptr<LayerConfig> MReLU::config(const std::string& layerName) {
	auto cfg = std::shared_ptr<LayerConfig>(new MReLUConfig());

	//定义我们这个插件支持half和float格式
	cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
	return cfg;
}

int MReLU::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = gridDims(count);
	auto block = blockDims(count);
	float bias = *this->config_->weights_[0]->cpu<float>();

	if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
		MReLUKernel <<<grid, block >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), bias, count);
	}
	else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
		MReLUKernel <<<grid, block>>> (inputs[0].ptr<halfloat>(), outputs[0].ptr<halfloat>(), halfloat(bias), count);
	}
	return 0;
}

RegisterPlugin(MReLU);