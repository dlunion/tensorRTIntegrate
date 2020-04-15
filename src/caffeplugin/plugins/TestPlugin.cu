

#include "TestPlugin.hpp"

typedef TRTInfer::halfloat halfloat;

template<typename _T>
__global__ void MyPluginKenel(_T* input, _T* output, int edge);

template<>
__global__ void MyPluginKenel(float* input, float* output, int edge) {

	KERNEL_POSITION;
	output[position] = (input[position] < 0 ? 0 : input[position]) + 1.3f;
}

template<>
__global__ void MyPluginKenel(halfloat* input, halfloat* output, int edge) {

	KERNEL_POSITION;

	halfloat zero = 0.0f;
	halfloat add = 1.3f;
	output[position] = (input[position] < zero ? zero : input[position]) + add;
}

nvinfer1::Dims TestPlugin::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
	return inputDims[0];
}

std::shared_ptr<LayerConfig> TestPlugin::config(const std::string& layerName) {
	auto cfg = TRTPlugin::config(layerName);

	//定义我们这个插件支持half和float格式
	cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
	return cfg;
}

int TestPlugin::enqueue(const std::vector<Plugin::GTensor>& inputs, std::vector<Plugin::GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = gridDims(count);
	auto block = blockDims(count);

	if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
		MyPluginKenel <<<grid, block >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
	}
	else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
		MyPluginKenel <<<grid, block>>> (inputs[0].ptr<halfloat>(), outputs[0].ptr<halfloat>(), count);
	}
	return 0;
}

RegisterPlugin(TestPlugin);