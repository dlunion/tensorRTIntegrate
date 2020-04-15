

#include "ChannelMultiplicationLayer.hpp"
#include <common/json.hpp>

using namespace Plugin;

template<typename _T>
__global__ void channelMultiplicationKernel(const _T* in, const _T* muld, _T* out, int input_area, int edge)
{
	KERNEL_POSITION;

	int c = position / input_area;
	out[position] = in[position] * muld[c];
}

namespace Plugin {

	std::shared_ptr<LayerConfig> ChannelMultiplicationLayer::config(const std::string& layerName) {
		auto cfg = TRTPlugin::config(layerName);
		cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
		return cfg;
	}

	int ChannelMultiplicationLayer::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
		auto& data = inputs[0];
		auto& mul = inputs[1];
		auto& out = outputs[0];
		int edge = data.count();

		if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
			channelMultiplicationKernel<float> <<<gridDims(edge), blockDims(edge), 0, stream >>> (data.ptr_float(), mul.ptr_float(), out.ptr_float(), data.height_ * data.width_, edge);
		}
		else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
			channelMultiplicationKernel<TRTInfer::halfloat> <<<gridDims(edge), blockDims(edge), 0, stream >>> (
				(const TRTInfer::halfloat*)data.ptr_half(), (const TRTInfer::halfloat*)mul.ptr_half(), out.ptr_half(), data.height_ * data.width_, edge
			);
		}
		return 0;
	}

	nvinfer1::Dims ChannelMultiplicationLayer::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
		return inputDims[0];
	}
}

RegisterPlugin(ChannelMultiplicationLayer);