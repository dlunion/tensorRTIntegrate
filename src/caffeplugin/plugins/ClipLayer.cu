

#include "ClipLayer.hpp"
#include <common/json.hpp>

using namespace Plugin;

template<typename _T>
__global__ void clipKernel(const _T* in, _T* out, int inw, int inh, int outw, int outh, int edge)
{
	KERNEL_POSITION;

	int outHeightIndex = position;
	int selectOutInnerY = outHeightIndex % outh;
	int nc = outHeightIndex / outh;
	int inHeightIndex = nc * inh + selectOutInnerY;
	in += inHeightIndex * inw;
	out += outHeightIndex * outw;
	for (int i = 0; i < outw; ++i) 
		*out++ = *in++;
}

namespace Plugin {

	std::shared_ptr<LayerConfig> ClipLayer::config(const std::string& layerName) {
		auto cfg = TRTPlugin::config(layerName);
		cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
		return cfg;
	}

	int ClipLayer::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
		auto& data = inputs[0];
		auto& out = outputs[0];

		int edge = out.num_ * out.channel_ * out.height_;
		if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
			clipKernel<float> <<<gridDims(edge), blockDims(edge), 0, stream >>> (data.ptr_float(), out.ptr_float(), data.width_, data.height_, out.width_, out.height_, edge);
		}
		else if(config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
			clipKernel<TRTInfer::halfloat> <<<gridDims(edge), blockDims(edge), 0, stream >>> (data.ptr_half(), out.ptr_half(), data.width_, data.height_, out.width_, out.height_, edge);
		}
		return 0;
	}

	nvinfer1::Dims ClipLayer::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
		return nvinfer1::Dims3(inputDims[0].d[0], inputDims[0].d[1] - 1, inputDims[0].d[2] - 1);
	}
}

RegisterPlugin(ClipLayer);