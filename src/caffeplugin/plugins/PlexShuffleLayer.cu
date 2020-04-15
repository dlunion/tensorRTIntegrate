

#include "PlexShuffleLayer.hpp"
#include <common/json.hpp>

using namespace Plugin;

template <typename _T>
__global__ void pixelShuffleKernel(
	const _T* bottom_data, _T* top_data, int input_c, int input_h,
	int input_w, int input_area, int output_h, int output_w, int output_area, int edge)
{
	KERNEL_POSITION;

	int input_c_index = position / input_area;
	int f_input_index = position % input_area;
	int input_row = f_input_index / input_w;
	int input_col = f_input_index % input_w;
	int output_c_index = input_c_index / 4;
	input_c_index = input_c_index % 4;
	int output_row = input_row * 2 + input_c_index / 2;
	int output_col = input_col * 2 + input_c_index % 2;
	int output_index = output_c_index * output_area + output_row * output_w + output_col;
	top_data[output_index] = bottom_data[position];
}

namespace Plugin {

	std::shared_ptr<LayerConfig> PlexShuffleLayer::config(const std::string& layerName){
		auto cfg = TRTPlugin::config(layerName);
		cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
		return cfg;
	}

	int PlexShuffleLayer::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
		auto& data = inputs[0];
		auto& out = outputs[0];

		int edge = data.count();
		if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
			pixelShuffleKernel<float> <<<gridDims(edge), blockDims(edge), 0, stream >>> (data.ptr_float(), out.ptr_float(), data.channel_, data.height_, data.width_,
				data.height_ * data.width_, out.height_, out.width_, out.height_ * out.width_, edge);
		}
		else {
			pixelShuffleKernel<TRTInfer::halfloat> <<<gridDims(edge), blockDims(edge), 0, stream >>> (data.ptr_half(), out.ptr_half(), data.channel_, data.height_, data.width_,
				data.height_ * data.width_, out.height_, out.width_, out.height_ * out.width_, edge);
		}
		return 0;
	}

	nvinfer1::Dims PlexShuffleLayer::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
		return nvinfer1::Dims3(inputDims[0].d[0] / 4, inputDims[0].d[1] *  2, inputDims[0].d[2] * 2);
	}
}

RegisterPlugin(PlexShuffleLayer);