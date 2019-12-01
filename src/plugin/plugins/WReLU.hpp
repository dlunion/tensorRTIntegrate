
#ifndef WReLU_HPP
#define WReLU_HPP

#include <plugin/plugin.hpp>

class WReLU : public Plugin::TRTPlugin {
public:
	SETUP_PLUGIN(WReLU, "WReLU*");

	virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
		auto ptr = new WReLU();
		return inputDims[0];
	}

	int enqueue(const std::vector<Plugin::GTensor>& inputs, std::vector<Plugin::GTensor>& outputs, cudaStream_t stream);
};

#endif //WReLU_HPP