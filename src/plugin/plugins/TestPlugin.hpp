
#ifndef WReLU_HPP
#define WReLU_HPP

#include <plugin/plugin.hpp>

using namespace Plugin;

class TestPlugin : public TRTPlugin {
public:
	SETUP_PLUGIN(TestPlugin, "TestPlugin*");

	virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;
	virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) override;
	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
};

#endif //WReLU_HPP