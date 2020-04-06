
#ifndef WReLU_HPP
#define WReLU_HPP

#include <onnxplugin/onnxplugin.hpp>

using namespace ONNXPlugin;

class HSigmoidConfig : public LayerConfig{
public:
	virtual void init() override;
};

class HSigmoid : public TRTPlugin {
public:
	SetupPlugin(HSigmoid);

	virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;

	//这个插件只有一个输出，输出的shape等于输入0的shape，因此返回input0的shape
	virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) override;

	//执行过程
	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
};

#endif //WReLU_HPP