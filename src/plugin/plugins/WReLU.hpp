
#ifndef WReLU_HPP
#define WReLU_HPP

#include <plugin/plugin.hpp>

using namespace Plugin;

class WReLU : public TRTPlugin {
public:
	//设置插件函数，通过宏，执行插件创建函数，同时执行模式匹配名字，用来对每个层的名字做模式匹配
	//该匹配方法用的是ccutil::patternMatch，请参照这个函数
	SETUP_PLUGIN(WReLU, "WReLU*");

	virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;

	//这个插件只有一个输出，输出的shape等于输入0的shape，因此返回input0的shape
	virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) override;

	//执行过程
	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
};

#endif //WReLU_HPP