
#ifndef DCNLayer_HPP
#define DCNLayer_HPP

#include <plugin/plugin.hpp>
#include <cublas_v2.h>

namespace Plugin {
	class DCNLayer : public TRTPlugin {
	public:
		//设置插件函数，通过宏，执行插件创建函数，同时执行模式匹配名字，用来对每个层的名字做模式匹配
		//该匹配方法用的是ccutil::patternMatch，请参照这个函数
		SETUP_PLUGIN(DCNLayer, "DCN*");

		DCNLayer();
		virtual ~DCNLayer();

		virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;

		//这个插件只有一个输出，输出的shape等于输入0的shape，因此返回input0的shape
		virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims);
		virtual size_t getWorkspaceSize(int maxBatchSize) const override;

		//执行过程
		virtual int enqueue(const std::vector<Plugin::GTensor>& inputs, std::vector<Plugin::GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
	private:
		cublasHandle_t cublasHandle_ = nullptr;
	};
}

#endif //DCNLayer_HPP