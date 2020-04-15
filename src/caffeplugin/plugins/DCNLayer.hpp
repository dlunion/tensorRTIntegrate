
#ifndef DCNLayer_HPP
#define DCNLayer_HPP

#include <caffeplugin/caffeplugin.hpp>
#include <cublas_v2.h>

namespace Plugin {
	class DCNLayer : public TRTPlugin {
	public:
		SETUP_PLUGIN(DCNLayer, "DCN*");

		DCNLayer();
		virtual ~DCNLayer();

		virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;

		virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims);
		virtual size_t getWorkspaceSize(int maxBatchSize) const override;
		virtual int enqueue(const std::vector<Plugin::GTensor>& inputs, std::vector<Plugin::GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
	private:
		cublasHandle_t cublasHandle_ = nullptr;
	};
}

#endif //DCNLayer_HPP