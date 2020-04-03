
#ifndef DCNv2_HPP
#define DCNv2_HPP

#include <onnxplugin/onnxplugin.hpp>
#include <cublas_v2.h>

using namespace ONNXPlugin;

class DCNv2 : public TRTPlugin {
public:
	SetupPlugin(DCNv2);

	virtual int initialize() override;
	virtual void terminate() override;
	virtual std::shared_ptr<LayerConfig> config(const std::string& layerName) override;
	virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims);
	virtual size_t getWorkspaceSize(int maxBatchSize) const override;
	virtual int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override;
private:
	cublasHandle_t cublasHandle_ = nullptr;
};

#endif //DCNv2_HPP