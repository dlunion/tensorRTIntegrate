
#ifndef ONNX_PLUGIN_HPP
#define ONNX_PLUGIN_HPP

#include <memory>
#include <NvInfer.h>
#include <vector>
#include <common/cc_util.hpp>
#include <common/trt_common.hpp>
#include <set>
#include <infer/trt_infer.hpp>
#include <NvInferRuntimeCommon.h>
#include <cuda_fp16.h>

namespace ONNXPlugin {

	enum Phase {
		CompilePhase,
		InferencePhase
	};

	struct GTensor {
		GTensor() {}
		GTensor(const TRTInfer::Tensor& tensor);
		GTensor(float* ptr, int n, int c, int h, int w);
		GTensor(TRTInfer::halfloat* ptr, int n, int c, int h, int w);
		int count(int start_axis = 0) const;
		inline int offset(int n = 0, int c = 0, int h = 0, int w = 0) const { return ((n * this->channel_ + c) * this->height_ + h) * this->width_ + w; }

		template<typename _T>
		inline _T* ptr() const { return (_T*)ptr_; }

		template<typename _T>
		inline _T* ptr(int n, int c = 0, int h = 0, int w = 0) const { return (_T*)ptr_ + offset(n, c, h, w); }

		inline float* ptr_float() const { return (float*)ptr_; }
		inline float* ptr_float(int n, int c = 0, int h = 0, int w = 0) const { return (float*)ptr_ + offset(n, c, h, w); }
		inline TRTInfer::halfloat* ptr_half() const { return (TRTInfer::halfloat*)ptr_; }
		inline TRTInfer::halfloat* ptr_half(int n, int c = 0, int h = 0, int w = 0) const { return (TRTInfer::halfloat*)ptr_ + offset(n, c, h, w); }

		int num_ = 0, channel_ = 0, height_ = 0, width_ = 0;
		void* ptr_ = nullptr;
		TRTInfer::DataType dtType_ = TRTInfer::DataType::dtFloat;
	};

	struct LayerConfig {

		///////////////////////////////////
		int nbOutput_ = 1;
		size_t workspaceSize_ = 0;
		std::set<nvinfer1::DataType> supportDataType_;
		std::set<nvinfer1::PluginFormat> supportPluginFormat_;

		std::vector<std::shared_ptr<TRTInfer::Tensor>> weights_;
		TRTInfer::DataType configDataType_;
		nvinfer1::PluginFormat configPluginFormat_;
		int configMaxbatchSize_ = 0;
		std::string info_;

		///////////////////////////////////
		std::vector<nvinfer1::Dims> input;
		std::vector<nvinfer1::Dims> output;
		std::string serializeData_;

		LayerConfig();
		void serialCopyTo(void* buffer);
		int serialize();
		void deserialize(const void* ptr, size_t length);
		void setup(const std::string& info, const std::vector<std::shared_ptr<TRTInfer::Tensor>>& weights);
		virtual void seril(ccutil::BinIO& out) {}
		virtual void deseril(ccutil::BinIO& in) {}
		virtual void init(){}
	};

	#define SetupPlugin(class_)			\
		virtual const char* getPluginType() const override{return #class_;};																		\
		virtual const char* getPluginVersion() const override{return "1";};																			\
		virtual nvinfer1::IPluginV2Ext* clone() const override{return new class_(*this);}

	#define RegisterPlugin(class_)		\
	class class_##PluginCreator__ : public nvinfer1::IPluginCreator{																				\
	public:																																			\
		const char* getPluginName() const override{return #class_;}																					\
		const char* getPluginVersion() const override{return "1";}																					\
		const nvinfer1::PluginFieldCollection* getFieldNames() override{return &mFieldCollection;}													\
																																					\
		nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override{									\
			auto plugin = new class_();																												\
			mFieldCollection = *fc;																													\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override{								\
			auto plugin = new class_();																												\
			plugin->pluginInit(name, serialData, serialLength);																						\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		void setPluginNamespace(const char* libNamespace) override{mNamespace = libNamespace;}														\
		const char* getPluginNamespace() const override{return mNamespace.c_str();}																	\
																																					\
	private:																																		\
		std::string mNamespace;																														\
		std::string mPluginName;																													\
		nvinfer1::PluginFieldCollection mFieldCollection{0, nullptr};																				\
	};																																				\
	REGISTER_TENSORRT_PLUGIN(class_##PluginCreator__);

	class TRTPlugin : public nvinfer1::IPluginV2Ext {
	public:
		virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override{return inputTypes[0];}
		virtual bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override{return false;}
		virtual bool canBroadcastInputAcrossBatch(int inputIndex) const override{return false;}

		virtual void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
									int nbOutputs, const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,
									const bool* inputIsBroadcast, const bool* outputIsBroadcast, nvinfer1::PluginFormat floatFormat, int maxBatchSize) override{
			this->configureWithFormat(inputDims, nbInputs, outputDims, nbOutputs, inputTypes[0], floatFormat, maxBatchSize);
		}

		virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) override {}
		virtual void detachFromContext() override {}
		virtual void setPluginNamespace(const char* pluginNamespace) override{this->namespace_ = pluginNamespace;};
		virtual const char* getPluginNamespace() const override{return this->namespace_.data();};

		virtual ~TRTPlugin();
		virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) = 0;
		virtual int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) = 0;

		void pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<TRTInfer::Tensor>>& weights);
		void pluginInit(const std::string& name, const void* serialData, size_t serialLength);

		virtual std::shared_ptr<LayerConfig> config(const std::string& layerName);
		virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const;
		virtual void configureWithFormat(
			const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
			int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize);
		virtual int getNbOutputs() const;
		virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims);
		virtual int initialize();
		virtual void terminate();
		virtual void destroy() override{}
		virtual size_t getWorkspaceSize(int maxBatchSize) const override;
		virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream);
		virtual size_t getSerializationSize() const override;
		virtual void serialize(void* buffer) const override;

	private:
		void mappingToGTensor();

	protected:
		std::string namespace_;
		std::string layerName_;
		Phase phase_ = CompilePhase;
		std::shared_ptr<LayerConfig> config_;
		std::vector<GTensor> inputTensors_;
		std::vector<GTensor> outputTensors_;
		std::vector<GTensor> weightTensors_;
	};

#define ExecuteKernel(numJobs, kernel, stream)		kernel<<<gridDims(numJobs), blockDims(numJobs), 0, stream>>>
}; //namespace Plugin

#endif //ONNX_PLUGIN_HPP