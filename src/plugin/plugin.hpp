
#ifndef PLUGIN_BASE_HPP
#define PLUGIN_BASE_HPP

#include <memory>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <vector>
#include <common/cc_util.hpp>
#include <cuda_runtime.h>
#include <set>
#include <infer/trt_infer.hpp>

namespace Plugin {

	//定义block内线程数，如果显卡性能较差，需要调整更低一些
#define GPU_BLOCK_THREADS  512
#define KERNEL_POSITION											\
	int position = (blockDim.x * blockIdx.x + threadIdx.x);		\
	if (position >= (edge)) return;

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
		//可以修改的配置
		int nbOutput_ = 1;
		size_t workspaceSize_ = 0;
		std::set<nvinfer1::DataType> supportDataType_;
		std::set<nvinfer1::PluginFormat> supportPluginFormat_;

		//config时请构造权重相关信息和配置，复制权重如果发现count不匹配时会报错
		std::vector<std::shared_ptr<TRTInfer::Tensor>> weights_;
		TRTInfer::DataType configDataType_;
		nvinfer1::PluginFormat configPluginFormat_;
		int configMaxbatchSize_ = 0;		

		///////////////////////////////////
		//自动赋值的配置
		std::vector<nvinfer1::Dims> input;
		std::vector<nvinfer1::Dims> output;
		std::string serializeData_;

		LayerConfig();
		void serialCopyTo(void* buffer);
		int serialize();
		void deserialize(const void* ptr, size_t length);
		void loadWeights(const nvinfer1::Weights* weights, int nbWeights);
		virtual void seril(ccutil::BinIO& out) {}
		virtual void deseril(ccutil::BinIO& in) {}
	};

	class TRTPlugin : public nvinfer1::IPluginExt {
	public:

		virtual ~TRTPlugin();
		virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) = 0;
		virtual int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) = 0;

		void pluginInit(const std::string& name, const nvinfer1::Weights* weights, int nbWeights);
		void pluginInit(const std::string& name, const void* serialData, size_t serialLength);

		//如果有权重，请对返回的config实例中的weights做初始化shape
		virtual std::shared_ptr<LayerConfig> config(const std::string& layerName);
		virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const;
		virtual void configureWithFormat(
			const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
			int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize);
		virtual int getNbOutputs() const;
		virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims);
		virtual void configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize);
		virtual int initialize();
		virtual void terminate();
		virtual size_t getWorkspaceSize(int maxBatchSize) const override;
		virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream);
		virtual size_t getSerializationSize();
		virtual void serialize(void* buffer);

	private:
		void mappingToGTensor();

	protected:
		std::string layerName_;
		Phase phase_ = CompilePhase;
		std::shared_ptr<LayerConfig> config_;
		std::vector<GTensor> inputTensors_;
		std::vector<GTensor> outputTensors_;
		std::vector<GTensor> weightTensors_;
	};

#define SETUP_PLUGIN(class_, pattern_)									  \
	static std::string pattern() {										  \
		return pattern_;												  \
	}																	  \
																		  \
	static std::shared_ptr<Plugin::TRTPlugin> creator() {					  \
		return std::shared_ptr<Plugin::TRTPlugin>(new class_());			  \
	}

#define RegisterPlugin(class_)	\
	static Plugin::PluginRegister __register##class_(class_::creator, class_::pattern())
	
	typedef std::shared_ptr<TRTPlugin>(*PluginCreater)();

	struct PluginInfo {
		PluginCreater creater;
		std::string pattern;
	};

	class PluginRegistry {
	public:
		virtual void addPlugin(PluginCreater creater, const std::string& pattern) = 0;
		virtual PluginInfo* findPlugin(const std::string& layerName) = 0;
	};

	class PluginRegister {
	public:
		PluginRegister(PluginCreater creater, const std::string& pattern);
	};

	class TRTBuilderPluginFactory : public nvcaffeparser1::IPluginFactoryExt, public nvinfer1::IPluginFactory {

	public:
		//trt的重载函数
		virtual bool isPluginExt(const char* layerName) override;
		virtual bool isPlugin(const char* layerName) override;
		virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
		virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;


		//如果能找到插件，则说明支持他
		virtual bool support(const std::string& layerName);

		//根据layerName创建插件
		virtual std::shared_ptr<TRTPlugin> createPlugin(const std::string& layerName);
		virtual nvinfer1::IPlugin* builderCreate(const std::string& layerName, const nvinfer1::Weights* weights, int nbWeights);
		virtual nvinfer1::IPlugin* inferCreate(const std::string& layerName, const void* serialData, size_t serialLength);

	private:
		std::vector<std::shared_ptr<nvinfer1::IPluginExt>> plugins_;
	};

	///////////////////////////////////////////////////////////////////////////////////////////
	std::shared_ptr<nvcaffeparser1::IPluginFactoryExt> createPluginFactoryForBuildPhase();
	std::shared_ptr<nvinfer1::IPluginFactory> createPluginFactoryForInferPhase();
	PluginRegistry* getPluginRegistry();

	dim3 gridDims(int numJobs);
	dim3 blockDims(int numJobs);

#define ExecuteKernel(numJobs, kernel, stream)		kernel<<<gridDims(numJobs), blockDims(numJobs), 0, stream>>>
}; //namespace Plugin

#endif //PLUGIN_BASE_HPP