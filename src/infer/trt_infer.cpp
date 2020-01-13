

#include "trt_infer.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cc_util.hpp>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <NvInferPlugin.h>

#if defined(HAS_PLUGIN)
#include <plugin/plugin.hpp>
#endif

using namespace nvinfer1;
using namespace std;

#define cuCheck(op)														 \
do {																	 \
	auto ret = (op);													 \
	if (ret != cudaSuccess) {											 \
		LOG(LFATAL) << #op << " fail, " << (int)ret << " != " << (int)cudaSuccess << ", " << cudaGetErrorString(ret);				 \
	}																	 \
} while (0);

static class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) {

		if (severity == Severity::kINTERNAL_ERROR) {
			INFO("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}
		else if (severity == Severity::kERROR) {
			INFO("NVInfer ERROR: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFO("NVInfer WARNING: %s", msg);
		}
	}
}gLogger;

namespace TRTInfer {

	void setDevice(int device_id) {
		cudaSetDevice(device_id);
	}

	bool initNVPlugins() {
		bool ok = initLibNvInferPlugins(&gLogger, "");
		if (!ok) {
			INFO("init lib nvinfer plugins fail.");
		}
		return ok;
	}

	Tensor::Tensor(int n, int c, int h, int w, DataType dtType) {
		this->dtType_ = dtType;
		resize(n, c, h, w);
	}

	Tensor::Tensor(const std::vector<int>& dims, DataType dtType):Tensor(dims.size(), dims.data(), dtType){}

	Tensor::Tensor(int ndims, const int* dims, DataType dtType) {

		//最多支持4维的构建
		Assert(ndims <= 4 && ndims > 0);

		int n = ndims > 0 ? dims[0] : 1;
		int c = ndims > 1 ? dims[1] : 1;
		int h = ndims > 2 ? dims[2] : 1;
		int w = ndims > 3 ? dims[3] : 1;
		this->dtType_ = dtType;
		resize(n, c, h, w);
	}

	Tensor::Tensor(){}

	void Tensor::release() {
		if (host_) {
			free(host_);
			host_ = nullptr;  //host为nullptr和其他地址有什么区别
		}

		if (device_) {
			cuCheck(cudaFree(device_));
			device_ = nullptr;
		}
		num_ = channel_ = height_ = width_ = 0;
		capacity_ = 0;
		bytes_ = 0;
	}

	bool Tensor::empty() {
		return host_ == nullptr;
	}

	size_t Tensor::count(int start_axis) const {

		int dims[] = { num_, channel_, height_, width_};
		start_axis = std::max(0, start_axis);
		start_axis = std::min(3, start_axis);

		int size = 1;
		for (int i = start_axis; i < 4; ++i) 
			size *= dims[i];
		return size;
	}

	void Tensor::reshape(int n, int c, int h, int w) {

		n = n == -1 ? this->num_ : n;
		c = c == -1 ? this->channel_ : c;
		h = h == -1 ? this->height_ : h;
		w = w == -1 ? this->width_ : w;

		Assert(this->count() == n * c * h * w);
		this->num_ = n;
		this->channel_ = c;
		this->height_ = h;
		this->width_ = w;
	}

	Tensor::Tensor(const Tensor& other) {
		this->operator=(other);
	}

	void Tensor::resizeLike(const Tensor& other) {
		this->resize(other.num_, other.channel_, other.height_, other.width_);
	}

	void Tensor::reshapeLike(const Tensor& other) {
		this->reshape(other.num_, other.channel_, other.height_, other.width_);
	}

	void Tensor::from(const void* ptr, int n, int c, int h, int w, DataType dtType) {
		this->dtType_ = dtType;
		resize(n, c, h, w);
		memcpy(host_, ptr, bytes_);
	}

	void Tensor::resize(const std::vector<int>& dims) {
		resize(dims.size(), dims.data());
	}

	void Tensor::resize(int ndims, const int* dims) {

		//最多支持4维的构建
		Assert(ndims <= 4 && ndims > 0);

		int n = ndims > 0 ? dims[0] : 1;
		int c = ndims > 1 ? dims[1] : 1;
		int h = ndims > 2 ? dims[2] : 1;
		int w = ndims > 3 ? dims[3] : 1;
		resize(n, c, h, w);
	}

	void Tensor::resize(int n, int c, int h, int w) {

		n = n == -1 ? this->num_ : n;
		c = c == -1 ? this->channel_ : c;
		h = h == -1 ? this->height_ : h;
		w = w == -1 ? this->width_ : w;

		int needed_size = n * c * h * w * elementSize();
		if (needed_size > capacity_) {
			release();

			this->bytes_ = needed_size;
			this->host_ = malloc(needed_size);
			this->capacity_ = needed_size;
			memset(this->host_, 0, this->bytes_);
		}
		this->num_ = n;
		this->channel_ = c;
		this->height_ = h;
		this->width_ = w;
		this->head_ = DataHead_InCPU;
		this->bytes_ = needed_size;
	}

	Tensor& Tensor::operator = (const Tensor& other) {
		from(other.cpu(), other.num(), other.channel(), other.height(), other.width(), other.type());
		return *this;
	}

	void Tensor::transposeInplace(int axis0, int axis1, int axis2, int axis3) {
		(*this) = transpose(axis0, axis1, axis2, axis3);
	}

	Tensor Tensor::transpose(int axis0, int axis1, int axis2, int axis3) {

		int dims[] = {num_, channel_, height_, width_};
		int muls[] = {count(1), count(2), count(3), 1};
		Tensor t(dims[axis0], dims[axis1], dims[axis2], dims[axis3], dtType_);

		int esize = elementSize();
		void* tptr = t.cpu();
		void* sptr = this->cpu();
		for (int a0 = 0; a0 < dims[axis0]; ++a0) {
			for (int a1 = 0; a1 < dims[axis1]; ++a1) {
				for (int a2 = 0; a2 < dims[axis2]; ++a2)
					for (int a3 = 0; a3 < dims[axis3]; ++a3) {
						//tptr[a0 * t.count(1) + a1 * t.count(2) + a2 * t.count(3) + a3] = sptr[a0 * muls[axis0] + a1 * muls[axis1] + a2 * muls[axis2] + a3 * muls[axis3]];
						size_t offsetTPtr = (a0 * t.count(1) + a1 * t.count(2) + a2 * t.count(3) + a3) * esize;
						size_t offsetSPtr = (a0 * muls[axis0] + a1 * muls[axis1] + a2 * muls[axis2] + a3 * muls[axis3]) * esize;
						memcpy((char*)tptr + offsetTPtr, (char*)sptr + offsetSPtr, esize);
					}
			}
		}
		return t;
	}

	void Tensor::toGPU(bool copyedIfCPU) {

		if (head_ == DataHead_InGPU)
			return;

		head_ = DataHead_InGPU;

		//if device memory is not malloc
		if (device_ == nullptr)
			cuCheck(cudaMalloc(&this->device_, this->bytes_));

		if (copyedIfCPU) {
			Assert(host_ != nullptr);
			cuCheck(cudaMemcpy(device_, host_, bytes_, cudaMemcpyHostToDevice));
		}
	}
	
	void Tensor::toCPU(bool copyedIfGPU) {

		if (head_ == DataHead_InCPU)
			return;

		head_ = DataHead_InCPU;

		if (copyedIfGPU) {
			Assert(host_ != nullptr && device_ != nullptr);
			cuCheck(cudaMemcpy(host_, device_, bytes_, cudaMemcpyDeviceToHost));
		}
	}

	void Tensor::toFloat() {

		if (type() == DataType::dtFloat)
			return;

		if (type() != DataType::dtHalfloat) {
			LOG(LFATAL) << "not implement function";
		}

		auto c = count();
		float* convert_memory = (float*)malloc(c * dataTypeSize(DataType::dtFloat));
		float* dst = convert_memory;
		halfloat* src = cpu<halfloat>();

		for (int i = 0; i < c; ++i)
			*dst++ = *src++;

		this->dtType_ = DataType::dtFloat;
		resize(-1);
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
	}

	void Tensor::toHalf() {

		if (type() == DataType::dtHalfloat)
			return;

		if (type() != DataType::dtFloat) {
			LOG(LFATAL) << "not implement function";
		}

		auto c = count();
		halfloat* convert_memory = (halfloat*)malloc(c * dataTypeSize(DataType::dtHalfloat));
		halfloat* dst = convert_memory;
		float* src = cpu<float>();

		for (int i = 0; i < c; ++i) 
			*dst++ = *src++;

		this->dtType_ = DataType::dtHalfloat;
		resize(-1);
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
	}

	void Tensor::setRandom(float low, float high) {
		int c = count();
		if (dtType_ == DataType::dtFloat) {
			float* ptr = cpu<float>();
			for (int i = 0; i < c; ++i)
				*ptr++ = ccutil::randrf(low, high);
		}
		else {
			halfloat* ptr = cpu<halfloat>();
			for (int i = 0; i < c; ++i)
				*ptr++ = ccutil::randrf(low, high);
		}
	}

	void Tensor::setTo(float value) {
		int c = count();
		if (dtType_ == DataType::dtFloat) {
			float* ptr = cpu<float>();
			for (int i = 0; i < c; ++i)
				*ptr++ = value;
		}
		else {
			halfloat* ptr = cpu<halfloat>();
			for (int i = 0; i < c; ++i)
				*ptr++ = value;
		}
	}

	void Tensor::print() {

		if (empty()) {
			cout << ccutil::format("Tensor %p ( is Empty )", this) << endl;
			return;
		}

		if (type() == DataType::dtFloat) {
			cout << ccutil::format("Tensor %p (%d x %d x %d x %d):", this, num_, channel_, height_, width_) << endl;

			int h = height_;
			int w = width_;
			for (int n = 0; n < num_; ++n) {
				for (int c = 0; c < channel_; ++c) {
					float* ptr = cpu<float>(n, c);
					cv::Mat m1(h, w, CV_32F, ptr);

					cout << ccutil::format("data[%d, %d, %d x %d]", n, c, h, w) << endl;
					cout << m1 << endl;
				}
			}
			cout << "===========================================" << endl;
		}
		else {
			INFO("not implement");
		}
	}

	void Tensor::copyFrom(const Tensor& other) {

		if (count() != other.count()) {
			resizeLike(other);
		}
		else {
			reshapeLike(other);
		}
		
		if (other.head_ == DataHead_InGPU) {
			if (this->device_ == nullptr) {
				cuCheck(cudaMalloc(&this->device_, this->bytes_));
			}

			this->head_ = DataHead_InGPU;
			cuCheck(cudaMemcpy(this->device_, other.device_, bytes_, cudaMemcpyDeviceToDevice));
		}
		else {
			this->head_ = DataHead_InCPU;
			memcpy(host_, other.host_, bytes_);
		}
	}

	void Tensor::setMatMeanScale(int n, const cv::Mat& image, float mean[3], float scale) {

		Assert(image.channels() == 3 && !image.empty() && type() == DataType::dtFloat);

		cv::Mat inputframe = image;
		if (inputframe.size() != cv::Size(width_, height_))
			cv::resize(inputframe, inputframe, cv::Size(width_, height_));

		inputframe.convertTo(inputframe, CV_32F);
		inputframe -= cv::Scalar(mean[0], mean[1], mean[2]);
		if (scale != 1) inputframe *= scale;

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height_, width_, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
	}

	void Tensor::setNormMat(int n, const cv::Mat& image, float mean[3], float std[3]) {

		Assert(image.channels() == 3 && !image.empty() && type() == DataType::dtFloat);
		
		float scale = 1 / 255.0;
		cv::Mat inputframe = image;
		if(inputframe.size() != cv::Size(width_, height_))
			cv::resize(inputframe, inputframe, cv::Size(width_, height_));

		inputframe.convertTo(inputframe, CV_32F, scale);

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height_, width_, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));

		for (int c = 0; c < 3; ++c)
			ms[c] = (ms[c] - mean[c]) / std[c];
	}

	void Tensor::setMat(int n, const cv::Mat& _image) {

		cv::Mat image = _image;
		Assert(!image.empty() && n < num_ && image.channels() == channel_ && CV_MAT_DEPTH(image.type()) == CV_32F && type() == DataType::dtFloat);
		
		if (image.size() != cv::Size(width_, height_))
			cv::resize(image, image, cv::Size(width_, height_));

		if (image.channels() == 1) {
			memcpy(cpu<float>(n), image.data, width_ * height_ * sizeof(float));
			return;
		}

		vector<cv::Mat> ms(image.channels());
		for (int i = 0; i < ms.size(); ++i) 
			ms[i] = cv::Mat(height_, width_, CV_32F, cpu<float>(n, i));

		cv::split(image, &ms[0]);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
	}

	////////////////////////////////////////////////////////////////////////////////
	template<typename _T>
	static void destroyNV(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	class EngineContext {
	public:
		virtual ~EngineContext() { destroy(); }

		bool buildModel(const string& data) {
			destroy();

			cuCheck(cudaStreamCreate(&stream_));

#if defined(HAS_PLUGIN)
			pluginFactory_ = Plugin::createPluginFactoryForInferPhase();
#endif
			runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroyNV<IRuntime>);

			if (runtime_ == nullptr)
				return false;

			engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(data.data(), data.size(), pluginFactory_.get()), destroyNV<ICudaEngine>);
			if (engine_ == nullptr)
				return false;

			context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroyNV<IExecutionContext>);
			return context_ != nullptr;
		}

	private:
		void destroy() {
			context_.reset();
			engine_.reset();
			runtime_.reset();
			pluginFactory_.reset();

			if (stream_) {cudaStreamDestroy(stream_);stream_ = nullptr;}
		}

	public:
		cudaStream_t stream_ = nullptr;
		shared_ptr<IExecutionContext> context_;
		shared_ptr<ICudaEngine> engine_;
		shared_ptr<nvinfer1::IPluginFactory> pluginFactory_;
		shared_ptr<IRuntime> runtime_ = nullptr;
	};

	class EngineImpl : public Engine {

	public:
		virtual bool load(const std::string& file);
		virtual void destroy();
		virtual void forward();
		virtual int maxBatchSize();

		virtual std::shared_ptr<Tensor> input(int index = 0);
		virtual std::shared_ptr<Tensor> output(int index = 0);
		virtual std::shared_ptr<Tensor> tensor(const std::string& name);

	private:
		void buildEngineInputAndOutputsMapper();

	private:
		std::vector<std::shared_ptr<Tensor>> inputs_;
		std::vector<std::shared_ptr<Tensor>> outputs_;
		std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
		std::map<std::string, int> blobsNameMapper_;
		std::shared_ptr<EngineContext> context_;
	};

	////////////////////////////////////////////////////////////////////////////////////
	void EngineImpl::destroy() {
		this->context_.reset();
		this->blobsNameMapper_.clear();
		this->outputs_.clear();
		this->inputs_.clear();
	}

	bool EngineImpl::load(const std::string& file) {

		destroy();
		string data = ccutil::loadfile(file);
		if (data.empty())
			return false;

		this->context_.reset(new EngineContext());

		//build model
		EngineContext* context = (EngineContext*)this->context_.get();
		if (!context->buildModel(data)) {
			this->context_.reset();
			return false;
		}

		buildEngineInputAndOutputsMapper();
		return true;
	}

	void EngineImpl::buildEngineInputAndOutputsMapper() {
		
		EngineContext* context = (EngineContext*)this->context_.get();
		int nbBindings = context->engine_->getNbBindings();

		inputs_.clear();
		outputs_.clear();
		orderdBlobs_.clear();
		blobsNameMapper_.clear();
		for (int i = 0; i < nbBindings; ++i) {

			auto dims = context->engine_->getBindingDimensions(i);
			//auto dtType = context->engine_->getBindingDataType(i);
			//这里的dtType是float
			Assert(dims.nbDims <= 4);

			int newDims[] = {1, 1, 1, 1};
			memcpy(newDims + 1, dims.d, sizeof(int) * dims.nbDims);
			auto mapperTensor = new Tensor(4, newDims, TRTInfer::DataType::dtFloat);
			auto newTensor = shared_ptr<Tensor>(mapperTensor);
			if (context->engine_->bindingIsInput(i)) {
				//if is input
				inputs_.push_back(newTensor);
			}
			else {
				//if is output
				outputs_.push_back(newTensor);
			}
			
			const char* bindingName = context->engine_->getBindingName(i);
			blobsNameMapper_[bindingName] = i;
			orderdBlobs_.push_back(newTensor);
		}
	}

	void EngineImpl::forward() {

		//检查input是否合理
		EngineContext* context = (EngineContext*)this->context_.get();
		int inputBatchSize = inputs_[0]->num();
		Assert(inputBatchSize <= context->engine_->getMaxBatchSize());

		for (int i = 0; i < outputs_.size(); ++i) 
			outputs_[i]->resize(inputBatchSize);

		//数据送到gpu
		vector<void*> bindings;
		for (int i = 0; i < orderdBlobs_.size(); ++i)
			bindings.push_back(orderdBlobs_[i]->gpu());

		void** bindingsptr = bindings.data();
		bool execute_result = context->context_->enqueue(inputBatchSize, bindingsptr, context->stream_, nullptr);
		Assert(execute_result);
		cuCheck(cudaStreamSynchronize(context->stream_));
	}

	std::shared_ptr<Tensor> EngineImpl::input(int index) {
		return this->inputs_[index];
	}

	std::shared_ptr<Tensor> EngineImpl::output(int index) {
		Assert(index >= 0 && index < outputs_.size());
		return outputs_[index];
	}

	int EngineImpl::maxBatchSize() {
		Assert(this->context_ != nullptr);
		return this->context_->engine_->getMaxBatchSize();
	}

	std::shared_ptr<Tensor> EngineImpl::tensor(const std::string& name) {
		Assert(this->blobsNameMapper_.find(name) != this->blobsNameMapper_.end());
		return orderdBlobs_[blobsNameMapper_[name]];
	}

	std::shared_ptr<Engine> loadEngine(const string& file) {
		
		std::shared_ptr<Engine> engine(new EngineImpl());
		if (!engine->load(file))
			engine.reset();
		return engine;
	}
};