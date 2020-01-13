

#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <cuda_fp16.h>

//为了避免对Plugin的引用，在inference时默认没有插件，复制hpp和cpp即可干活，所以定义了这个宏，当使用插件时打开即可
//该宏定义只是在反序列化模型之前对pluginFactory做了赋值，没有其他地方再有引用了
#define HAS_PLUGIN

namespace TRTInfer {

	typedef __half halfloat;

	enum DataHead {
		DataHead_InGPU,
		DataHead_InCPU
	};

	enum class DataType : int {
		dtFloat,
		dtHalfloat
	};

	inline int dataTypeSize(DataType dt){
		switch (dt) {
		case DataType::dtFloat: return sizeof(float);
		case DataType::dtHalfloat: return sizeof(halfloat);
		default: return -1;
		}
	}

	//Tensor类，对张量的封装，对GPU、CPU交互操作做了封装
	//使用者只需要关心数据如何输入以及如何处理，而无需关心数据应该这么复制到GPU以及何时复制到CPU
	class Tensor {
	public:
		Tensor();
		Tensor(const Tensor& other);
		Tensor(int n, int c, int h, int w, DataType dtType = DataType::dtFloat);
		Tensor(int ndims, const int* dims, DataType dtType = DataType::dtFloat);
		Tensor(const std::vector<int>& dims, DataType dtType = DataType::dtFloat);
		Tensor& operator = (const Tensor& other);

		inline DataType type() const { return dtType_; }
		inline std::vector<int> dims() const { return {num_, channel_, height_, width_}; }
		inline cv::Size size() const { return cv::Size(width_, height_); }
		inline int offset(int n = 0, int c = 0, int h = 0, int w = 0) { return ((n * this->channel_ + c) * this->height_ + h) * this->width_ + w; }
		inline int num() const { return num_; }
		inline int channel() const { return channel_; }
		inline int height() const { return height_; }
		inline int width() const { return width_; }
		inline int bytes() const { return bytes_; }
		inline int bytes(int start_axis) const { return count(start_axis) * elementSize(); }
		inline int elementSize() const { return dataTypeSize(dtType_); }

		void copyFrom(const Tensor& other);
		void release();
		void setTo(float value);
		void setRandom(float low = 0, float high = 1);
		bool empty();
		void resize(int n, int c = -1, int h = -1, int w = -1);
		void resize(int ndims, const int* dims);
		void resize(const std::vector<int>& dims);
		void reshape(int n, int c = -1, int h = -1, int w = -1);
		void reshapeLike(const Tensor& other);
		void resizeLike(const Tensor& other);
		size_t count(int start_axis = 0) const;
		void print();

		/* 到gpu
		* copyedIfCPU = true，如果数据在cpu上，则复制到gpu，并修改head为InGPU
		* copyedIfCPU = false，无论数据集是否在cpu上，都无视，并修改head为InGPU
		* 如果gpu没有分配空间，数据又在cpu上时，会为gpu分配空间*/
		void toGPU(bool copyedIfCPU = true);
		void toCPU(bool copyedIfGPU = true);
		void toHalf();
		void toFloat();

		inline const void* cpu() const { ((Tensor*)this)->toCPU(); return host_; }
		inline const void* gpu() const { ((Tensor*)this)->toGPU(); return device_; }
		inline void* cpu() { toCPU(); return host_; }
		inline void* gpu() { toGPU(); return device_; }
		
		template<typename DataT> inline const DataT* cpu() const { ((Tensor*)this)->toCPU(); return (const DataT*)host_; }
		template<typename DataT> inline const DataT* gpu() const { ((Tensor*)this)->toGPU(); return (const DataT*)device_; }
		template<typename DataT> inline DataT* cpu() { toCPU(); return (DataT*)host_; }
		template<typename DataT> inline DataT* gpu() { toGPU(); return (DataT*)device_; }
		template<typename DataT> inline DataT* cpu(int n, int c = 0, int h = 0, int w = 0) { return cpu<DataT>() + offset(n, c, h, w); }
		template<typename DataT> inline DataT* gpu(int n, int c = 0, int h = 0, int w = 0) { return gpu<DataT>() + offset(n, c, h, w); }
		template<typename DataT> inline float& at(int n = 0, int c = 0, int h = 0, int w = 0) { return *(cpu<DataT>() + offset(n, c, h, w)); }
		template<typename DataT> inline cv::Mat at(int n = 0, int c = 0) { return cv::Mat(height_, width_, CV_32F, cpu<DataT>(n, c)); }

		/* 设置图像，如果图像大小不一致会resize，通道数、宽高、类型CV_32F要求必须匹配，
		* 否则报错，不会对数值做任何操作（例如归一化、缩放）*/
		void setMat(int n, const cv::Mat& image);

		/* 设置图像，result = (image[c] / 255 - mean[c]) / std[c]
		* 要求图像通道一致，不要求其他格式*/
		void setNormMat(int n, const cv::Mat& image, float mean[3], float std[3]);

		//result = (image - mean) * scale
		void setMatMeanScale(int n, const cv::Mat& image, float mean[3], float scale = 1.0f);
		Tensor transpose(int axis0, int axis1, int axis2, int axis3);
		void transposeInplace(int axis0, int axis1, int axis2, int axis3);
		void from(const void* ptr, int n, int c = 1, int h = 1, int w = 1, DataType dtType = DataType::dtFloat);

	private:
		int num_ = 0, channel_ = 0, height_ = 0, width_ = 0;
		void* device_ = nullptr;
		void* host_ = nullptr;
		size_t capacity_ = 0;
		size_t bytes_ = 0;
		DataHead head_ = DataHead_InCPU;
		DataType dtType_ = DataType::dtFloat;
	};

	//TensorRT的inference封装为Engine对象，允许直接加载模型，通过Tensor来操作输入和输出，也允许对input(0)做resize(batch)实现多batch的输入
	//forward时会自动对output做resize，使得其num维度一致
	class Engine {
	public:
		virtual bool load(const std::string& file) = 0;
		virtual void destroy() = 0;
		virtual void forward() = 0;
		virtual int maxBatchSize() = 0;

		virtual std::shared_ptr<Tensor> input(int index = 0) = 0;
		virtual std::shared_ptr<Tensor> output(int index = 0) = 0;

		/* 根据名字获取tensor指针，该tensor可能是输入也可能是输出，包括且仅包括构建模型时的input和markedOutput指定的Tensor*/
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) = 0;
	};

	void setDevice(int device_id);
	std::shared_ptr<Engine> loadEngine(const std::string& file);
	bool initNVPlugins();
};	//TRTInfer


#endif //TRT_INFER_HPP