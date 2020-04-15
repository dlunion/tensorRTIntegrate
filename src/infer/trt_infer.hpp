

#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

class __half;
struct CUstream_st;

namespace TRTInfer {

	typedef __half halfloat;
	typedef CUstream_st* CUStream;

	enum DataHead {
		DataHead_InGPU,
		DataHead_InCPU
	};

	enum class DataType : int {
		dtFloat,
		dtHalfloat
	};

	int dataTypeSize(DataType dt);

	class Tensor {
	public:
		Tensor();
		Tensor(const Tensor& other);
		Tensor(int n, int c, int h, int w, DataType dtType = DataType::dtFloat);
		Tensor(int ndims, const int* dims, DataType dtType = DataType::dtFloat);
		Tensor(const std::vector<int>& dims, DataType dtType = DataType::dtFloat);
		Tensor& operator = (const Tensor& other);
		virtual ~Tensor();

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
		const char* shapeString();

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
		inline cv::Mat atMat(int n = 0, int c = 0) { return cv::Mat(height_, width_, CV_32F, cpu<float>(n, c)); }

		void setMat(int n, const cv::Mat& image);

		void setNormMat(int n, const cv::Mat& image, float mean[3], float std[3]);
		void setNormMatGPU(int n, const cv::Mat& image, float mean[3], float std[3]);

		//result = (image - mean) * scale
		void setMatMeanScale(int n, const cv::Mat& image, float mean[3], float scale = 1.0f);
		Tensor transpose(int axis0, int axis1, int axis2, int axis3);
		void transposeInplace(int axis0, int axis1, int axis2, int axis3);
		void from(const void* ptr, int n, int c = 1, int h = 1, int w = 1, DataType dtType = DataType::dtFloat);

		void* getTempGPUMemory(size_t size);
		void releaseTempGPUMemory();

	private:
		int num_ = 0, channel_ = 0, height_ = 0, width_ = 0;
		void* device_ = nullptr;
		void* host_ = nullptr;
		size_t capacity_ = 0;
		size_t bytes_ = 0;
		DataHead head_ = DataHead_InCPU;
		DataType dtType_ = DataType::dtFloat;
		char shapeString_[100];
		void* tempGPUMemory_ = nullptr;
		size_t tempGPUMemoryLength = 0;
	};

	class Engine {
	public:
		virtual bool load(const std::string& file) = 0;
		virtual void destroy() = 0;
		virtual void forward(bool sync = true) = 0;
		virtual int maxBatchSize() = 0;
		virtual CUStream getCUStream() = 0;
		virtual void synchronize() = 0;

		virtual std::shared_ptr<Tensor> input(int index = 0) = 0;
		virtual std::shared_ptr<Tensor> output(int index = 0) = 0;
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) = 0;
	};

	void setDevice(int device_id);
	std::shared_ptr<Engine> loadEngine(const std::string& file);
	bool initNVPlugins();
};	//TRTInfer


#endif //TRT_INFER_HPP