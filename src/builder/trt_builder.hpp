

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace TRTBuilder {

	typedef std::function<void(int current, int count, cv::Mat& inputOutput)> Int8Process;

	//cudaSetDevice的封装，避免包含cuda头文件
	void setDevice(int device_id);

	enum ModelSourceType {
		//对于来自于uff模型的支持，可以自己写上去
		ModelSourceType_FromCaffe,
		ModelSourceType_FromONNX
	};

	//定义模型的来源方式，通过这个类，自动的分辨是来自caffe还是onnx，便于处理
	class ModelSource {
	public:
		ModelSource(const std::string& prototxt, const std::string& caffemodel);
		ModelSource(const std::string& onnxmodel);
		ModelSourceType type() const;
		std::string prototxt() const;
		std::string caffemodel() const;
		std::string onnxmodel() const;

	private:
		std::string prototxt_, caffemodel_;
		std::string onnxmodel_;
		ModelSourceType type_;
	};

	enum TRTMode {
		TRTMode_FP32,
		TRTMode_FP16,
		TRTMode_INT8
	};

	const char* modeString(TRTMode type);

	bool compileTRT(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		Int8Process int8process = nullptr,						//如果是int8，则三个参数需要设置
		const std::string& int8ImageDirectory = "",
		const std::string& int8EntropyCalibratorFile = "");
};

#endif //TRT_BUILDER_HPP