

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace TRTBuilder {

	//INT8模式下的自定义预处理函数
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

	//把模式转换为字符串FP32、FP16、INT8
	const char* modeString(TRTMode type);


	//当处于INT8模式时，int8process必须制定
	//     int8ImageDirectory和int8EntropyCalibratorFile指定一个即可
	//     如果初次生成，指定了int8EntropyCalibratorFile，calibrator会保存到int8EntropyCalibratorFile指定的文件
	//     如果已经生成过，指定了int8EntropyCalibratorFile，calibrator会从int8EntropyCalibratorFile指定的文件加载，而不是
	//          从int8ImageDirectory读取图片再重新生成
	//当处于FP32或者FP16时，int8process、int8ImageDirectory、int8EntropyCalibratorFile都不需要指定
	bool compileTRT(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		Int8Process int8process = nullptr,					
		const std::string& int8ImageDirectory = "",
		const std::string& int8EntropyCalibratorFile = "");
};

#endif //TRT_BUILDER_HPP