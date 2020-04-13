

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace TRTBuilder {

	typedef std::function<void(int current, int count, cv::Mat& inputOutput)> Int8Process;

	void setDevice(int device_id);

	enum ModelSourceType {
		ModelSourceType_FromCaffe,
		ModelSourceType_FromONNX
	};

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

	class InputDims {
	public:
		InputDims(int channels, int height, int width);

		int channels() const;
		int height() const;
		int width() const;

	private:
		int channels_, height_, width_;
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
		unsigned int batchSize,
		const ModelSource& source,
		const std::string& savepath,
		Int8Process int8process = nullptr,					
		const std::string& int8ImageDirectory = "",
		const std::string& int8EntropyCalibratorFile = "",
		const std::vector<InputDims> inputsDimsSetup = {});
};

#endif //TRT_BUILDER_HPP