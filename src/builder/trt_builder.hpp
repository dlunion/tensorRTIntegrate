

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace TRTBuilder {
	typedef std::function<void(int current, int count, cv::Mat& inputOutput)> Int8Process;

	void setDevice(int device_id);

	bool caffeToTRTFP32OrFP16(
		const std::string& deployFile,
		const std::string& modelFile,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const std::string& savepath,
		const std::string& mode = "FP32");	//if fp16, set mode = "FP16"

	bool caffeToTRT_INT8(
		const std::string& deployFile,
		const std::string& modelFile,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const std::string& savepath,
		const std::string& imageDirectory,
		const Int8Process& preprocess);

	bool onnxToTRTFP32OrFP16(
		const std::string& modelFile,
		unsigned int maxBatchSize,
		const std::string& savepath,
		const std::string& mode = "FP32");
};

#endif //TRT_BUILDER_HPP