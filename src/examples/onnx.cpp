
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

namespace examples {

	void onnx() {

		if (!ccutil::exists("models/demo.onnx")) {
			INFOE("models/demo.onnx not exists, run< python plugin_onnx_export.py > generate demo.onnx.");
			return;
		}

		INFOW("onnx to trtmodel...");
		/*TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 4,
			TRTBuilder::ModelSource("models/demo.onnx"),
			"models/demo.fp32.trtmodel", 
			{TRTBuilder::InputDims(3, 5, 5), TRTBuilder::InputDims(3, 5, 5)}
		);*/

		/*TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 4,
			TRTBuilder::ModelSource("models/lab_direction.onnx"),
			"models/demo.fp32.trtmodel",
			{TRTBuilder::InputDims(3, 64, 64)}
		);*/
		INFO("done.");

		INFO("load model: models/demo.fp32.trtmodel");
		auto engine = TRTInfer::loadEngine("models/demo.fp32.trtmodel");
		if (!engine) {
			INFO("can not load model.");
			return;
		}

		INFO("forward...");

		engine->input(0)->setTo(0.25);
		engine->forward();
		auto output = engine->output(0);
		output->print();
		INFO("done.");
	}
};