


#include <cc_util.hpp>
#include "infer/trt_infer.hpp"

namespace examples {

	// src/examples/*.cpp
	void center_net_coco2x_dcn();
	void center_track_coco_tracking();
	void dbface();
	void onnx();
};

int main() {

	//save logs to file
	ccutil::setLoggerSaveDirectory("logs");
	TRTInfer::setDevice(0);

	examples::onnx();
	examples::dbface();
	examples::center_net_coco2x_dcn();
	examples::center_track_coco_tracking();

	return 0;
}