
#include "plugin.hpp"
#include <plugin/plugins/WReLU.hpp>

using namespace nvinfer1;
using namespace std;

namespace Plugin {

	void Manager::buildPluginList() {
		AddPlugin(WReLU);
	}

};// namespace Plugin