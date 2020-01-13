# TensorRT封装

---


## 更新
* 1、增加对FP16插件的支持，修改Tensor类支持FP16
* 2、修改WRelu插件支持FP16和Float两种格式的案例
* 3、把模型带在代码里面了，开箱用   
---


## 环境
* tensorRT6.0.1.5（可以修改为其他版本）
* opencv3.4.6（可以任意修改为其他版本）
* cudnn7.6.3（可以任意修改为其他版本）
* cuda10.0（可以任意修改为其他版本）
* Visual Studio 2017（可以用其他版本打开，但需要修改对应opencv版本）
* <font color=red>如果要修改版本，你需要下载cuda/cudnn/tensorRT三者同时匹配的版本，因为他们互相存在依赖，否则只要你是cuda10.0就可以很轻易编译这个项目</font>
* 提交的代码中包含了模型
---


## 案例-Inference
```
auto engine = TRTInfer::loadEngine("models/efficientnet-b0.fp32.trtmodel");
float mean[3] = {0.485, 0.456, 0.406};
float std[3] = {0.229, 0.224, 0.225};
Mat image = imread("img.jpg");
engine->input()->setNormMat(0, image, mean, std);
engine->forward();
engine->output(0)->print();
```

---

## 案例-INT8
定义预处理函数，量化INT8需要执行推理，需要正确的分布输入
```
auto preprocess = [](int current, int count, cv::Mat& inputOutput) {
    INFO("process: %d / %d", current, count);
    inputOutput.convertTo(inputOutput, CV_32F, 1 / 255.0f, -0.5f);
};
```

执行编译INT8模型，TRTBuilder::ModelSource指定模型从caffe而来，也可以是onnx
```
TRTBuilder::compileTRT(
    TRTBuilder::TRTMode_INT8, 
    {"fc_blob1"}, 4,
    TRTBuilder::ModelSource("models/handmodel_resnet18.prototxt", "models/handmodel_resnet18.caffemodel"),
    "models/handmodel_resnet18.int8.trtmodel", 
    preprocess, 
    "models/int8data", 
    "models/handmodel_resnet18.calibrator.txt"
);
```


## 支持
* Linux
* Windows
* 该封装代码是垮平台的

## 说明
* main.cpp里面有3个案例，分别是Int8Caffe、Onnx、Plugin
* 所有lib依赖项，均采用[import_lib.cpp](src/import_lib.cpp)导入，而不是通过配置
* infer文件夹为对TensorRT封装的Inference代码，目的是简化TensorRT前向过程，封装为Engine类和提供友好高性能的Tensor类支持
* plugin文件夹为对plugin的封装，pluginFactory的实现，以及友好的接口，写新插件只需要
  * 1.plugins里面添加插件类，继承自Plugin::TRTPlugin
  * 2.outputDims和enqueue方法，参照[WReLU.cu](src/plugin/plugins/WReLU.cu)和[WReLU.hpp](src/plugin/plugins/WReLU.hpp)，指明该插件的返回维度信息，以及插件前向时的运算具体实现，并在cpp/cu底下加上RegisterPlugin(WReLU);，参考[WReLU.cu](src/plugin/plugins/WReLU.cu)，完成注册
* builder文件夹则是对模型转换做封装，int8Caffe模型编译，onnx模型编译，fp32/fp16模型编译，通过简单的接口实现模型编译到trtmodel