

#include "DCNv2.hpp"
#include <common/json.hpp>
#include <cublas_v2.h>

typedef TRTInfer::halfloat halfloat;

#define cublasCheck(op)														 \
do {																	 \
	auto ret = (op);													 \
	if (ret != CUBLAS_STATUS_SUCCESS) {											 \
		INFO("%s fail, %d != %d", #op, ret, CUBLAS_STATUS_SUCCESS);				 \
		abort();													     \
	}																	 \
} while (0);


template<typename _T>
static __global__ void sigmoidKernel(_T* input, _T* output, int edge);

template<>
__global__ void sigmoidKernel(float* input, float* output, int edge) {

	KERNEL_POSITION;
	output[position] = 1 / (1 + exp(-input[position]));
}

template<>
__global__ void sigmoidKernel(halfloat* input, halfloat* output, int edge) {

	KERNEL_POSITION;
	halfloat one = 1.0f;
	output[position] = one / (one + hexp(-input[position]));
}

static __device__ float dmcnIm2colBilinearFP32(const float *bottom_data, const int data_width,
	const int height, const int width, float h, float w)
{
	int h_low = floor(h);
	int w_low = floor(w);
	int h_high = h_low + 1;
	int w_high = w_low + 1;

	float lh = h - h_low;
	float lw = w - w_low;
	float hh = 1 - lh, hw = 1 - lw;

	float v1 = 0;
	if (h_low >= 0 && w_low >= 0)
		v1 = bottom_data[h_low * data_width + w_low];
	float v2 = 0;
	if (h_low >= 0 && w_high <= width - 1)
		v2 = bottom_data[h_low * data_width + w_high];
	float v3 = 0;
	if (h_high <= height - 1 && w_low >= 0)
		v3 = bottom_data[h_high * data_width + w_low];
	float v4 = 0;
	if (h_high <= height - 1 && w_high <= width - 1)
		v4 = bottom_data[h_high * data_width + w_high];

	float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

	float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
	return val;
}

static __device__ halfloat dmcnIm2colBilinearFP16(const halfloat *bottom_data, const int data_width,
	const int height, const int width, const halfloat& h, const halfloat& w)
{
	int h_low = hfloor(h);
	int w_low = hfloor(w);
	int h_high = h_low + 1;
	int w_high = w_low + 1;

	halfloat one = 1.0f;
	halfloat h_low_hf = h_low;
	halfloat w_low_hf = w_low;
	halfloat lh = h - h_low_hf;
	halfloat lw = w - w_low_hf;
	halfloat hh = one - lh, hw = one - lw;

	halfloat zero = 0.0f;
	halfloat v1 = zero;
	if (h_low >= 0 && w_low >= 0)
		v1 = bottom_data[h_low * data_width + w_low];
	halfloat v2 = zero;
	if (h_low >= 0 && w_high <= width - 1)
		v2 = bottom_data[h_low * data_width + w_high];
	halfloat v3 = zero;
	if (h_high <= height - 1 && w_low >= 0)
		v3 = bottom_data[h_high * data_width + w_low];
	halfloat v4 = zero;
	if (h_high <= height - 1 && w_high <= width - 1)
		v4 = bottom_data[h_high * data_width + w_high];

	halfloat w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
	return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

template<typename _T>
static __global__ void DCNIm2colKernel(
	const _T *data_input, const _T *data_offset, const _T *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	_T *data_output, int edge);

template<>
__global__ void DCNIm2colKernel(
	const float *data_input, const float *data_offset, const float *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	float *data_output, int edge)
{
	KERNEL_POSITION;

	const int f_area_input = width_input * height_input;
	const int f_area_output = width_output * height_output;

	// index index of output matrix
	const int w_output = position % width_output;
	const int h_output = (position / width_output) % height_output;
	const int c_input = (position / width_output / height_output) % num_channels;

	const int c_output = c_input * kernel_h * kernel_w;
	const int deformable_group_index = c_input / channel_per_deformable_group;
	const int h_input = h_output * stride_h - pad_h;
	const int w_input = w_output * stride_w - pad_w;


	int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
	float *data_output_ptr = data_output + data_output_offset;
	const float *data_input_ptr = data_input + c_input * f_area_input;
	const float *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
	const float *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

	for (int i = 0; i < kernel_h; ++i)
	{
		for (int j = 0; j < kernel_w; ++j)
		{
			const int row = i + h_input;
			const int col = j + w_input;
			const int kernel_index = i * kernel_w + j;

			const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
			const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
			const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

			const float offset_h = data_offset_ptr[offset_h_offset];
			const float offset_w = data_offset_ptr[offset_w_offset];
			const float mask = data_mask_ptr[mask_offset];

			float val = 0;
			const float h_im = h_input + i * dilation_h + offset_h;
			const float w_im = w_input + j * dilation_w + offset_w;

			if (h_im > -1 && w_im > -1 && h_im < height_input && w_im < width_input)
			{
				val = dmcnIm2colBilinearFP32(data_input_ptr, width_input, height_input, width_input, h_im, w_im);
			}
			*data_output_ptr = val * mask;
			data_output_ptr += f_area_output;
		}
	}
}

template<>
__global__ void DCNIm2colKernel(
	const halfloat *data_input, const halfloat *data_offset, const halfloat *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	halfloat *data_output, int edge)
{
	KERNEL_POSITION;

	const int f_area_input = width_input * height_input;
	const int f_area_output = width_output * height_output;

	// index index of output matrix
	const int w_output = position % width_output;
	const int h_output = (position / width_output) % height_output;
	const int c_input = (position / width_output / height_output) % num_channels;

	const int c_output = c_input * kernel_h * kernel_w;
	const int deformable_group_index = c_input / channel_per_deformable_group;
	const int h_input = h_output * stride_h - pad_h;
	const int w_input = w_output * stride_w - pad_w;

	halfloat width_input_hf = __float2half(width_input);
	halfloat height_input_hf = __float2half(height_input);

	halfloat h_input_hf = __float2half(h_input);
	halfloat w_input_hf = __float2half(w_input);
	halfloat dilation_h_hf = __float2half(dilation_h);
	halfloat dilation_w_hf = __float2half(dilation_w);

	int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
	halfloat *data_output_ptr = data_output + data_output_offset;
	const halfloat *data_input_ptr = data_input + c_input * f_area_input;
	const halfloat *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
	const halfloat *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

	halfloat n_one = -1.0f;
	halfloat zero = 0.0f;

	for (int i = 0; i < kernel_h; ++i)
	{
		for (int j = 0; j < kernel_w; ++j)
		{
			halfloat i_hf = __float2half(i);
			halfloat j_hf = __float2half(j);
			const int row = i + h_input;
			const int col = j + w_input;
			const int kernel_index = i * kernel_w + j;

			const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
			const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
			const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

			const halfloat offset_h = data_offset_ptr[offset_h_offset];
			const halfloat offset_w = data_offset_ptr[offset_w_offset];
			const halfloat mask = data_mask_ptr[mask_offset];

			halfloat val = zero;
			halfloat h_im = h_input_hf + i_hf * dilation_h_hf + offset_h;
			halfloat w_im = w_input_hf + j_hf * dilation_w_hf + offset_w;

			if (h_im > n_one && w_im > n_one && h_im < height_input_hf && w_im < width_input_hf)
			{
				val = dmcnIm2colBilinearFP16(data_input_ptr, width_input_hf, height_input_hf, width_input_hf, h_im, w_im);
			}
			*data_output_ptr = val * mask;
			data_output_ptr += f_area_output;
		}
	}
}

template<typename _T>
static __global__ void biasKernel(_T* data_input, const _T* bias, const int f_area, int edge) {

	KERNEL_POSITION;
	int bias_index = position / f_area;
	data_input[position] += bias[bias_index];
}

template<typename _T>
inline void segemm_native(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha, /* host or device pointer */
	const _T *A,
	int lda,
	const _T *B,
	int ldb,
	float beta, /* host or device pointer */
	_T *C,
	int ldc);

template<>
inline void segemm_native<float>(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha, /* host or device pointer */
	const float *A,
	int lda,
	const float *B,
	int ldb,
	float beta, /* host or device pointer */
	float *C,
	int ldc) {
	cublasCheck(cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
	//cublasCheck(cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT));
}

template<>
inline void segemm_native<TRTInfer::halfloat>(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha,
	const TRTInfer::halfloat *A,
	int lda,
	const TRTInfer::halfloat *B,
	int ldb,
	float beta, 
	TRTInfer::halfloat *C,
	int ldc) {

	auto halpha = TRTInfer::halfloat(alpha);
	auto hbeta = TRTInfer::halfloat(beta);
	//cublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &halpha, A, lda, B, ldb, &hbeta, C, ldc));
	cublasCheck(cublasGemmEx(handle, transa, transb, m, n, k, &halpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &hbeta, C, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DFALT));
}

template<typename _T>
static void enqueue_native(cublasHandle_t handle, const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
	auto& data = inputs[0];
	auto& om = inputs[1];
	auto& out = outputs[0];

	int kernel_size = weights[0].width_;
	int deformable_group = 1;
	size_t maskSize = (size_t)data.height_ * data.width_ * kernel_size * kernel_size * deformable_group;
	size_t im2colSize = (size_t)data.channel_ * kernel_size * kernel_size * out.height_ * out.width_;

	const int m = out.channel_;
	const int n = out.count(2);
	const int k = data.channel_ * kernel_size * kernel_size;
	float alpha = 1.0;
	float beta = 0.0;

	cublasCheck(cublasSetStream(handle, stream));
	for (int ibatch = 0; ibatch < data.num_; ++ibatch) {
		_T* maskWorkspacePtr = (_T*)workspace + (maskSize + im2colSize) * ibatch;
		_T* im2colWorkspacePtr = (_T*)workspace + (maskSize + im2colSize) * ibatch + maskSize;

		_T* inputMask = om.ptr<_T>(ibatch, om.channel_ / 3 * 2);
		ExecuteKernel(maskSize, sigmoidKernel, stream)(inputMask, maskWorkspacePtr, maskSize);

		_T* datainput = data.ptr<_T>(ibatch);
		_T* offset = om.ptr<_T>(ibatch);

		ExecuteKernel(im2colSize, DCNIm2colKernel, stream)(
			datainput, offset, maskWorkspacePtr, data.height_, data.width_, kernel_size, kernel_size, 1, 1, 1, 1, 1, 1, data.channel_, data.num_, data.channel_, deformable_group,
			out.height_, out.width_, im2colWorkspacePtr, im2colSize);

		_T* weightKernel = weights[0].ptr<_T>();
		segemm_native(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, im2colWorkspacePtr, n, weightKernel, k, beta, out.ptr<_T>(ibatch), n);

		if (weights.size() > 1) {
			_T* weightBias = weights[1].ptr<_T>();
			size_t edge = out.count(1);
			size_t area = out.count(2);
			ExecuteKernel(edge, biasKernel, stream)(out.ptr<_T>(ibatch), weightBias, area, edge);
		}
	}
}

int DCNv2::initialize(){
	cublasCheck(cublasCreate(&cublasHandle_));
	return 0;
}

void DCNv2::terminate(){
	cublasCheck(cublasDestroy(cublasHandle_));
	cublasHandle_ = nullptr;
}

int DCNv2::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
	
	if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
		enqueue_native<float>(cublasHandle_, inputs, outputs, weights, workspace, stream);
	}
	else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
		enqueue_native<TRTInfer::halfloat>(cublasHandle_, inputs, outputs, weights, workspace, stream);
	}
	return 0;
}

nvinfer1::Dims DCNv2::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
	//INFO("inputDims = %d, %d, %d, %d, %d", inputDims[0].nbDims, inputDims[0].d[0], inputDims[0].d[1], inputDims[0].d[2], inputDims[0].d[3]);
	return nvinfer1::Dims3(config_->weights_[0]->num(), inputDims[0].d[1], inputDims[0].d[2]);
}

size_t DCNv2::getWorkspaceSize(int maxBatchSize) const {

	int kernel_size = 3;
	int deformable_group = 1;

	//inputChannel * k * k * outputHeight * outputWidth
	size_t im2colSize = (size_t)config_->input[0].d[0] * kernel_size * kernel_size * config_->output[0].d[1] * config_->output[0].d[2];
	size_t maskSize = (size_t)config_->input[0].d[1] * config_->input[0].d[2] * kernel_size * kernel_size * deformable_group;
	config_->workspaceSize_ = (im2colSize + maskSize) * maxBatchSize * TRTInfer::dataTypeSize(config_->configDataType_);
	return config_->workspaceSize_;
}

std::shared_ptr<LayerConfig> DCNv2::config(const std::string& layerName) {
	auto cfg = TRTPlugin::config(layerName);

	cfg->supportDataType_ = {nvinfer1::DataType::kFLOAT};
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
	return cfg;
}

RegisterPlugin(DCNv2);