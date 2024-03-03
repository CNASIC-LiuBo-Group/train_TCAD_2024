#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

// 定义一个256*256的纹理
//CUDA 11.7 DEPRECATED
//texture<int, cudaTextureType2D, cudaReadModeElementType> texRef;

/*
*********************************************************************
function name: matmul_cuda_app
description: dot product of two arbitrarily sized matrices by LuT.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  output_app: app output image of size m x k.
  m,n,k: sizes of matrices.
  batch_size: Number of images in each batch.
  weight_scale, activation_scale: quant scale.
return: none
Acknowledgement: Original code from 'lzhengchun/matrix-cuda' on github.
link: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
*********************************************************************
*/
__global__ void matmul_cuda_app(
  cudaTextureObject_t texObj,
  const float *image,
  const float *weight,
  const float *bias,
  float *output_app,
  const int m,
  const int n,
  const int k,
  const int batch_size,
  const float weight_scale,
  const float activation_scale)
{

    // This code doesn't really get much faster using shared memory, since
    // accesses to the image matrix are all sequential anyway. The first access
    // already caches everything, making shared memory useless.

    int img = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float product_appx = 0.0f;
    if( col < k && row < m && img < batch_size){
        for(int i = 0; i < n; i++){
            int texx = int(image[(img*m*n)+(row*n) + i]) + 128;
            int texy = int(weight[i * k + col]) + 128;

            product_appx += float(tex2D<int>(texObj, texx, texy));
	}
        output_app[(img*m*k)+(row*k) + col] = product_appx * weight_scale * activation_scale + bias[col];
    }
}

/*
*********************************************************************
function name: conv_app_forward
description: convolutional layer that calls the app matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
  b: Number of images in each batch.
  weight_scale, activation_scale: quant scale.
  mul: LuT.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor conv_app_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	int b,
	float weight_scale,
	float activation_scale,
  torch::Tensor mul
) {

 // scale  
	input = input/activation_scale;
	weight = weight/weight_scale;
	

  // Create an output of size b X m X k, directly on the GPU.
	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
  	auto output_app = torch::zeros({b, m, k}, options);
	torch::Tensor intMul = mul.to(torch::kInt);
  // mul read
/*
  std::vector<int> mulvec2d;
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
        mulvec2d.push_back(int(mul[i][j].item<float>()));
    }
  }
*/

  // 在主机上分配内存并复制数据到设备
  int* deviceData;
  cudaMalloc(&deviceData, sizeof(int) * 256 * 256);
  cudaMemcpy(deviceData, intMul.data_ptr<int>(), sizeof(int) * 256 * 256, cudaMemcpyHostToDevice);

  // 设置纹理内存描述符
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
  cudaChannelFormatDesc channelDesc = 
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, 256, 256);
  //cudaMemcpyToArray(cuArray, 0, 0, deviceData, sizeof(int) * 256 * 256, cudaMemcpyDeviceToDevice);
  // 绑定纹理内存
  //cudaBindTextureToArray(texRef, cuArray, channelDesc);
  const size_t spitch = 256 * sizeof(int);
  cudaMemcpy2DToArray(cuArray, 0, 0, deviceData, spitch, 256 * sizeof(int),
                        256, cudaMemcpyHostToDevice);
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Use this block size to not exceed 1024 threads across all 3 dimensions.
  // You can also do dimblock(16 x 16 x 4) to use all 1024 threads if your
  // batches are small.
	unsigned int block_size = 8;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;
	unsigned int grid_images = (b + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows, grid_images);
	dim3 dimBlock(block_size, block_size, block_size);

  // This is not the 'pytorch recommended way' of launching this kernel.
  // But it works just fine so I've left it this way since it is easier to debug
  // if there is an issue launching the kernel for example.

	matmul_cuda_app<<<dimGrid, dimBlock>>>(
    texObj,
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
    		output_app.data_ptr<float>(),
		m, n, k, b, weight_scale, activation_scale
	);

  cudaDeviceSynchronize();
  
  //cudaUnbindTexture(texRef);
  cudaDestroyTextureObject(texObj);
  // 释放资源
  cudaFree(deviceData);
  cudaFreeArray(cuArray);
  return output_app;
}

/*
*********************************************************************
function name: linear_app_forward
description: linear layer that calls the matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor linear_app_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	float weight_scale,
	float activation_scale,
  torch::Tensor mul
) {
  // scale  
	input = input/activation_scale;
	weight = weight/weight_scale;

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output_app = torch::zeros({m,k}, options);
	torch::Tensor intMul = mul.to(torch::kInt);


  // 在主机上分配内存并复制数据到设备
  int* deviceData;
  cudaMalloc(&deviceData, sizeof(int) * 256 * 256);
  cudaMemcpy(deviceData, intMul.data_ptr<int>(), sizeof(int) * 256 * 256, cudaMemcpyHostToDevice);
  // 设置纹理内存描述符
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
  cudaChannelFormatDesc channelDesc = 
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, 256, 256);
  //cudaMemcpyToArray(cuArray, 0, 0, deviceData, sizeof(int) * 256 * 256, cudaMemcpyDeviceToDevice);
  // 绑定纹理内存
  //cudaBindTextureToArray(texRef, cuArray, channelDesc);
  const size_t spitch = 256 * sizeof(int);
  cudaMemcpy2DToArray(cuArray, 0, 0, deviceData, spitch, 256 * sizeof(int),
                        256, cudaMemcpyHostToDevice);
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	unsigned int block_size = 32;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(block_size, block_size);

  // Linear layers have a vector input. But to re-use the matmul kernel,
  // just pass in a 'batch' of inputs as an m X n matrix, to be multiplied
  // by the n x k weights, to get 'm' output images.

	matmul_cuda_app<<<dimGrid, dimBlock>>>(
    texObj,
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output_app.data_ptr<float>(),
		m, n, k, 1, weight_scale, activation_scale // Pass in b=1 since there is no z-dimension for linear layers
	);

  cudaDeviceSynchronize();

  //cudaUnbindTexture(texRef);
  cudaDestroyTextureObject(texObj);
  // 释放资源
  cudaFree(deviceData);
  cudaFreeArray(cuArray);
  return output_app;
}

// Binding to generate the .so file, to call from python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Implementation of forward pass of conv and linear app layers in CUDA";
  m.def("conv_app_forward", &conv_app_forward, "conv_app_forward (CUDA)");
	m.def("linear_app_forward", &linear_app_forward, "linear_app_forward (CUDA)");
}
