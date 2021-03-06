#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tf_convex_quad_iou/custom_ops/iou/cc/kernels/iou_matrix_ops.h"

#define TO_DEBUG (i == 2 )

namespace tensorflow {
namespace convex_quad_iou {

using GPUDevice = Eigen::GpuDevice;


namespace {


template <typename T>
__global__ void IoUMatrixKernel(const T* __restrict__ anchors,
                                const T* __restrict__ quads,
                                T* __restrict__ output,
                                const int anchors_size,
                                const int quads_size) {
	int size = anchors_size * quads_size;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
	     i += blockDim.x * gridDim.x) {

		int iAnchors = i / quads_size;
		int iQuads = i - iAnchors * quads_size;
		output[i] = ComputeIoU<T>(anchors + 8 * iAnchors,
		                          quads + 8 * iQuads);

	}
}

template <typename T>
__global__ void QuadCopyKernel(const T * __restrict__ input,
                           T * __restrict__ output,
                           const int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
	     i += blockDim.x * gridDim.x) {
		CopyQuad<T>(input + 8 * i,output + 8* i);
	}
}


} // namespace

namespace functor {

template <typename T>
struct IoUMatrixFunctor<GPUDevice,T> {
	void operator()(OpKernelContext * ctx, const GPUDevice & d,
	                const T * __restrict__ anchors,
	                const T * __restrict__ quads,
	                T * __restrict__ output,
	                const int anchors_size,
	                const int quads_size) {
		const int output_size =
			anchors_size * quads_size;
		GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d,
		                                            IoUMatrixKernel<T>,0,0);

		TF_CHECK_OK(GpuLaunchKernel(IoUMatrixKernel<T>,
		                            config.block_count,
		                            config.thread_per_block,
		                            0,
		                            d.stream(),
		                            anchors,quads,
		                            output,
		                            anchors_size,quads_size));
	}
};

template struct IoUMatrixFunctor<GPUDevice,Eigen::half>;
template struct IoUMatrixFunctor<GPUDevice,float>;
template struct IoUMatrixFunctor<GPUDevice,double>;

template <typename T>
struct QuadCopyFunctor<GPUDevice,T> {
	void operator()(OpKernelContext * ctx, const GPUDevice & d,
	                const T * __restrict__ input,
	                T * __restrict__ output,
	                const int size) {
		GpuLaunchConfig config = GetGpuLaunchConfig(size, d,
		                                            QuadCopyKernel<T>,0,0);
		TF_CHECK_OK(GpuLaunchKernel(QuadCopyKernel<T>,
		                            config.block_count,
		                            config.thread_per_block,
		                            0,
		                            d.stream(),
		                            input,
		                            output,
		                            size));
	}
};

template struct QuadCopyFunctor<GPUDevice,Eigen::half>;
template struct QuadCopyFunctor<GPUDevice,float>;
template struct QuadCopyFunctor<GPUDevice,double>;


} // namespace functor

} // namespace convex_quad_iou
} // namespace tensorflow



#endif // GOOGLE_CUDA
