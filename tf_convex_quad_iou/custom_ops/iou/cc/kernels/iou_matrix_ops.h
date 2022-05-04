#pragma once

#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {
namespace convex_quad_iou {
namespace functor {

template <typename Device,typename T>
struct IoUMatrixFunctor {
	void operator()(OpKernelContext * ctx, const Device & d,
	                const T * __restrict__ anchors,
	                const T * __restrict__ quads,
	                T * __restrict__ output,
	                const int anchors_size,
	                const int quads_size);
};



} // namespace functor
} // namespace convex_quad_iou
} // namespace tensorflow
