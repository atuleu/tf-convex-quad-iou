#include "tf_convex_quad_iou/custom_ops/iou/cc/kernels/iou_matrix_ops.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace convex_quad_iou {


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
using Float2 = Eigen::Matrix<T,2,1>;

template<typename U>
void RotateArray(U * array,int size) {
	U temp = array[0];
	for ( int i = 1; i < size; ++i) {
		array[i-1] = array[i];
	}
	array[size-1] = temp;
}

template <typename T>
class Line {
	T a,b,c;
public:
	Line(const Float2<T> & v1,
	     const Float2<T> & v2)
		: a ( v2.y() - v1.y())
		, b ( v1.x() - v2.x())
		, c ( v2.x() * v1.y() - v2.y() * v1.x() ) {
	}
	T Project(const Float2<T> & p) const {
		return a * p.x() + b * p.y() + c;
	}

	bool Intersects(const Line & other) const {
		T w(a * other.b - b * other.a);
		return abs(w) > 1e-6;
	}

	Float2<T> Intersection(const Line & other) const {
		T w(a * other.b - b * other.a);
		return Float2<T>(b * other.c - c * other.b, c * other.a - a * other.c) / w;
	}

	std::string Format() const {
		return "a: " + std::to_string(a)
			+ "b: " + std::to_string(b)
			+ "c: " + std::to_string(c);
	}

};

template <typename T>
int ComputeIntersection(const Float2<T> * quad,
                        const Float2<T> * quadRotated,
                        Float2<T> * intersection) {
	int count = 4;
	for ( int i = 0; i < 4; ++i) {
		Line<T> l(quad[i],quadRotated[i]);
		T lineValues[8],lineValuesRotated[8];
		Eigen::Matrix<T,2,1> intersectionRotated[8],tempIntersection[8];
		for ( int j = 0; j < count; ++j) {
			lineValues[j] = l.Project(intersection[j]);
			lineValuesRotated[j] = lineValues[j];
			intersectionRotated[j] = intersection[j];
		}
		RotateArray(lineValuesRotated,count);
		RotateArray(intersectionRotated,count);

		int tempCount = count;
		count = 0;
#define appendIntersection(v) do { \
			tempIntersection[count] = v; \
			count += 1; \
		} while(0)

		for ( int j = 0; j < tempCount; j++) {
			if ( lineValues[j] <= 0 ) {
				appendIntersection(intersection[j]);
			}
			if ( ( lineValues[j] * lineValuesRotated[j] ) < 0 ) {
				Line<T> l1(intersection[j],intersectionRotated[j]);
				appendIntersection(l.Intersection(l1));
			}
		}

		for ( int j = 0; j < count; ++j) {
			intersection[j] = tempIntersection[j];
		}

	}

	return count;
}
Eigen::IOFormat Fmt(3,0,", ",", ","","","(",")");
template <typename T>
T ComputeArea(const Float2<T> * poly,
              const Float2<T> * polyRotated,
              const int size) {
	T result(0);
	for ( int i = 0; i < size; ++i) {
		result += poly[i].x() * polyRotated[i].y() - poly[i].y() * polyRotated[i].x();
	}
	return abs(result / T(2.0));
}

template <typename T>
T ComputeIoU( const Float2<T> * quad1,
              const Float2<T> * quad2) {
	Float2<T> quad1Rotated[4],quad2Rotated[4];
	Float2<T> intersection[8],intersectionRotated[8];
	for ( int i = 0; i < 4; ++i) {
		quad1Rotated[i] = quad1[i];
		quad2Rotated[i] = quad2[i];
		intersection[i] = quad2[i];
	}

	RotateArray(quad1Rotated,4);
	RotateArray(quad2Rotated,4);
	int size = ComputeIntersection(quad1,quad1Rotated,intersection);
	for ( int i = 0; i < size; ++i) {
		std::cerr << intersection[i].format(Fmt) << std::endl;
		intersectionRotated[i] = intersection[(i + 1) % size];
	}

	T area1 = ComputeArea(quad1,quad1Rotated,4);
	T area2 = ComputeArea(quad2,quad2Rotated,4);
	T areaIntersection(0.0);
	if ( size > 2) {
		areaIntersection = ComputeArea(intersection,intersectionRotated,size);
	}
	return areaIntersection / ( area1 + area2 - areaIntersection);
}


template <typename T>
struct IoUMatrixFunctor<CPUDevice, T> {
	void operator()(OpKernelContext* ctx, const CPUDevice& d,
	                const T* __restrict__ anchors,
	                const T* __restrict__ quads,
	                T* __restrict__ output,
	                const int anchors_size,
	                const int quads_size) {
		for ( int i = 0; i < anchors_size; ++i) {
			for ( int j = 0; j < quads_size; ++j) {
				Eigen::Matrix<T,2,1> quad1[4];
				Eigen::Matrix<T,2,1> quad2[4];
				std::cerr << "--------------------------------------------------------------------------------" << std::endl;
				for ( int k = 0; k < 4; ++k) {
					quad1[k] = Eigen::Matrix<T,2,1>(anchors[8 * i + 2 * k],
					                                anchors[8 * i + 2 * k + 1]);
					quad2[k] = Eigen::Matrix<T,2,1>(quads[8 * j + 2 * k],
					                                quads[8 * j + 2 * k + 1]);
					std::cerr << quad1[k].format(Fmt)
					          << "|" << quad2[k].format(Fmt)
					          << std::endl;
				}
				std::cerr << std::endl;
				T iou = ComputeIoU(quad1,quad2);
				output[i * quads_size + j] = iou;
				std::cerr << iou << std::endl;
			}
		}
	}
};

} // namespace functor


template <typename Device, typename T>
class IoUMatrixOp : public OpKernel {
public:
	explicit IoUMatrixOp(OpKernelConstruction * ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext * ctx) override {
		const Tensor & anchors = ctx->input(0);
		const Tensor & quads = ctx->input(1);

		const TensorShape & anchorsShape = anchors.shape();
		const TensorShape & quadsShape = quads.shape();


		OP_REQUIRES(ctx,
		            anchorsShape.dims() == 3
		            && anchorsShape.dim_size(1) == 4
		            && anchorsShape.dim_size(2) == 2,
		            errors::InvalidArgument("anchors must be of size [None, 4, 2], but is: ",
		                                    anchorsShape.DebugString()));
		OP_REQUIRES(ctx,
		            quadsShape.dims() == 3
		            && quadsShape.dim_size(1) == 4
		            && quadsShape.dim_size(2) == 2,
		            errors::InvalidArgument("quads must be of size [None, 4, 2], but is: ",
		                                    quadsShape.DebugString()));

		TensorShape outputShape;
		int anchorsSize = anchorsShape.dim_size(0);
		int quadsSize = quadsShape.dim_size(0);

		OP_REQUIRES_OK(ctx,TensorShapeUtils::MakeShape(gtl::ArraySlice<int32>({anchorsSize,quadsSize}),
		                                               &outputShape));

		Tensor * output = nullptr;
		OP_REQUIRES_OK(ctx,ctx->allocate_output(0,outputShape,&output));

		if (anchorsSize <= 0 || quadsSize <= 0) {
			return;
		}
		functor::IoUMatrixFunctor<Device,T>()(ctx,ctx->eigen_device<Device>(),
		                                      anchors.flat<T>().data(),
		                                      quads.flat<T>().data(),
		                                      output->flat<T>().data(),
		                                      anchorsSize,
		                                      quadsSize);
	}

private:
	TF_DISALLOW_COPY_AND_ASSIGN(IoUMatrixOp);
};

#define REGISTER(TYPE)	  \
	REGISTER_KERNEL_BUILDER(Name("ConvexQuadIoU>IoUMatrix").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
	                        IoUMatrixOp<CPUDevice, TYPE>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA
#define REGISTER(TYPE)                                                       \
	REGISTER_KERNEL_BUILDER(Name("ConvexQuadIoU>IoUMatrix").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
	                        IoUMatrixOp<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA

} // namespace convex_quad_iou
} // namespace tensorflow
