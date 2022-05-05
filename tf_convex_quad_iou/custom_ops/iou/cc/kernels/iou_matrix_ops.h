#pragma once


#define EIGEN_USE_THREADS


#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace convex_quad_iou {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
using Float2 = Eigen::Matrix<T,2,1>;

template <typename T>
class Line {
	T a,b,c;
public:
	EIGEN_DEVICE_FUNC
	Line(const Float2<T> & v1,
	     const Float2<T> & v2)
		: a ( v2.y() - v1.y())
		, b ( v1.x() - v2.x())
		, c ( v2.x() * v1.y() - v2.y() * v1.x() ) {
	}
	EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
	T Project(const Float2<T> & p) const {
		return a * p.x() + b * p.y() + c;
	}

	EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
	Float2<T> Intersection(const Line & other) const {
		T w(a * other.b - b * other.a);
		return Float2<T>(b * other.c - c * other.b, c * other.a - a * other.c) / w;
	}
};

template<typename U>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void RotateArray(U * array,int size) {
	U temp = array[0];
	for ( int i = 1; i < size; ++i) {
		array[i-1] = array[i];
	}
	array[size-1] = temp;
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
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

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T ComputeIoU( const T * quad1Buffer,
              const T * quad2Buffer) {
	Float2<T> quad1[4],quad2[4];
	Float2<T> quad1Rotated[4],quad2Rotated[4];
	Float2<T> intersection[8],intersectionRotated[8];

	for ( int i = 0; i < 4; ++i) {
		quad1[i] = Eigen::Map<const Float2<T>>(quad1Buffer + 2 * i);
		quad2[i] = Eigen::Map<const Float2<T>>(quad2Buffer + 2 * i);
		quad1Rotated[i] = quad1[i];
		quad2Rotated[i] = quad2[i];
		intersection[i] = quad2[i];
	}

	RotateArray(quad1Rotated,4);
	RotateArray(quad2Rotated,4);
	int size = ComputeIntersection(quad1,quad1Rotated,intersection);
	for ( int i = 0; i < size; ++i) {
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
