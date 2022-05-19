#pragma once


#define EIGEN_USE_THREADS


#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

#include "tensorflow/core/framework/op_kernel.h"

#include <iomanip>

namespace tensorflow {
namespace convex_quad_iou {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
using Float2 = Eigen::Matrix<T,2,1>;

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T isLeft(const Float2<T> & p1,
         const Float2<T> & p2,
         const Float2<T> & a) {
	return ( p2.x() - p1.x() ) * ( a.y() - p1.y()) - ( a.x() - p1.x() ) * ( p2.y() - p1.y() );
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool isInside(const Float2<T> & vertex,
              const Float2<T> * quad) {

	int wn = 0;
	for ( int i = 0; i < 4; ++i ) {
		int j = (i+1) % 4;
		T sideCriterion = isLeft(quad[i],quad[j],vertex);

		bool onUpEdge = (vertex.y() >= quad[i].y()) && ( vertex.y() <= quad[j].y() );
		bool onDownEdge = (vertex.y() <= quad[i].y()) && ( vertex.y() >= quad[j].y() );

		if ( onUpEdge && sideCriterion >= 0) {
			wn += sideCriterion == 0.0 ? 1 : 2;
		}
		if ( onDownEdge && sideCriterion <= 0 ) {
			wn -= sideCriterion == 0.0 ? 1 : 2;
		}
	}
	return wn != 0;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool intersectSegments(const Float2<T> & a1,
                       const Float2<T> & a2,
                       const Float2<T> & b1,
                       const Float2<T> & b2,
                       Float2<T> * result) {
	T D = (a1.x() - a2.x() ) * ( b1.y() - b2.y() ) - ( a1.y() - a2.y() ) * ( b1.x() - b2.x() );
	if ( abs(D) < 1e-6 ) {
		// parallel case, we have no intersection.
		return false;
	}
	T t = (a1.x() - b1.x() ) * ( b1.y() - b2.y() ) - ( a1.y() - b1.y() ) * ( b1.x() - b2.x() );
	t /= D;
	T u = (a1.x() - b1.x() ) * ( a1.y() - a2.y() ) - ( a1.y() - b1.y() ) * ( a1.x() - a2.x() );
	u /= D;
	if ( t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0 ) {
		return false;
	}
	*result = a1 + t * ( a2 - a1);
	return true;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
int insertNoDuplicate(Float2<T> * array,
                      const Float2<T> & v,
                      int size) {
	for ( int i = 0; i < size; ++i ) {
		if ( ( array[i] - v ).array().abs().maxCoeff() < 1e-6 ) {
			return size;
		}
	}
	array[size] = v;
	return size + 1;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
int computeIntersectionVertices(const Float2<T> * quad1,
                                const Float2<T> * quad2,
                                Float2<T> * intersection) {
	int size = 0;
	bool v1Inside2[4];
	bool v2Inside1[4];

	for ( int i = 0; i < 4; ++i) {
		v1Inside2[i] = isInside(quad1[i],quad2);
	}



	for ( int i = 0; i < 4; ++i) {
		v2Inside1[i] = isInside(quad2[i],quad1);
	}


	for ( int i = 0; i < 4; ++i) {
		if ( v1Inside2[i] ) {
			size = insertNoDuplicate(intersection,quad1[i],size);
		}
		if ( v2Inside1[i] ) {
			size = insertNoDuplicate(intersection,quad2[i],size);
		}
	}

	for ( int i = 0; i < 4; ++i) {
		// we prune away non-intersecting edges (or consecutive concurrent edges)
		int nextI = (i+1) % 4;
		if ( v1Inside2[i] == true && v1Inside2[nextI] == true ) {
			continue;
		}

		for ( int j = 0; j < 4; ++j ) {
			int nextJ = (j+1) % 4;
			if ( v2Inside1[j] == true && v2Inside1[nextJ] == true ) {
				continue;
			}

			Float2<T> res;
			if ( intersectSegments(quad1[i],
			                       quad1[nextI],
			                       quad2[j],
			                       quad2[nextJ],
			                       &res) ) {

				size = insertNoDuplicate(intersection,res,size);
				if ( size >= 8 ) {
					return size;
				}
			}
		}
	}
	return size;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void completeVertices(Float2<T> * vertices,
                     int size,
                     const int finalSize) {
	if ( size <= 0 ) {
		vertices[0] = Float2<T>::Zero();
		size = 1;
	}
	for ( ; size < finalSize; ++size ) {
		vertices[size] = vertices[size-1];
	}
}

template <typename U>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void swap( U * array, int i, int j) {
	U tmp = array[i];
	array[i] = array[j];
	array[j] = tmp;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void sortVerticesArroundCentroid(Float2<T> * vertices) {
	T angles[8];
	Float2<T> centroid = Float2<T>::Zero();
	for ( int i = 0; i < 8; ++i) {
		centroid += vertices[i] / T{8.0};
	}

	for ( int i = 0; i < 8; ++i) {
		Float2<T> atCentroid = vertices[i] - centroid;
		angles[i] = T {atan2(atCentroid.y(),atCentroid.x())};
	}

#define compareSwap(i,j,values,otherArray) do {	  \
		if ( values[i] > values[j] ) {\
			swap(values,i,j); \
			swap(otherArray,i,j); \
		} \
	}while(0)

	// 8 input optimal sorting network
	// (http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N8L19D6

	compareSwap(0,2,angles,vertices);
	compareSwap(1,3,angles,vertices);
	compareSwap(4,6,angles,vertices);
	compareSwap(5,7,angles,vertices);

	compareSwap(0,4,angles,vertices);
	compareSwap(1,5,angles,vertices);
	compareSwap(2,6,angles,vertices);
	compareSwap(3,7,angles,vertices);

	compareSwap(0,1,angles,vertices);
	compareSwap(2,3,angles,vertices);
	compareSwap(4,5,angles,vertices);
	compareSwap(6,7,angles,vertices);

	compareSwap(2,4,angles,vertices);
	compareSwap(3,5,angles,vertices);

	compareSwap(1,4,angles,vertices);
	compareSwap(3,6,angles,vertices);

	compareSwap(1,2,angles,vertices);
	compareSwap(3,4,angles,vertices);
	compareSwap(5,6,angles,vertices);
}



template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void computeIntersection(const Float2<T> * quad1,
                         const Float2<T> * quad2,
                         Float2<T> * intersection) {
	int size = computeIntersectionVertices(quad1,quad2,intersection);
	completeVertices(intersection,size,8);
	sortVerticesArroundCentroid(intersection);
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T computeArea(const Float2<T>* vertices,
              const int size) {
	T result = T {0.0};
	for ( int i = 0; i < size; ++i) {
		int j = (i + 1) % size;
		result += vertices[i].x() * vertices[j].y() - vertices[i].y() * vertices[j].x();
	}
	return abs(result) / T{2.0};
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T ComputeIoU( const T * quad1Buffer,
              const T * quad2Buffer) {
	Float2<T> quad1[4],quad2[4];
	Float2<T> intersection[8];
	for ( int i = 0; i < 4; ++i) {
		quad1[i] = Eigen::Map<const Float2<T>>(quad1Buffer + 2 * i);
		quad2[i] = Eigen::Map<const Float2<T>>(quad2Buffer + 2 * i);
	}

	computeIntersection(quad1,quad2,intersection);

	T area1 = computeArea(quad1,4);
	T area2 = computeArea(quad2,4);
	T areaIntersection(0.0);
	if ( area1 > T{0.0} && area2 > T{0.0} ) {
		areaIntersection = std::min(computeArea(intersection,8),std::min(area1,area2));
	} else {
		area1 = T{1.0};
		area2 = T{1.0};
	}
	return areaIntersection / ( area1 + area2 - areaIntersection);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void CopyQuad(const T * input,
              T * output) {
	output[0] = input[0];
	output[1] = input[1];
	output[2] = input[2];
	output[3] = input[3];
	output[4] = input[4];
	output[5] = input[5];
	output[6] = input[6];
	output[7] = input[7];
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

template <typename Device,typename T>
struct QuadCopyFunctor {
	void operator()(OpKernelContext * ctx, const Device & d,
	                const T * __restrict__ input,
	                T * __restrict__ output,
	                const int size);
};



} // namespace functor
} // namespace convex_quad_iou
} // namespace tensorflow
