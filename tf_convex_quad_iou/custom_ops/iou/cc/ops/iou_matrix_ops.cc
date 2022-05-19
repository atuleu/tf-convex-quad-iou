#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace convex_quad_iou {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

REGISTER_OP("ConvexQuadIoU>IoUMatrix")
	.Input("anchors: T")
	.Input("quads: T")
	.Output("output: T")
	.Attr("T: {half, float, double}")
	.SetShapeFn([](InferenceContext * c) {
		            for ( int i = 0; i < 2; ++i ) {
			            ShapeHandle input;
			            TF_RETURN_IF_ERROR(c->WithRank(c->input(i),3,&input));
			            DimensionHandle dim1,dim2;
			            TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(i),1),4,&dim1));
			            TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(i),2),2,&dim2));
		            }

		            c->set_output(0,c->Matrix(c->Dim(c->input(0),0),
		                                      c->Dim(c->input(1),0)));

		            return Status::OK();
	            })
	.Doc(R"doc(IoU Matrix Op.)doc");

REGISTER_OP("ConvexQuadIoU>QuadCopy")
	.Input("input: T")
	.Output("output: T")
	.Attr("T: {half, float, double}")
	.SetShapeFn([](InferenceContext * c) {
		            for ( int i = 0; i < 1; ++i ) {
			            ShapeHandle input;
			            TF_RETURN_IF_ERROR(c->WithRank(c->input(i),3,&input));
			            DimensionHandle dim1,dim2;
			            TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(i),1),4,&dim1));
			            TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(i),2),2,&dim2));
		            }
		            c->set_output(0,c->input(0));
		            return Status::OK();
	            })
	.Doc(R"doc(Copy Quad Op.)doc");


} // namespace convex_quad_iou
} // namespace tensorflow
