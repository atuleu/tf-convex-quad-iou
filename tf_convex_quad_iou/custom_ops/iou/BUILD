licenses(["notice"])  # MIT

package(default_visibility = ["//visibility:public"])

load("//tf_convex_quad_iou:tf_convex_quad_iou.bzl", "custom_op_library")

custom_op_library(
    name = "_iou_matrix_ops.so",
    srcs = [
        "cc/kernels/iou_matrix_ops.cc",
        "cc/kernels/iou_matrix_ops.h",
        "cc/ops/iou_matrix_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/iou_matrix_ops.h",
        "cc/kernels/iou_matrix_ops_gpu.cu.cc",
    ],
)