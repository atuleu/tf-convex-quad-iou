from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

iou_matrix_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_ops/iou/_iou_matrix_ops.so'))
iou_matrix = iou_matrix_ops.ConvexQuadIoU_IoUMatrix
