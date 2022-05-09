from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

import tensorflow as tf

from typing import Optional, Text

iou_matrix_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_ops/iou/_iou_matrix_ops.so'))


@tf.function
@tf.autograph.experimental.do_not_convert
def iou_matrix(anchors: tf.Tensor,
               quads: tf.Tensor,
               name: Optional[Text] = None) -> tf.Tensor:
    """Computes the IoU matrices between anchors and quads.

    Computes the Intersection over Union between two sets of convex
    quadrilateral.  Vertices coordinates must be ordered in
    mathematical order arround the quad centroid.

    Args:
        anchors:  a [N,4,2] Tensor reprensenting the vertices of each anchor quads
        quads: a [M,4,2]  Tensor reprensenting the vertices of each quads to test.
    Returns:
        a [N,M] Tensor with the IoU between each sets of quads
    """
    with tf.name_scope(name or "iou_matrix"):
        return tf.stop_gradient(
            iou_matrix_ops.ConvexQuadIoU_IoUMatrix(anchors=anchors,
                                                   quads=quads))
