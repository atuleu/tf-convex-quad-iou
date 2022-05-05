import numpy as np
import math
from tf_convex_quad_iou import iou_matrix

sqrt2 = math.sqrt(2)

anchors = np.array([
    [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ],
    [
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
    ],
    [
        [0, sqrt2],
        [-sqrt2, 0],
        [0, -sqrt2],
        [sqrt2, 0],
    ],
    [
        [-1, 1],
        [0, 0],
        [1, 0],
        [2, 1],
    ],
    [
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
    ],
],
                   dtype=np.float32)

quads = np.array([
    [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ],
    [
        [100, 200],
        [101, 200],
        [101, 201],
        [100, 201],
    ],
],
                 dtype=np.float32)

expected = np.array([
    [1.0, 0.0],
    [1 / 7, 0.0],
    [(1 - (sqrt2 - 1)**2) / (1 + (sqrt2 - 1)**2), 0.0],
    [(2 - 1 / 2) / (4 + 1 / 2), 0.0],
    [0.0, 0.0],
])


def test_iou_matrix():
    np.testing.assert_almost_equal(
        iou_matrix(anchors=anchors, quads=quads).numpy(),
        expected,
    )
