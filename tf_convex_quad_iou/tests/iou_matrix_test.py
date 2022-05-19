import numpy as np
import math
from tf_convex_quad_iou import iou_matrix, quad_copy
import tensorflow as tf
from dataclasses import dataclass
import pytest

sqrt2 = math.sqrt(2)

anchors = np.array([[
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
], [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
], [
    [0, sqrt2],
    [-sqrt2, 0],
    [0, -sqrt2],
    [sqrt2, 0],
], [
    [-1, 1],
    [0, 0],
    [1, 0],
    [2, 1],
], [
    [1, 1],
    [2, 1],
    [2, 2],
    [1, 2],
], [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
]],
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
    [1 / 3, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])


def test_iou_matrix():
    r = iou_matrix(anchors=anchors, quads=quads, debug=True).numpy()
    print(r)
    print(expected)
    np.testing.assert_almost_equal(
        r,
        expected,
    )


def _build_anchors():

    @dataclass
    class AnchorBox:
        x: float
        y: float
        h: float
        w: float
        angle: float

        def quad(self):
            c = math.cos(self.angle)
            s = math.sin(self.angle)
            rot = np.array([[c, s], [-s, c]], dtype=np.float32)
            return np.array([self.x, self.y], dtype=np.float32) + np.array(
                [
                    rot @ [-self.w, -self.h],
                    rot @ [self.w, -self.h],
                    rot @ [self.w, self.h],
                    rot @ [-self.w, self.h],
                ],
                dtype=np.float32)

    angles = [0, math.pi / 4]
    sizes = [28, 14]
    aspectRatios = [1, 2, 3, 1 / 2, 1 / 3]
    aboxes = []
    for s in sizes:
        for y in range(s):
            cy = (0.5 + y) / s
            for x in range(s):
                cx = (0.5 + x) / s
                aboxes.extend([
                    AnchorBox(x=cx,
                              y=cy,
                              angle=a,
                              h=1 / s * math.sqrt(ar),
                              w=1 / (s * math.sqrt(ar))) for a in angles
                    for ar in aspectRatios
                ])

    res = np.ndarray(shape=(len(aboxes), 4, 2), dtype=np.float32)
    for i, abox in enumerate(aboxes):
        res[i, :, :] = abox.quad()
    return res


_anchors = tf.constant(_build_anchors(), dtype=tf.float32)


@dataclass
class ConfusedData:
    Quads: np.array
    Expected: np.array


confused = [
    ConfusedData(Quads=np.array([
        [
            [+3.1667882e-01, -1.13034469e-02],
            [+3.458394e-01, -4.0464036e-02],
            [+4.3332119e-01, +4.7017735e-02],
            [+4.041606e-01, +7.617833e-02],
        ],
        [
            [+3.2142856e-01, +1.862645149230957e-09],
            [+3.57142865657806e-01, -3.5714285e-02],
            [+4.28571432828903e-01, +3.571428328289032e-02],
            [+3.928571343421936e-01, +7.14285746216774e-02],
        ],
    ]),
                 Expected=np.array([[1.0, 0.68989813], [0.68989813, 1.0]],
                                   dtype=np.float32)),
    ConfusedData(Quads=np.array([
        [
            [0.32142857, 0.89285713],
            [0.4642857, 0.89285713],
            [0.4642857, 1.0357143],
            [0.32142857, 1.0357143],
        ],
        [
            [0.29184186, 0.96428573],
            [0.39285713, 0.86327046],
            [0.4938724, 0.96428573],
            [0.39285713, 1.065301],
        ],
    ]),
                 Expected=np.array([[1.0, 0.707107], [0.707107, 1.0]],
                                   dtype=np.float32)),
    ConfusedData(Quads=np.array([
        [
            [-0.03265049, -0.00739667],
            [0.06836477, -0.00739667],
            [0.06836477, 0.04311096],
            [-0.03265049, 0.04311096],
        ],
        [
            [0.07449237, -0.00739667],
            [0.17550763, -0.00739667],
            [0.17550763, 0.04311096],
            [0.07449237, 0.04311096],
        ],
    ]),
                 Expected=np.array([[1.0, 0.0], [0.0, 1.0]],
                                   dtype=np.float32)),
]


@pytest.mark.parametrize("d", confused)
def test_iou_matrix_confused(d):
    quads = tf.constant(d.Quads, tf.float32)
    iou = iou_matrix(quads, quads)
    np.testing.assert_almost_equal(iou.numpy(), d.Expected, decimal=6)


@pytest.mark.slow
@pytest.mark.parametrize("index", range(tf.shape(_anchors)[0]))
def test_iou_matrix_with_anchor(index):
    iou = iou_matrix(_anchors, _anchors[index, ...][None, ...])
    amax = tf.argmax(iou, axis=0)
    maxIoU = tf.math.reduce_max(iou).numpy()
    np.testing.assert_equal(
        amax.numpy(), [index],
        err_msg=
        f"between quads {_anchors[amax.numpy()[0],...].numpy()} and {_anchors[index,...].numpy()}"
    )
    np.testing.assert_allclose(
        maxIoU,
        1.0,
        rtol=1e-2,
        err_msg=
        f"expected maximal IoU to be 1.0 for anchor {index}, got {maxIoU}")
