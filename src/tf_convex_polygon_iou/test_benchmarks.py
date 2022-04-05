import unittest
import tensorflow as tf
import numpy as np
import math
from tf_convex_polygon_iou._tf_implementation import IoUMatrix


class BenchmarkTestCase(unittest.TestCase):

    def buildArgs(self):

        def _rotateBox(x, y, w, h, angle):
            rot = np.array([[math.cos(angle), math.sin(angle)],
                            [math.sin(-angle),
                             math.cos(angle)]])
            return np.array([x, y]) + 0.5 * np.array([
                rot @ [-w, -h],
                rot @ [w, -h],
                rot @ [w, h],
                rot @ [-w, h],
            ])

        def _anchorsAt(depth, angles, ratios):
            anchors = [(a, r) for a in angles for r in ratios]
            res = np.zeros((depth * depth * len(angles) * len(ratios), 4, 2))
            idx = 0
            for y in range(depth):
                cy = (0.5 + y) / depth
                for x in range(depth):
                    cx = (0.5 + x) / depth
                    for angle, ratio in anchors:
                        w = 1 / depth * math.sqrt(ratio)
                        h = 1 / depth / math.sqrt(ratio)
                        res[idx, :, :] = _rotateBox(cx, cy, w, h, angle)
                        idx += 1

            return res

        fmSizes = [28, 14, 7]
        angles = [0, math.pi / 4]
        ratios = [1, 2, 3, 1 / 2, 1 / 3]
        anchors = np.zeros((0, 4, 2))
        for s in fmSizes:
            anchors = np.concatenate(
                [anchors, _anchorsAt(s, angles, ratios)], axis=0)
        quads = np.array([
            _rotateBox(0.23, 0.42, 0.2, 0.4, np.deg2rad(24)),
            _rotateBox(0.8, 0.6, 0.4, 0.5, np.deg2rad(-34)),
            _rotateBox(0.5, 0.52, 0.8, 0.8, np.deg2rad(43)),
            _rotateBox(0.7, 0.1, 0.3, 0.2, np.deg2rad(5)),
        ])
        return tf.constant(anchors, tf.float32), tf.constant(quads, tf.float32)

    def test_benchmark(self):
        with tf.compat.v1.Session() as sess:
            anchors, quads = self.buildArgs()
            bm = tf.test.Benchmark()
            bm.run_op_benchmark(
                sess=sess,
                min_iters=40,
                op_or_tensor=IoUMatrix(anchors, quads),
                store_trace=True,
            )
