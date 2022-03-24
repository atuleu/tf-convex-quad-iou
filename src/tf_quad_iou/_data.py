from dataclasses import dataclass
import numpy as np
import math
import tensorflow as tf


@dataclass
class SegmentIntersectionData:
    Segment1: np.ndarray
    Segment2: np.ndarray
    Expected: np.ndarray

    @staticmethod
    def ragged(l):
        return (
            tf.concat([d.Segment1[None, :, :] for d in l], axis=0),
            tf.concat([d.Segment2[None, :, :] for d in l], axis=0),
            tf.ragged.constant([d.Expected for d in l]),
        )


SEGMENT_INTERSECTIONS = [
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [0, 1]], dtype=np.float32),
        Segment2=np.array([[0, 0], [1, 0]], dtype=np.float32),
        Expected=np.array([[0, 0]], dtype=np.float32),
    ),
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [1, 1]], dtype=np.float32),
        Segment2=np.array([[1, 0], [0, 1]], dtype=np.float32),
        Expected=np.array([[0.5, 0.5]], dtype=np.float32),
    ),
    # co-linear condition
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [0, 1]], dtype=np.float32),
        Segment2=np.array([[0, 0], [0, 2]], dtype=np.float32),
        Expected=np.zeros((0, 2), dtype=np.float32),
    ),
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [0, 1]], dtype=np.float32),
        Segment2=np.array([[1, 0], [1, 1]], dtype=np.float32),
        Expected=np.zeros(shape=(0, 2), dtype=np.float32),
    ),
    # intersection outside of segment
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [0, 1]], dtype=np.float32),
        Segment2=np.array([[1, 1], [2, 1]], dtype=np.float32),
        Expected=np.zeros(shape=(0, 2), dtype=np.float32),
    ),
    SegmentIntersectionData(
        Segment1=np.array([[0, 0], [0, 1]], dtype=np.float32),
        Segment2=np.array([[3, 2], [2, 2]], dtype=np.float32),
        Expected=np.zeros(shape=(0, 2), dtype=np.float32),
    ),
]


@dataclass
class PointsInPolygonData:
    Polygon: np.ndarray
    Points: np.ndarray
    Expected: np.ndarray

    @staticmethod
    def ragged(l):
        return (
            tf.constant([d.Polygon for d in l]),
            tf.constant([d.Points for d in l]),
            tf.constant([d.Expected for d in l]),
        )


POINT_IN_QUADS = [
    # -- With a square --o
    PointsInPolygonData(
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32),
        np.array(
            [
                [0, 0],
                [1, 2],
                [1, -2],
                [-1, -1],
                [1, 1],
                [-1, 1],
                [1, -1],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        ),
    ),
    # -- With an hourglass --
    # just above the center is inside
    PointsInPolygonData(
        np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32),
        np.array(
            [
                [0.0, 1.0e-6],
                [1.0e-6, 0.0],
                [0.0, 0.0],
                [-1, -1],
                [1, 1],
                [-1, 1],
                [1, -1],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
            dtype=bool,
        ),
    ),
]


@dataclass
class BoxesIntersectionData:
    Box1: np.ndarray
    Box2: np.ndarray
    Expected: np.ndarray
    IoU: float

    @staticmethod
    def ragged(l):
        return (
            tf.constant([d.Box1 for d in l]),
            tf.constant([d.Box2 for d in l]),
            tf.ragged.constant([d.Expected for d in l])
        )

    @staticmethod
    def IoUMatrix(l):
        defaultBox = l[0].Box1
        filtered = [d for d in l if (d.Box1 == defaultBox).all()]
        return (
            tf.constant([d.Box2 for d in filtered]),
            tf.concat([defaultBox[None, ...], [[[0, 0]] * 4]], axis=0),
            tf.concat(
                [[[d.IoU] for d in filtered], [[0.0]] * len(filtered)],
                axis=-1,
            ),
        )


sqrt2 = math.sqrt(2)
BOX_INTERSECTIONS = [
    BoxesIntersectionData(
        Box1=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
            ],
            dtype=np.float32,
        ),
        IoU=1.0,
    ),
    BoxesIntersectionData(
        Box1=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [0, 0],
                [0, 2],
                [2, 2],
                [2, 0],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.float32,
        ),
        IoU=1 / 7,
    ),
    BoxesIntersectionData(
        Box1=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [0, sqrt2],
                [sqrt2, 0],
                [0, -sqrt2],
                [-sqrt2, 0],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [-1, -sqrt2 + 1],
                [-sqrt2 + 1, -1],
                [sqrt2 - 1, -1],
                [1, -sqrt2 + 1],
                [1, sqrt2 - 1],
                [sqrt2 - 1, 1],
                [-sqrt2 + 1, 1],
                [-1, sqrt2 - 1],
            ],
            dtype=np.float32,
        ),
        IoU=(1 - (sqrt2 - 1) ** 2) / (1 + (sqrt2 - 1) ** 2),
    ),
    BoxesIntersectionData(
        Box1=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [-1, 1],
                [0, 0],
                [1, 0],
                [2, 1],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [-1, 1],
            ],
            dtype=np.float32,
        ),
        IoU=(2 - 1 / 2) / (4 + 1 / 2),
    ),
    BoxesIntersectionData(
        Box1=np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [1, 1],
                [2, 1],
                [2, 2],
                [1, 2],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [1, 1],
            ],
            dtype=np.float32,
        ),
        IoU=0.0,
    ),
    BoxesIntersectionData(
        Box1=np.array(
            [
                [0, 0],
                [8, 0],
                [8, 10],
                [0, 10],
            ],
            dtype=np.float32,
        ),
        Box2=np.array(
            [
                [0, 0],
                [6, 0],
                [6, 8],
                [0, 8],
            ],
            dtype=np.float32,
        ),
        Expected=np.array(
            [
                [0, 0],
                [6, 0],
                [6, 8],
                [0, 8],
            ],
            dtype=np.float32,
        ),
        IoU=48 / 80,
    ),
]


@dataclass
class PolygonArea:
    Polygon: np.ndarray
    Area: np.float32

    def ragged(l):
        mask = tf.ragged.constant([[True] * d.Polygon.shape[0] for d in l])
        vertices = tf.ragged.constant([d.Polygon for d in l]).to_tensor()
        polygons = tf.where(
            mask.to_tensor()[..., None], vertices, vertices[:, 0, :][:, None, :]
        )
        return polygons, tf.constant([d.Area for d in l])


POLYGON_AREA = [
    PolygonArea(
        Polygon=np.array(
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=np.float32,
        ),
        Area=1,
    ),
    PolygonArea(
        Polygon=np.array(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
            ],
            dtype=np.float32,
        ),
        Area=1,
    ),
]
