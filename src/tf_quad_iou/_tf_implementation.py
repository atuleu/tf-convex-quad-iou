import tensorflow as tf
import numpy as np


@tf.function
def intersectSegment(a: tf.Tensor, b: tf.Tensor, returnMask=False):
    """Intersects list of segments

    Literally taken from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Args:
        a (tf.Tensor): (N,2,2) endpoint of line
        b (tf.Tensor): (N,2,2) endpoint of line

    Returns:
        tf.RaggedTensor: (N,None,2) of the intersections
    """
    x1 = a[..., 0, 0]
    y1 = a[..., 0, 1]

    x2 = a[..., 1, 0]
    y2 = a[..., 1, 1]

    x3 = b[..., 0, 0]
    y3 = b[..., 0, 1]

    x4 = b[..., 1, 0]
    y4 = b[..., 1, 1]

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = tf.math.divide_no_nan((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4), D)
    u = tf.math.divide_no_nan((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2), D)

    inT = tf.math.logical_and(t >= 0, t <= 1)
    inU = tf.math.logical_and(u >= 0, u <= 1)
    inBoth = tf.math.logical_and(inT, inU)
    mask = tf.logical_and(tf.math.abs(D) > 1e-8, inBoth)
    res = a[..., 0, :] + t[..., None] * (a[..., 1, :] - a[..., 0, :])
    if returnMask:
        return res, mask
    return tf.RaggedTensor.from_row_lengths(tf.boolean_mask(res, mask),
                                            tf.cast(mask, tf.int64))


@tf.function
def pointsInPolygon(polygon, points):
    """Test if a point lies in a given polygon

    Args:
        polygon (tf.Tensor): (N,4,2) ragged tensor of vertices of the polygons (order does matter)
        point (tf.Tensor): (N,M,2) points to test for

    Returns:
        tf.Tensor: (N,M) boolean vector True if points lies inside P
    """
    polygonRolled = tf.roll(polygon, -1, axis=-2)
    side_criterion = (polygonRolled[..., 0] - polygon[..., 0])[:, None, :] * (
        points[..., 1][..., None] - polygon[..., 1][:, None, :]) - (
            points[..., 0][..., None] - polygon[..., 0][:, None, :]) * (
                polygonRolled[..., 1] - polygon[..., 1])[:, None, :]
    onUpEdge = tf.math.logical_and(
        points[..., 1][..., None] >= polygon[..., 1][:, None, :],
        points[..., 1][..., None] < polygonRolled[..., 1][:, None, :])

    onDownEdge = tf.math.logical_and(
        points[..., 1][..., None] < polygon[..., 1][:, None, :],
        points[..., 1][..., None] >= polygonRolled[..., 1][:, None, :])
    windingNumber = tf.cast(
        tf.math.logical_and(onUpEdge, side_criterion > 0.0),
        tf.int8) - tf.cast(
            tf.math.logical_and(onDownEdge, side_criterion < 0.0), tf.int8)

    windingNumber = tf.math.reduce_sum(windingNumber, axis=-1)
    return windingNumber != 0


def _edges(a):
    return tf.concat([
        a[:, :, None, :],
        tf.roll(a, -1, axis=-2)[:, :, None, :],
    ],
                     axis=-2)


@tf.function
def intersectQuads(a, b):
    """Computes the intersection of quads

    Args:
        a (tf.Tensor): (N,4,2) array of quad to compute intersection
        b (tf.Tensor): (N,4,2) array of quad to compute intersection

    Returns:
        tf.Tensor: (N,24,2) array of the intersection definition.  It
            may contains duplicate points.

    """
    N = tf.shape(a)[0]
    cornersAInB = pointsInPolygon(b, a)
    cornersBInA = pointsInPolygon(a, b)
    corners = tf.concat([a, b], axis=1)
    maskInside = tf.concat([cornersAInB, cornersBInA], axis=1)

    edgesA = tf.reshape(tf.tile(_edges(a), multiples=[1, 1, 4, 1]),
                        shape=[-1, 16, 2, 2])
    edgesB = tf.tile(_edges(b), multiples=[1, 4, 1, 1])  # N,16,2,2
    intersections, maskIntersections = intersectSegment(edgesA,
                                                        edgesB,
                                                        returnMask=True)

    mask = tf.concat([maskInside, maskIntersections], axis=1)
    sizes = tf.math.reduce_sum(tf.cast(mask, tf.int64), axis=-1)
    allCorners = tf.RaggedTensor.from_row_lengths(
        tf.boolean_mask(tf.concat([corners, intersections], axis=1), mask),
        sizes,
    )

    cornersAtCentroid = allCorners - tf.math.reduce_mean(allCorners,
                                                         axis=1)[:, None, :]
    angles = tf.math.atan2(
        cornersAtCentroid[..., 1],
        cornersAtCentroid[..., 0],
    )

    indexes = tf.argsort(angles.to_tensor(default_value=float('Inf')), axis=1)
    sortedCorners = tf.gather(allCorners.to_tensor(), indexes, batch_dims=1)
    finalMask = tf.RaggedTensor.from_row_lengths(
        tf.ones(tf.math.reduce_sum(sizes), tf.bool),
        sizes,
    )
    return tf.where(
        finalMask.to_tensor()[..., None],
        sortedCorners,
        sortedCorners[:, 0, :][:, None, :],
    )


def uniqueVertex(a):
    res = []
    for i in range(a.shape[0]):
        p = a[i, ...].numpy()
        indexes = np.unique(p, axis=0, return_index=True)[1]
        res.append(np.array([p[index, ...] for index in sorted(indexes)]))
    return tf.ragged.constant(res)


@tf.function
def polygonArea(a):
    """
    Computes the area of a polygon

    Args:
        a (tf.Tensor): (N,M,2) N polygon of M vertices

    Returns:
        tf.Tensor: (N,) The area of the polygons
    """
    rolledA = tf.roll(a, shift=-1, axis=-2)
    return 0.5 * tf.abs(
        tf.math.reduce_sum(
            a[..., 0] * rolledA[..., 1] - a[..., 1] * rolledA[..., 0],
            axis=-1))


@tf.function
def IoUMatrix(a, b):
    """Computes Intersection over union matrix of collection of boxes.

    Args:
        a (tf.Tensor): (N,4,2) a list of convex quads
        b (tf.Tensor): (M,4,2) a list of convex quads

    Returns:
        tf.Tensor: (N,M) the matrices of iou of a and b
    """
    numA = tf.shape(a)[0]
    numB = tf.shape(b)[0]
    flatA = tf.reshape(tf.tile(a, multiples=[1, numB, 1]), shape=[-1, 4, 2])
    flatB = tf.tile(b, multiples=[numA, 1, 1])

    intersections = intersectQuads(flatA, flatB)
    areaA = polygonArea(flatA)
    areaB = polygonArea(flatB)
    areaIntersection = polygonArea(intersections)
    return tf.reshape(areaIntersection / (areaA + areaB - areaIntersection),
                      shape=[numA, numB])
