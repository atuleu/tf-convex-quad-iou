import numpy as np


def intersectSegment(a: np.ndarray, b: np.ndarray):
    """Intersects two segments

    Literally taken from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Args:
        a (np.ndarray): (2,2) endpoint of line
        b (np.ndarray): (2,2) endpoint of line

    Returns:
        np.ndarray: (1,2) or (1,0) the intersection point if it exists
    """
    x1, y1 = a[0, :]
    x2, y2 = a[1, :]

    x3, y3 = b[0, :]
    x4, y4 = b[1, :]

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(D) < 1e-8:
        return np.zeros((0, 2))

    t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t /= D

    u = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    u /= D

    if t < 0 or t > 1 or u < 0 or u > 1:
        return np.zeros((0, 2))

    return np.array([[x1 + t * (x2 - x1), y1 + t * (y2 - y1)]])


def pointsInPolygon(polygon: np.ndarray, points: np.ndarray):
    """Test if a point lies in a given polygon

    Args:
        polygon (np.ndarray): (K,2) vertices of the polygon (order does matter)
        point (np.ndarray): (N,2) points to test for

    Returns:
        np.ndarray: (N,) boolean vector True if points lies inside P
    """
    polygonRolled = np.roll(polygon, -1, axis=0)

    side_criterion = (polygonRolled[:, 0] - polygon[:, 0]) * (
        points[:, 1][:, None] - polygon[:, 1]) - (
            points[:, 0][:, None] - polygon[:, 0]) * (polygonRolled[:, 1] -
                                                      polygon[:, 1])
    onUpEdge = (points[:, 1][:, None] >=
                polygon[:, 1]) * (points[:, 1][:, None] < polygonRolled[:, 1])
    onDownEdge = (points[:, 1][:, None] < polygon[:, 1]) * (
        points[:, 1][:, None] >= polygonRolled[:, 1])

    windingNumber = (onUpEdge * (side_criterion > 0.0)).astype(
        np.int32) - (onDownEdge * (side_criterion < 0.0)).astype(np.int32)
    windingNumber = np.sum(windingNumber, axis=-1)
    return windingNumber != 0


def intersectPolygons(a: np.ndarray, b: np.ndarray):
    """
    Computes the intersection of quads

    Args:
        a (np.ndarray): (K,2) array of quad to compute intersection
        b (np.ndarray): (K,2) array of quad to compute intersection

    Returns:
        np.ndarray: (N,2) array of the intersections it may contains duplicate
            points. N can go up to K ** 2 + 2 * K.
    """
    cornersAinB = pointsInPolygon(b, a)
    cornersBinA = pointsInPolygon(a, b)

    corners_inside = np.concatenate([a[cornersAinB, :], b[cornersBinA, :]],
                                    axis=0)
    intersections = np.zeros((16, 2))
    numberOfIntersections = 0
    for startA, endA in zip(a, np.roll(a, -1, axis=0)):
        for startB, endB in zip(b, np.roll(b, -1, axis=0)):
            intersection = intersectSegment(
                np.concatenate([endA[None, :], startA[None, :]], axis=0),
                np.concatenate([endB[None, :], startB[None, :]], axis=0),
            )
            if intersection.shape[0] == 1:
                intersections[numberOfIntersections] = intersection
                numberOfIntersections += 1
    allCorners = np.concatenate(
        [corners_inside, intersections[:numberOfIntersections, :]], axis=0)
    if allCorners.shape[0] == 0:
        return np.array([], dtype=np.float32)
    cornersAtCentroid = allCorners - np.mean(allCorners, axis=0)
    angles = np.arctan2(cornersAtCentroid[:, 1], cornersAtCentroid[:, 0])
    return allCorners[np.argsort(angles), :]


def polygonArea(a: np.ndarray):
    """
    Computes the area of a polygon

    Args:
        a (np.ndarray): (N,2)
    Returns:
        float32: the area of the polygon
    """
    if len(a.shape) < 2 or a.shape[-2] == 0:
        return 0.0
    rolledA = np.roll(a, -1, axis=-2)
    return 0.5 * abs(np.sum(a[:, 0] * rolledA[:, 1] - a[:, 1] * rolledA[:, 0]))


def uniqueVertex(a: np.ndarray):
    """Returns unique polygon vertices, keeping the order

    Args:
        a (np.ndarray): (N,2) a list of vertices

    Returns:
        np.ndarray: unique vertices in a, keeping it sorted
    """
    indexes = np.unique(a, axis=0, return_index=True)[1]
    return np.array([a[index, ...] for index in sorted(indexes)])


def IoUMatrix(a: np.ndarray, b: np.ndarray):
    """Computes Intersection over union matrix of collection of boxes.

    Args:
        a (np.ndarray): (N,4,2) a list of convex quads
        b (np.ndarray): (M,4,2) a list of convex quads

    Returns:
        np.ndarray: (N,M) the matrices of iou of a and b
    """
    res = np.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            areaA = polygonArea(a[i, ...])
            areaB = polygonArea(b[j, ...])
            inter = intersectPolygons(a[i, ...], b[j, ...])
            areaInter = polygonArea(inter)
            res[i, j] = areaInter / (areaA + areaB - areaInter)
    return res
