import numpy as np


def intersect_segment(a: np.ndarray, b: np.ndarray):
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
    if (abs(D) < 1e-8):
        return np.array([[]])

    t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t /= D

    u = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    u /= D

    if (t < 0 or t > 1 or u < 0 or u > 1):
        return np.array([[]])

    return np.array([[x1 + t * (x2 - x1), y1 + t * (y2 - y1)]])


def points_in_polygon(polygon: np.ndarray, points: np.ndarray):
    """Test if a point lies in a given polygon

    Args:
        polygon (np.ndarray): (None,2) vertices of the polygon (order does matter)
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
                polygon[:, 1]) * (points[:, 1][:, None] <= polygonRolled[:, 1])
    onDownEdge = (points[:, 1][:, None] <= polygon[:, 1]) * (
        points[:, 1][:, None] >= polygonRolled[:, 1])

    windingNumber = 2 * (onUpEdge * (side_criterion > 0.0)).astype(
        np.int32) + (onUpEdge * (side_criterion == 0.0)).astype(
            np.int32) - 2 * (onDownEdge *
                             (side_criterion < 0.0)).astype(np.int32) - (
                                 onDownEdge *
                                 (side_criterion == 0.0)).astype(np.int32)
    windingNumber = np.sum(windingNumber, axis=-1)
    return windingNumber != 0


def intersect_quads(a: np.ndarray, b: np.ndarray):
    """
    Computes the intersection of quads

    Args:
        a (np.ndarray): (4,2) array of quad to compute intersection
        b (np.ndarray): (4,2) array of quad to compute intersection

    Returns:
        np.ndarray: (N,2) array of the intersections it may contains duplicate
            points.
    """
    cornersAinB = points_in_polygon(b, a)
    cornersBinA = points_in_polygon(a, b)
    corners_inside = np.concatenate([a[cornersAinB, :], b[cornersBinA, :]],
                                    axis=0)
    intersections = np.zeros((16, 2))
    numberOfIntersections = 0
    for startA, endA in zip(a, np.roll(a, -1, axis=0)):
        for startB, endB in zip(b, np.roll(b, -1, axis=0)):
            intersection = intersect_segment(
                np.concatenate([endA[None, :], startA[None, :]], axis=0),
                np.concatenate([endB[None, :], startB[None, :]], axis=0),
            )
            if intersection.shape[1] == 2:
                intersections[numberOfIntersections] = intersection
                numberOfIntersections += 1

    allCorners = np.concatenate(
        [corners_inside, intersections[:numberOfIntersections, :]], axis=0)
    allCorners = np.unique(allCorners, axis=0)
    cornersAtCentroid = allCorners - np.mean(allCorners, axis=0)
    angles = np.arctan2(cornersAtCentroid[:, 1], cornersAtCentroid[:, 0])
    return allCorners[np.argsort(angles), :]
