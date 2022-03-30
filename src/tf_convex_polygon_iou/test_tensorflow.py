import unittest
import tf_convex_polygon_iou._data as udata
import tf_convex_polygon_iou._tf_implementation as tfi
import numpy as np


class TensorflowTestCase(unittest.TestCase):

    def test_segment_intersection(self):
        for d in udata.SEGMENT_INTERSECTIONS:
            with self.subTest(str(d)):
                points = tfi.intersectSegment(
                    d.Segment1[None, ...],
                    d.Segment2[None, ...],
                )
                np.testing.assert_almost_equal(
                    points[0, ...].numpy(),
                    d.Expected,
                )

    def test_segment_intersection_ragged(self):
        segment1, segment2, expected = udata.SegmentIntersectionData.ragged(
            udata.SEGMENT_INTERSECTIONS)
        r = tfi.intersectSegment(segment1, segment2)
        np.testing.assert_almost_equal(
            r.to_tensor(default_value=np.Inf).numpy(),
            expected.to_tensor(default_value=np.Inf))

    def test_points_in_polygons(self):
        for d in udata.POINTS_IN_POLYGONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.pointsInPolygon(d.Polygon[None, ...],
                                        d.Points[None, ...])[0, ...].numpy(),
                    d.Expected,
                )

    def test_points_in_polygons_ragged(self):
        polygons, points, expected = udata.PointsInPolygonData.ragged(
            udata.POINTS_IN_POLYGONS)
        r = tfi.pointsInPolygon(polygons, points)
        np.testing.assert_almost_equal(r.numpy(), expected.numpy())

    def test_polygons_intersection(self):
        for d in udata.POLYGON_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.uniqueVertex(
                        tfi.intersectPolygons(
                            d.Polygon1[None, ...],
                            d.Polygon2[None, ...],
                        ))[0, ...].numpy(), d.Expected)

    def test_polygons_intersection_ragged(self):
        polygon1, polygon2, expected = udata.PolygonsIntersectionData.ragged(
            udata.POLYGON_INTERSECTIONS)

        r = tfi.intersectPolygons(polygon1, polygon2)
        r = tfi.uniqueVertex(r)
        np.testing.assert_almost_equal(
            r.to_tensor(default_value=float('Inf')).numpy(),
            expected.to_tensor(default_value=float('Inf')).numpy())

    def test_polygon_area(self):
        for d in udata.POLYGON_AREA:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(tfi.polygonArea(d.Polygon),
                                               d.Area)

    def test_polygon_area_ragged(self):
        polygons, area = udata.PolygonArea.ragged(udata.POLYGON_AREA)
        np.testing.assert_almost_equal(
            tfi.polygonArea(polygons).numpy(), area.numpy())

    def test_quad_IoU(self):
        for d in udata.POLYGON_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.IoUMatrix(d.Polygon1[None, ...],
                                  d.Polygon2[None, ...])[0, 0], d.IoU)

    def test_quad_IoUMatrix(self):
        polygons1, polygons2, expected = udata.PolygonsIntersectionData.IoUMatrix(
            udata.POLYGON_INTERSECTIONS)
        np.testing.assert_almost_equal(
            tfi.IoUMatrix(polygons1, polygons2).numpy(), expected.numpy())
