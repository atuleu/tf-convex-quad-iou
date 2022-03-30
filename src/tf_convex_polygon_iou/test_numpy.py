import unittest
import tf_convex_polygon_iou._data as udata
import tf_convex_polygon_iou._numpy_implementation as npi
import numpy as np


class NumpyTestCase(unittest.TestCase):

    def test_segment_intersection(self):
        for d in udata.SEGMENT_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.intersectSegment(
                        d.Segment1,
                        d.Segment2,
                    ),
                    d.Expected,
                )

    def test_points_in_polygon(self):
        for d in udata.POINTS_IN_POLYGONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.pointsInPolygon(d.Polygon, d.Points),
                    d.Expected,
                )

    def test_polygons_intersection(self):
        for d in udata.POLYGON_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.uniqueVertex(
                        npi.intersectPolygons(
                            d.Polygon1,
                            d.Polygon2,
                        )), d.Expected)

    def test_polygon_area(self):
        for d in udata.POLYGON_AREA:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(npi.polygonArea(d.Polygon),
                                               d.Area)

    def test_IoU(self):
        for d in udata.POLYGON_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.IoUMatrix(d.Polygon1[None, ...],
                                  d.Polygon2[None, ...])[0, 0], d.IoU)

    def test_IoUMatrix(self):
        polygons1, polygons2, expected = udata.PolygonsIntersectionData.IoUMatrix(
            udata.POLYGON_INTERSECTIONS)
        np.testing.assert_almost_equal(
            npi.IoUMatrix(polygons1.numpy(), polygons2.numpy()),
            expected.numpy())
