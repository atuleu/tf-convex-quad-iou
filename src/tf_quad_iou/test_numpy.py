import unittest
import tf_quad_iou._data as udata
import tf_quad_iou._numpy_implementation as npi
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
        for d in udata.POINT_IN_QUADS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.pointsInPolygon(d.Polygon, d.Points),
                    d.Expected,
                )

    def test_boxes_intersection(self):
        for d in udata.BOX_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    npi.uniqueVertex(npi.intersectQuads(
                        d.Box1,
                        d.Box2,
                    )), d.Expected)

    def test_boxes_area(self):
        for d in udata.POLYGON_AREA:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(npi.polygonArea(d.Polygon),
                                               d.Area)
