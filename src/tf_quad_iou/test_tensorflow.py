import unittest
import tf_quad_iou._data as udata
import tf_quad_iou._tf_implementation as tfi
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

    def test_points_in_polygon(self):
        for d in udata.POINT_IN_QUADS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.pointsInPolygon(d.Polygon[None, ...],
                                        d.Points[None, ...])[0, ...].numpy(),
                    d.Expected,
                )

    def test_points_in_polygon_ragged(self):
        polygons, points, expected = udata.PointsInPolygonData.ragged(
            udata.POINT_IN_QUADS)
        r = tfi.pointsInPolygon(polygons, points)
        np.testing.assert_almost_equal(r.numpy(), expected.numpy())

    def test_boxes_intersection(self):
        for d in udata.BOX_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.uniqueVertex(
                        tfi.intersectQuads(
                            d.Box1[None, ...],
                            d.Box2[None, ...],
                        ))[0, ...].numpy(), d.Expected)

    def test_boxes_intersection_ragged(self):
        quad1, quad2, expected = udata.BoxesIntersectionData.ragged(
            udata.BOX_INTERSECTIONS)

        r = tfi.intersectQuads(quad1, quad2)
        r = tfi.uniqueVertex(r)
        np.testing.assert_almost_equal(
            r.to_tensor(default_value=float('Inf')).numpy(),
            expected.to_tensor(default_value=float('Inf')).numpy())

    def test_boxes_area(self):
        for d in udata.POLYGON_AREA:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(tfi.polygonArea(d.Polygon),
                                               d.Area)

    def test_boxes_area_ragged(self):
        polygons, area = udata.PolygonArea.ragged(udata.POLYGON_AREA)
        np.testing.assert_almost_equal(
            tfi.polygonArea(polygons).numpy(), area.numpy())

    def test_quad_IoU(self):
        for d in udata.BOX_INTERSECTIONS:
            with self.subTest(str(d)):
                np.testing.assert_almost_equal(
                    tfi.IoUMatrix(d.Box1[None, ...], d.Box2[None, ...])[0, 0],
                    d.IoU)

    def test_quad_IoUMatrix(self):
        quads1, quads2, expected = udata.BoxesIntersectionData.IoUMatrix(
            udata.BOX_INTERSECTIONS)
        np.testing.assert_almost_equal(
            tfi.IoUMatrix(quads1, quads2).numpy(), expected.numpy())
