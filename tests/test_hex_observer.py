"""
Tests for HexObserver pure math and geometry methods.
These are largely hardware-free and give high coverage returns.
"""

import pytest
import math
import numpy as np
import unittest.mock as mock
import sys, os

# Patch cv2 before importing observer modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hex_cfg():
    """A HexGridConfiguration with default params."""
    from observer.HexObserver import HexGridConfiguration

    return HexGridConfiguration(size=30.0, rotation_deg=0.0, offset_xy=(0, 0))


@pytest.fixture(scope="module")
def hex_cc(hex_cfg):
    """
    A minimal HexCaptureConfiguration with hex set but no cameras/rsc,
    suitable for pure-math method tests.
    """
    from observer.HexObserver import HexCaptureConfiguration

    cc = object.__new__(HexCaptureConfiguration)
    cc.hex = hex_cfg
    cc.cameras = {}
    cc.rsc = None
    cc._grid_cache = {}
    return cc


# ---------------------------------------------------------------------------
# HexGridConfiguration
# ---------------------------------------------------------------------------


class TestHexGridConfiguration:
    def test_default_size(self, hex_cfg):
        assert hex_cfg.size == 30.0

    def test_anchor_computed_from_size(self):
        from observer.HexObserver import HexGridConfiguration

        cfg = HexGridConfiguration(size=60.0)
        # anchor_xy = (-int((60/2.86)+0.25), -int(60/2))
        expected_x = -int((60 / 2.86) + 0.25)
        expected_y = -int(60 / 2)
        assert cfg.anchor_xy == (expected_x, expected_y)

    def test_hex_nudges_default_empty(self, hex_cfg):
        assert hex_cfg.hex_nudges == {}

    def test_dimensions_default(self):
        from observer.HexObserver import HexGridConfiguration

        cfg = HexGridConfiguration()
        assert cfg.width == 1600
        assert cfg.height == 1600


# ---------------------------------------------------------------------------
# Static / pure math methods
# ---------------------------------------------------------------------------


class TestAxialRound:
    def test_origin(self, hex_cc):
        assert hex_cc.axial_round(0.0, 0.0) == (0, 0)

    def test_exact_integer(self, hex_cc):
        assert hex_cc.axial_round(3.0, -2.0) == (3, -2)

    def test_rounding_q_dominates(self, hex_cc):
        # dx > dy and dx > dz → rx snapped
        q, r = hex_cc.axial_round(0.6, 0.1)
        assert isinstance(q, int) and isinstance(r, int)

    def test_rounding_r_dominates(self, hex_cc):
        q, r = hex_cc.axial_round(0.1, 0.6)
        assert isinstance(q, int) and isinstance(r, int)

    def test_symmetry(self, hex_cc):
        q1, r1 = hex_cc.axial_round(1.4, -0.8)
        q2, r2 = hex_cc.axial_round(-1.4, 0.8)
        assert q1 == -q2 and r1 == -r2


class TestPixelToAxialFrac:
    def test_origin(self, hex_cc):
        q, r = hex_cc.pixel_to_axial_frac(0.0, 0.0, 30.0)
        assert q == pytest.approx(0.0)
        assert r == pytest.approx(0.0)

    def test_known_point(self, hex_cc):
        size = 30.0
        # For pointy-top: r = (2/3)*(y/size), q = x/(sqrt(3)*size) - r/2
        y = size * 3  # r = 2
        x = math.sqrt(3) * size * 2  # q = 2 - 1 = 1 when r=2
        q, r = hex_cc.pixel_to_axial_frac(x, y, size)
        assert r == pytest.approx(2.0, abs=0.01)


class TestAxialDistance:
    def test_same_cell(self, hex_cc):
        assert hex_cc.axial_distance((0, 0), (0, 0)) == 0

    def test_adjacent(self, hex_cc):
        assert hex_cc.axial_distance((0, 0), (1, 0)) == 1
        assert hex_cc.axial_distance((0, 0), (0, 1)) == 1
        assert hex_cc.axial_distance((0, 0), (-1, 1)) == 1

    def test_two_steps(self, hex_cc):
        assert hex_cc.axial_distance((0, 0), (2, 0)) == 2
        assert hex_cc.axial_distance((0, 0), (1, 1)) == 2

    def test_symmetric(self, hex_cc):
        a, b = (3, -2), (-1, 4)
        assert hex_cc.axial_distance(a, b) == hex_cc.axial_distance(b, a)

    def test_default_origin(self, hex_cc):
        assert hex_cc.axial_distance((3, -1)) == hex_cc.axial_distance((3, -1), (0, 0))


class TestMakeAffine:
    def test_identity_at_zero_rotation(self, hex_cc):
        M = hex_cc.make_affine_2x3()
        assert M.shape == (2, 3)
        # cos(0)=1, sin(0)=0
        assert M[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert M[1, 1] == pytest.approx(1.0, abs=1e-5)
        assert M[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert M[1, 0] == pytest.approx(0.0, abs=1e-5)

    def test_90_degree_rotation(self):
        from observer.HexObserver import HexGridConfiguration, HexCaptureConfiguration

        cfg = HexGridConfiguration(size=30.0, rotation_deg=90.0, offset_xy=(0, 0))
        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = cfg
        M = cc.make_affine_2x3()
        # cos(90)≈0, sin(90)≈1
        assert M[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert M[1, 0] == pytest.approx(1.0, abs=1e-5)

    def test_with_offset(self):
        from observer.HexObserver import HexGridConfiguration, HexCaptureConfiguration

        cfg = HexGridConfiguration(size=30.0, rotation_deg=0.0, offset_xy=(10, 20))
        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = cfg
        M = cc.make_affine_2x3()
        # Translation column should include offset contribution
        assert M.shape == (2, 3)


class TestApplyAffine:
    def test_single_point_no_rotation(self, hex_cc):
        pts = np.array([[100.0, 200.0]], dtype=np.float32)
        result = hex_cc.apply_affine_pts(pts)
        assert result.shape == (1, 2)

    def test_multiple_points(self, hex_cc):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
        result = hex_cc.apply_affine_pts(pts)
        assert result.shape == (3, 2)

    def test_origin_maps_consistently(self, hex_cc):
        pts = np.array([[0.0, 0.0]], dtype=np.float32)
        r1 = hex_cc.apply_affine_pts(pts)
        r2 = hex_cc.apply_affine_pts(pts)
        assert np.allclose(r1, r2)


class TestAxialToPixel:
    def test_origin(self, hex_cc):
        result = hex_cc.axial_to_pixel(0, 0)
        assert result.shape == (2,)
        # Center at q=0,r=0 should be near (0,0) in grid space
        assert float(result[0]) == pytest.approx(0.0, abs=1.0)
        assert float(result[1]) == pytest.approx(0.0, abs=1.0)

    def test_unit_q(self, hex_cc):
        # Moving one step in q should increase x
        p0 = hex_cc.axial_to_pixel(0, 0)
        p1 = hex_cc.axial_to_pixel(1, 0)
        assert float(p1[0]) > float(p0[0])

    def test_unit_r(self, hex_cc):
        p0 = hex_cc.axial_to_pixel(0, 0)
        p1 = hex_cc.axial_to_pixel(0, 1)
        assert float(p1[1]) > float(p0[1])


class TestPixelToAxial:
    def test_roundtrip_origin(self, hex_cc):
        # pixel_to_axial should round-trip axial_to_pixel
        q0, r0 = 0, 0
        px = hex_cc.axial_to_pixel(q0, r0)
        q1, r1 = hex_cc.pixel_to_axial(float(px[0]), float(px[1]), apply_affine=False)
        assert (q1, r1) == (q0, r0)

    def test_roundtrip_nonzero(self, hex_cc):
        for q, r in [(2, 1), (-1, 3), (0, -2), (3, -3)]:
            px = hex_cc.axial_to_pixel(q, r)
            qr, rr = hex_cc.pixel_to_axial(
                float(px[0]), float(px[1]), apply_affine=False
            )
            assert (qr, rr) == (q, r), f"Failed roundtrip for ({q},{r})"


class TestHexAtAxial:
    def test_returns_correct_shape(self, hex_cc):
        poly = hex_cc.hex_at_axial(0, 0)
        assert poly.shape == (6, 1, 2)

    def test_6_corners(self, hex_cc):
        poly = hex_cc.hex_at_axial(2, -1)
        assert len(poly) == 6

    def test_without_affine(self, hex_cc):
        poly = hex_cc.hex_at_axial(0, 0, apply_affine=False)
        assert poly.shape == (6, 1, 2)

    def test_center_approximately_correct(self, hex_cc):
        # The centroid of the 6 corners should be near axial_to_pixel(q, r)
        q, r = 1, 2
        poly = hex_cc.hex_at_axial(q, r, apply_affine=False)
        pts = poly.reshape(-1, 2).astype(float)
        centroid = pts.mean(axis=0)
        expected = hex_cc.axial_to_pixel(q, r)
        assert float(centroid[0]) == pytest.approx(float(expected[0]), abs=2.0)
        assert float(centroid[1]) == pytest.approx(float(expected[1]), abs=2.0)


class TestRealspaceDimensions:
    def test_default(self, hex_cc):
        w, h = hex_cc.realspaceDimensions()
        assert w == 1600
        assert h == 1600

    def test_custom(self):
        from observer.HexObserver import HexGridConfiguration, HexCaptureConfiguration

        cfg = HexGridConfiguration(size=20.0)
        cfg.width = 800
        cfg.height = 600
        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = cfg
        w, h = cc.realspaceDimensions()
        assert w == 800
        assert h == 600

    def test_no_hex(self):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = None
        w, h = cc.realspaceDimensions()
        assert w == 1600 and h == 1600


class TestGetVertexNudge:
    def test_no_nudges_returns_zero(self, hex_cc):
        hex_cc.hex.hex_nudges = {}
        dx, dy = hex_cc.get_vertex_nudge("Camera 0", 0, 0, 0)
        assert dx == 0.0 and dy == 0.0

    def test_no_cam_in_nudges(self, hex_cc):
        hex_cc.hex.hex_nudges = {"Other": {"0,0": [5, 3]}}
        dx, dy = hex_cc.get_vertex_nudge("Camera 0", 0, 0, 0)
        assert dx == 0.0 and dy == 0.0

    def test_self_hex_nudged(self, hex_cc):
        hex_cc.hex.hex_nudges = {"Camera 0": {"0,0": [10, -5]}}
        dx, dy = hex_cc.get_vertex_nudge("Camera 0", 0, 0, 0)
        # vertex 0 checks [(q, r), (q+1,r-1), (q+1,r)] = (0,0),(1,-1),(1,0)
        # Only (0,0) key exists → average of 1 = (10, -5)
        assert dx == pytest.approx(10.0)
        assert dy == pytest.approx(-5.0)

    def test_neighbor_nudge_averaged(self, hex_cc):
        # vertex 0 neighbors are (q+1,r-1) and (q+1,r) for (0,0)
        hex_cc.hex.hex_nudges = {"Camera 0": {"0,0": [10, 0], "1,-1": [20, 0]}}
        dx, dy = hex_cc.get_vertex_nudge("Camera 0", 0, 0, 0)
        assert dx == pytest.approx(15.0)  # (10+20)/2

    def test_all_six_vertices(self, hex_cc):
        hex_cc.hex.hex_nudges = {"Camera 0": {"2,3": [6, 4]}}
        for vi in range(6):
            dx, dy = hex_cc.get_vertex_nudge("Camera 0", 2, 3, vi)
            # At least the (q,r) hex itself contributes for all vertices
            assert isinstance(dx, float) and isinstance(dy, float)


# ---------------------------------------------------------------------------
# HexCaptureConfiguration — buildConfiguration / loadConfiguration
# ---------------------------------------------------------------------------


class TestHexConfigSerialization:
    def test_build_includes_hex(self):
        from observer.HexObserver import HexCaptureConfiguration, HexGridConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = HexGridConfiguration(size=25.0)
        cc.cameras = {}
        cc.rsc = None
        # Patch super().buildConfiguration()
        with mock.patch(
            "observer.CalibratedObserver.CalibratedCaptureConfiguration.buildConfiguration",
            return_value={"cameras": {}},
        ):
            cfg = cc.buildConfiguration()
        assert "hex" in cfg
        assert cfg["hex"]["size"] == 25.0

    def test_build_without_hex(self):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = None
        cc.cameras = {}
        with mock.patch(
            "observer.CalibratedObserver.CalibratedCaptureConfiguration.buildConfiguration",
            return_value={"cameras": {}},
        ):
            cfg = cc.buildConfiguration()
        assert "hex" not in cfg


# ---------------------------------------------------------------------------
# axial_distance used as static method
# ---------------------------------------------------------------------------


class TestAxialDistanceStatic:
    def test_direct_call(self):
        from observer.HexObserver import HexCaptureConfiguration

        d = HexCaptureConfiguration.axial_distance((0, 0), (3, -1))
        assert d == 3

    def test_pixel_to_axial_frac_static(self):
        from observer.HexObserver import HexCaptureConfiguration

        q, r = HexCaptureConfiguration.pixel_to_axial_frac(0, 0, 30.0)
        assert q == 0.0 and r == 0.0


# ---------------------------------------------------------------------------
# objectToHull — constituent_axials path
# ---------------------------------------------------------------------------


class TestObjectToHull:
    def test_constituent_axials_returns_array(self, hex_cc):
        obj = mock.MagicMock()
        obj.constituent_axials = [(0, 0), (1, 0)]

        # objectToHull needs realspaceDimensions and hex_at_axial
        # both work on hex_cc already; just need cv2 fillPoly to work
        result = hex_cc.objectToHull(obj)
        # Returns np.array (may be empty if cv2 mock returns nothing)
        assert isinstance(result, np.ndarray)

    def test_empty_constituent_axials_falls_back(self, hex_cc):
        obj = mock.MagicMock()
        obj.constituent_axials = []
        # Falls through to legacy path which needs rsc
        hex_cc.rsc = None
        result = hex_cc.objectToHull(obj)
        assert isinstance(result, np.ndarray)

    def test_no_constituent_axials_attr(self, hex_cc):
        obj = mock.MagicMock(spec=[])  # no constituent_axials
        hex_cc.rsc = None
        result = hex_cc.objectToHull(obj)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Extra tests to cover all other geometry, nudges and distance methods
# ---------------------------------------------------------------------------


class TestHexObserverExtra:
    def test_cam_coord_to_axial_with_nudges(self, hex_cc):
        hex_cc.hex.hex_nudges = {"Camera 0": {"1,1": [2, 3]}}
        hex_cc.rsc = mock.MagicMock()
        hex_cc.rsc.camCoordToRealSpace.return_value = (100.0, 150.0)

        hex_cc.pixel_to_axial = mock.MagicMock(return_value=(1, 1))
        hex_cc.cam_hex_at_axial = mock.MagicMock(
            return_value=np.array([[[50, 50]], [[60, 60]]], dtype=np.int32)
        )

        with mock.patch("cv2.pointPolygonTest", return_value=1.0):
            res = hex_cc.camCoordToAxial("Camera 0", (50.0, 50.0))
        assert res == (1, 1)

    def test_axial_to_cam_coord(self, hex_cc):
        hex_cc.rsc = mock.MagicMock()
        hex_cc.rsc.realSpaceToCamCoord.return_value = (40, 50)
        res = hex_cc.axialToCamCoord("Camera 0", (1, 2))
        assert res == (40, 50)

    def test_tracked_object_last_distance(self, hex_cc):
        # New object
        obj = mock.MagicMock()
        obj.isNewObject = True
        assert hex_cc.trackedObjectLastDistance(obj) == 0

        # Existing object
        obj.isNewObject = False
        prev_version = mock.MagicMock()
        obj.previousVersion.return_value = prev_version

        hex_cc.changeSetToAxialCoord = mock.MagicMock()
        hex_cc.changeSetToAxialCoord.side_effect = [(3, 3), (1, 1)]  # current, previous
        assert hex_cc.trackedObjectLastDistance(obj) == 4

    def test_axial_distance_between_objects(self, hex_cc):
        hex_cc.rsc = mock.MagicMock()
        hex_cc.rsc.changeSetToRealCenter.side_effect = [(10, 10), (20, 20)]
        hex_cc.pixel_to_axial = mock.MagicMock()
        hex_cc.pixel_to_axial.side_effect = [(0, 0), (2, 2)]

        res = hex_cc.axialDistanceBetweenObjects(mock.MagicMock(), mock.MagicMock())
        assert res == 4

    def test_define_object_from_axial(self, hex_cfg):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = hex_cfg
        cc.rsc = mock.MagicMock()

        cam = mock.MagicMock()
        cam.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cc.cameras = {"Camera 0": cam}

        conv = mock.MagicMock()
        conv.convertRealToCameraSpace.return_value = (50.0, 50.0)
        cc.rsc.closestConverterToRealCoord.return_value = conv

        obj = cc.define_object_from_axial("objA", 0, 0)
        assert obj.oid == "objA"
        assert obj.constituent_axials == [(0, 0)]

    def test_define_object_from_axials(self, hex_cfg):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = hex_cfg
        cc.rsc = mock.MagicMock()

        cam = mock.MagicMock()
        cam.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cc.cameras = {"Camera 0": cam}

        conv = mock.MagicMock()
        conv.convertRealToCameraSpace.return_value = (50.0, 50.0)
        cc.rsc.closestConverterToRealCoord.return_value = conv

        obj = None
        mock_contour = np.array(
            [[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32
        )

        import observer.HexObserver
        import observer.Observer

        with (
            mock.patch.object(
                observer.HexObserver.cv2,
                "findContours",
                return_value=([mock_contour], None),
            ),
            mock.patch.object(
                observer.HexObserver.cv2,
                "approxPolyDP",
                side_effect=lambda cnt, *args, **kwargs: cnt,
            ),
            mock.patch.object(observer.Observer.cv2, "contourArea", return_value=100.0),
        ):
            obj = cc.define_object_from_axials("objMulti", [(0, 0), (1, 0)])
        assert obj.oid == "objMulti"
        assert obj.constituent_axials == [(0, 0), (1, 0)]

    def test_draw_grid(self, hex_cfg):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = hex_cfg
        cc._grid_cache = {}
        grid = cc.draw_grid()
        assert grid.shape == (1600, 1600, 3)

    def test_draw_hex_grid_overlay(self, hex_cfg):
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = hex_cfg
        cc._grid_cache = {}
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        out = cc.draw_hex_grid_overlay(img, 10, 10, 100, 100, alpha=0.5)
        assert out.shape == (200, 200, 3)

    def test_minimap_and_hull_latency(self, hex_cfg):
        import time
        from observer.HexObserver import HexCaptureConfiguration

        cc = object.__new__(HexCaptureConfiguration)
        cc.hex = hex_cfg
        cc.cameras = {}
        cc.rsc = mock.MagicMock()
        cc._grid_cache = {}

        # Test object
        obj = mock.MagicMock()
        obj.oid = "test_obj"
        obj.constituent_axials = [(0, 0), (1, 0), (0, 1)]

        # We need mock for realspaceDimensions and hex_at_axial
        cc.realspaceDimensions = mock.MagicMock(return_value=(1600, 1600))
        cc.hex_at_axial = mock.MagicMock(
            return_value=np.array(
                [[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]], [[5, 15]], [[15, 5]]],
                dtype=np.float32,
            )
        )

        # Measure latency of first run (uncached)
        t0 = time.time()
        hull1 = cc.objectToHull(obj)
        first_hull_dur = time.time() - t0

        # Measure latency of subsequent cached runs
        t1 = time.time()
        for _ in range(50):
            hull2 = cc.objectToHull(obj)
        cached_hull_dur = (time.time() - t1) / 50.0

        # The cached run should be practically instantaneous (e.g. less than 2ms)
        assert cached_hull_dur < 0.002, (
            f"Cached hull calculation too slow: {cached_hull_dur}s"
        )
        assert np.array_equal(hull1, hull2)

        # Test buildMiniMap latency
        # Mock dependencies of buildMiniMap: buildCameraRealspaceContours, buildCameraRealspaceUnionContours, draw_hex_grid_overlay
        cc.buildCameraRealspaceContours = mock.MagicMock(return_value=[])
        cc.buildCameraRealspaceUnionContours = mock.MagicMock(return_value=[])
        cc.draw_hex_grid_overlay = mock.MagicMock(
            return_value=np.ones((1600, 1600, 3), dtype=np.uint8) * 255
        )

        from observer.HexObserver import MiniMapObject

        objects = [MiniMapObject(obj, (255, 0, 0))]

        # First call (uncached) with show_grid=True
        cc.show_grid = True
        t2 = time.time()
        map1 = cc.buildMiniMap(objects)
        first_map_dur = time.time() - t2

        # Cached calls
        t3 = time.time()
        for _ in range(50):
            map2 = cc.buildMiniMap(objects)
        cached_map_dur = (time.time() - t3) / 50.0

        # The cached buildMiniMap should be extremely fast (e.g. less than 5ms)
        assert cached_map_dur < 0.005, (
            f"Cached minimap render too slow: {cached_map_dur}s"
        )
        assert map1[0, 0, 0] == 255, (
            "Grid overlay was not drawn when show_grid was True"
        )

        # Now toggle show_grid to False - should break cache key and return dark image
        cc.show_grid = False
        map3 = cc.buildMiniMap(objects)
        assert map3[0, 0, 1] == 0, (
            "Grid overlay was still drawn when show_grid was False"
        )
