"""
Unit tests for observer/CalibratedObserver.py focusing on coordinate converters,
geometry calculations, contour scaling, point ordering, and helper functions.
"""
import pytest
import numpy as np
import cv2
from unittest import mock

# Ensure import of observer modules
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from observer.CalibratedObserver import (
    distanceFormula,
    scale_contour,
    order_points_clockwise,
    CameraRealSpaceConverter,
    RealSpaceConverter,
    MiniMapObject,
)

# ---------------------------------------------------------------------------
# Test Pure Mathematical and Helper Functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_distance_formula_2d(self):
        pt0 = (0, 0)
        pt1 = (3, 4)
        assert distanceFormula(pt0, pt1) == pytest.approx(5.0)

    def test_distance_formula_3d(self):
        pt0 = (0, 0, 0)
        pt1 = (1, 2, 2)
        assert distanceFormula(pt0, pt1) == pytest.approx(3.0)

    def test_distance_formula_mismatched_dimensions(self):
        pt0 = (0, 0)
        pt1 = (1, 2, 3)
        with pytest.raises(Exception) as excinfo:
            distanceFormula(pt0, pt1)
        assert "Cannot compute distance for dimensions" in str(excinfo.value)

    def test_scale_contour(self):
        # A simple square contour: [[10, 10], [20, 10], [20, 20], [10, 20]]
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
        scaled = scale_contour(contour, 2.0)
        assert scaled.shape == contour.shape
        # Centroid is (15, 15). Center-offset is scaled by 2.
        # Original [10, 10] relative to (15, 15) is (-5, -5). Scaled by 2 is (-10, -10) -> (5, 5).
        # Original [20, 20] relative to (15, 15) is (5, 5). Scaled by 2 is (10, 10) -> (25, 25).
        assert scaled[0][0].tolist() == [5, 5]
        assert scaled[2][0].tolist() == [25, 25]

    def test_scale_contour_zero_moment(self):
        # A degenerate single point contour (moments m00 will be 0)
        contour = np.array([[[10, 10]]], dtype=np.int32)
        scaled = scale_contour(contour, 1.5)
        # With m00 = 0,cx=0, cy=0.
        # Centered: [10, 10] - [0, 0] = [10, 10]
        # Scaled: [15, 15]
        assert scaled[0][0].tolist() == [15, 15]

    def test_order_points_clockwise(self):
        # 4 corners of a square but shuffled
        cam_pts = [(0, 10), (10, 10), (10, 0), (0, 0)]
        real_pts = [(0, 100), (100, 100), (100, 0), (0, 0)]
        ordered_cam, ordered_real = order_points_clockwise(cam_pts, real_pts)
        
        # Verify ordering is clockwise/counterclockwise by checking angles
        # The ordering should be consistent between both arrays
        assert len(ordered_cam) == 4
        assert len(ordered_real) == 4
        # Centroid is (5, 5).
        # Angles of original points relative to centroid:
        # (0, 10) -> atan2(5, -5) = 135 deg
        # (10, 10) -> atan2(5, 5) = 45 deg
        # (10, 0) -> atan2(-5, 5) = -45 deg
        # (0, 0) -> atan2(-5, -5) = -135 deg
        # Argsort angles: -135, -45, 45, 135 -> (0,0), (10,0), (10,10), (0,10)
        assert ordered_cam[0].tolist() == [0.0, 0.0]
        assert ordered_cam[1].tolist() == [10.0, 0.0]
        assert ordered_cam[2].tolist() == [10.0, 10.0]
        assert ordered_cam[3].tolist() == [0.0, 10.0]

        assert ordered_real[0].tolist() == [0.0, 0.0]
        assert ordered_real[1].tolist() == [100.0, 0.0]


# ---------------------------------------------------------------------------
# Test CameraRealSpaceConverter
# ---------------------------------------------------------------------------

class TestCameraRealSpaceConverter:
    def test_get_angle(self):
        pt0 = (0, 0)
        pt1 = (10, 0)
        pt2 = (0, 10)
        angle = CameraRealSpaceConverter.getAngle(pt0, pt1, pt2)
        assert angle == pytest.approx(90.0)

    def test_triangle_to_square(self):
        # A right-angled triangle in camera space:
        # A (0,0) - 90 deg, B (10,0) - 60 deg, D (0, 5.7735) - 30 deg (approx)
        tri_pts = [(0, 0), (10, 0), (0, 17.3205)] # 30-60-90 triangle
        square = CameraRealSpaceConverter.triangleToSquare(tri_pts)
        assert len(square) == 4
        # Pt A (90 deg corner) should be (0,0)
        assert square[0].tolist() == [0.0, 0.0]

    def test_init_with_3_points(self):
        # 30-60-90 triangles
        cam_tri = [(0, 0), (10, 0), (0, 17.32)]
        real_tri = [(0, 0), (100, 0), (0, 173.2)]
        converter = CameraRealSpaceConverter(
            camName="Cam1",
            camTriPts=cam_tri,
            realTriPts=real_tri
        )
        assert converter.camRect.shape == (4, 2)
        assert converter.realRect.shape == (4, 2)
        assert converter.M is not None
        assert converter.M.shape == (3, 3)

    def test_init_with_4_points(self):
        cam_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        real_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        converter = CameraRealSpaceConverter(
            camName="Cam2",
            camTriPts=cam_pts,
            realTriPts=real_pts
        )
        assert len(converter.camRect) == 4
        assert converter.M is not None

    def test_init_with_5_points(self):
        # 5 points to trigger cv2.findHomography
        cam_pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
        real_pts = [(0, 0), (100, 0), (100, 100), (0, 100), (50, 50)]
        converter = CameraRealSpaceConverter(
            camName="Cam3",
            camTriPts=cam_pts,
            realTriPts=real_pts
        )
        assert converter.M is not None

    def test_centroids(self):
        cam_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        real_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        converter = CameraRealSpaceConverter("CamCentroid", cam_pts, real_pts)
        assert converter.camSpaceCentroid.tolist() == [5, 5]
        assert converter.realSpaceCentroid.tolist() == [50, 50]

    def test_coordinate_conversion_roundtrip(self):
        cam_pts = [(0, 0), (20, 0), (20, 20), (0, 20)]
        real_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        converter = CameraRealSpaceConverter("CamRoundTrip", cam_pts, real_pts)
        
        # Convert camera (10, 10) -> should be real (50, 50)
        real_coord = converter.convertCameraToRealSpace((10, 10))
        assert real_coord[0] == pytest.approx(50.0)
        assert real_coord[1] == pytest.approx(50.0)

        # Convert back
        cam_coord = converter.convertRealToCameraSpace(real_coord)
        assert cam_coord[0] == pytest.approx(10.0)
        assert cam_coord[1] == pytest.approx(10.0)

    def test_show_unwarped_image(self):
        cam_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        real_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        converter = CameraRealSpaceConverter("CamUnwarp", cam_pts, real_pts)
        cam = mock.MagicMock()
        cam.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam.cropToActiveZone.return_value = cam.mostRecentFrame
        unwarped = converter.showUnwarpedImage(cam)
        assert unwarped.shape == (1200, 1200, 3)


# ---------------------------------------------------------------------------
# Test RealSpaceConverter
# ---------------------------------------------------------------------------

class TestRealSpaceConverter:
    @pytest.fixture
    def rsc_pairs(self):
        # Format: [ [camName, [camTriPts, realTriPts]], ... ]
        cam1_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        real1_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        cam2_pts = [(0, 0), (5, 0), (5, 5), (0, 5)]
        real2_pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
        
        return [
            ("Cam1", [cam1_pts, real1_pts]),
            ("Cam2", [cam2_pts, real2_pts])
        ]

    def test_init_rsc(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        assert "Cam1" in rsc.converters
        assert "Cam2" in rsc.converters
        assert len(rsc.converters["Cam1"]) == 1

    def test_closest_converters(self, rsc_pairs):
        # Let's add multiple converters to Cam1
        cam1_pts_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        real1_pts_a = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        cam1_pts_b = [(50, 50), (60, 50), (60, 60), (50, 60)]
        real1_pts_b = [(500, 500), (600, 500), (600, 600), (500, 600)]
        
        pairs = [
            ("Cam1", [cam1_pts_a, real1_pts_a]),
            ("Cam1", [cam1_pts_b, real1_pts_b])
        ]
        rsc = RealSpaceConverter(pairs)
        
        # A point close to (5,5) should select the first converter
        conv_near_a = rsc.closestConverterToCamCoord("Cam1", (2, 2))
        assert conv_near_a.camSpaceCentroid.tolist() == [5, 5]

        # A point close to (55, 55) should select the second converter
        conv_near_b = rsc.closestConverterToCamCoord("Cam1", (58, 58))
        assert conv_near_b.camSpaceCentroid.tolist() == [55, 55]

        # Closest converter to real coordinate
        conv_real_a = rsc.closestConverterToRealCoord("Cam1", (80, 80))
        assert conv_real_a.realSpaceCentroid.tolist() == [50, 50]

        conv_real_b = rsc.closestConverterToRealCoord("Cam1", (550, 550))
        assert conv_real_b.realSpaceCentroid.tolist() == [550, 550]

    def test_cam_and_real_conversions(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        
        real_pt = rsc.camCoordToRealSpace("Cam1", (5, 5))
        assert real_pt[0] == pytest.approx(50.0)
        assert real_pt[1] == pytest.approx(50.0)

        cam_pt = rsc.realSpaceToCamCoord((50.0, 50.0), "Cam1")
        assert cam_pt[0] == pytest.approx(5.0)
        assert cam_pt[1] == pytest.approx(5.0)

    def test_change_set_calculations(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        
        # Mock ChangeSet and CameraChange objects
        change_cam1 = mock.MagicMock()
        change_cam1.changeType = "add"
        change_cam1.center = (5, 5)

        change_cam2 = mock.MagicMock()
        change_cam2.changeType = "add"
        change_cam2.center = (2.5, 2.5) # Centroid of Cam2 is (2.5, 2.5) in Cam, mapping to (150, 150) in Real

        cs = mock.MagicMock()
        cs.changeSet = {"Cam1": change_cam1, "Cam2": change_cam2}

        # Center points in Real Space should be:
        # Cam1 -> (50, 50)
        # Cam2 -> (150, 150)
        centers = rsc.changeSetCenterPoints(cs)
        assert centers["Cam1"][0] == pytest.approx(50.0)
        assert centers["Cam2"][0] == pytest.approx(150.0)

        # changeSetToRealCenter (weighted average of centers)
        real_ctr_x, real_ctr_y = rsc.changeSetToRealCenter(cs)
        assert 50.0 <= real_ctr_x <= 150.0

        # changeSetCenterDeltas
        deltas = rsc.changeSetCenterDeltas(cs)
        assert "Cam1" in deltas
        assert "Cam2" in deltas

        # changeSetWithinSameRealSpace
        # Since deltas will be (150-50)/2 = 50, which is > default tolerance=30, this should be False
        assert not rsc.changeSetWithinSameRealSpace(cs, tolerance=30)
        # If tolerance is 200, it should be True
        assert rsc.changeSetWithinSameRealSpace(cs, tolerance=200)

    def test_tracked_object_calculations(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        
        change_cam1 = mock.MagicMock()
        change_cam1.changeType = "add"
        change_cam1.center = (5, 5)

        obj = mock.MagicMock()
        obj.changeSet = {"Cam1": change_cam1}

        prev_change_cam1 = mock.MagicMock()
        prev_change_cam1.changeType = "add"
        prev_change_cam1.center = (10, 10) # (100, 100) in Real Space

        prev_version = mock.MagicMock()
        prev_version.empty = False
        prev_version.changeSet = {"Cam1": prev_change_cam1}
        obj.previousVersion.return_value = prev_version

        # current center: (50, 50), previous center: (100, 100) -> distance should be sqrt(50^2 + 50^2) = 70.71
        dist = rsc.trackedObjectLastDistance(obj)
        assert dist == pytest.approx(70.7106, abs=0.01)

        # distance between two objects
        obj2 = mock.MagicMock()
        change2 = mock.MagicMock()
        change2.changeType = "add"
        change2.center = (8, 8) # (80, 80) in Real space
        obj2.changeSet = {"Cam1": change2}

        assert rsc.distanceBetweenObjects(obj, obj2) == pytest.approx(30.0 * (2**0.5), abs=0.01)

    def test_tracked_object_last_distance_empty_prev(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        obj = mock.MagicMock()
        prev_version = mock.MagicMock()
        prev_version.empty = True
        obj.previousVersion.return_value = prev_version
        assert rsc.trackedObjectLastDistance(obj) == 0

    def test_camera_real_space_overlap(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        cam1 = mock.MagicMock()
        cam1.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam1.cropToActiveZone.return_value = cam1.mostRecentFrame
        cam1.activeZone = [[0, 0], [10, 0], [10, 10], [0, 10]]
        
        cam2 = mock.MagicMock()
        cam2.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam2.cropToActiveZone.return_value = cam2.mostRecentFrame
        cam2.activeZone = [[0, 0], [5, 0], [5, 5], [0, 5]]
        
        cameras = {"Cam1": cam1, "Cam2": cam2}
        
        overlap_contours = rsc.cameraRealSpaceOverlap(cameras)
        assert isinstance(overlap_contours, (list, tuple))

        overlap_contours_2 = rsc.cameraRealSpaceOverlap(cameras, out_size=(100, 100))
        assert isinstance(overlap_contours_2, (list, tuple))

    def test_unwarped_overlaid_cameras(self, rsc_pairs):
        rsc = RealSpaceConverter(rsc_pairs)
        cam1 = mock.MagicMock()
        cam1.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam1.cropToActiveZone.return_value = cam1.mostRecentFrame
        
        cam2 = mock.MagicMock()
        cam2.mostRecentFrame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam2.cropToActiveZone.return_value = cam2.mostRecentFrame
        
        cameras = {"Cam1": cam1, "Cam2": cam2}
        overlaid = rsc.unwarpedOverlaidCameras(cameras)
        assert overlaid.shape == (1200, 1200, 3)


# ---------------------------------------------------------------------------
# Test CalibratedObserver
# ---------------------------------------------------------------------------

class TestCalibratedObserverClass:
    def test_circle_to_contour(self):
        from observer.CalibratedObserver import CalibratedObserver
        center = (50, 50)
        radius = 10
        contour = CalibratedObserver.circle_to_contour(center, radius, num_points=20)
        assert contour.shape == (20, 1, 2)
        # Check that all points lie approx on the circle of radius 10
        for pt in contour:
            x, y = pt[0]
            dist = ((x - 50)**2 + (y - 50)**2)**0.5
            assert dist == pytest.approx(10.0, abs=1.5)

    def test_distinct_colors(self):
        from observer.CalibratedObserver import CalibratedObserver
        # Mock cc to pass init
        mock_cc = mock.MagicMock()
        obs = CalibratedObserver(mock_cc)
        
        colors = obs.distinct_colors(n=1, new=True)
        assert len(colors) == 1
        for c in colors:
            assert len(c) == 3
            assert all(50 <= val <= 230 for val in c)

        # Check caching works
        colors_cached = obs.distinct_colors(n=1)
        assert colors_cached == colors

    def test_calibrate_to_object_and_build_rsc(self):
        from observer.CalibratedObserver import CalibrationObserver
        mock_cc = mock.MagicMock()
        mock_cc.cameras = {"Cam1": mock.MagicMock()}
        
        obs = CalibrationObserver(mock_cc)
        
        obs.next_triangle = [[10, 10], [20, 20], [30, 30]]
        obs.first_triangle = [[1, 1], [1, 1], [1, 1]]
        
        calib_obj = mock.MagicMock()
        change = mock.MagicMock()
        change.changeContours = [np.array([[[10, 10]], [[20, 10]], [[10, 20]]], dtype=np.int32)]
        calib_obj.changeSet = {"Cam1": change}
        
        mock_poly = np.array([[[10, 10]], [[20, 10]], [[10, 20]]], dtype=np.int32)
        with mock.patch('cv2.approxPolyDP', return_value=mock_poly), \
             mock.patch('cv2.arcLength', return_value=10.0):
            obs.calibrateToObject(calib_obj, (0, 0))
            
        assert len(obs.calibrationPts) == 1
        assert "Cam1" in obs.calibrationPts[0]
        
        obs.buildRealSpaceConverter()
        assert mock_cc.rsc is not None


class TestCalibratedCaptureConfigurationClass:
    def test_line_of_sight(self):
        from observer.CalibratedObserver import CalibratedCaptureConfiguration
        cc = object.__new__(CalibratedCaptureConfiguration)
        
        hull_a = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
        hull_b = np.array([[[50, 50]], [[60, 50]], [[60, 60]], [[50, 60]]], dtype=np.int32)
        
        def mock_object_to_hull(obj, *args, **kwargs):
            if obj.oid == "objA":
                return hull_a
            elif obj.oid == "objB":
                return hull_b
            else:
                return np.array([[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32)
                
        cc.objectToHull = mock_object_to_hull
        
        obj_a = mock.MagicMock()
        obj_a.oid = "objA"
        obj_b = mock.MagicMock()
        obj_b.oid = "objB"
        obj_c = mock.MagicMock()
        obj_c.oid = "objC"
        
        assert cc.line_of_sight(obj_a, obj_b, []) is True
        assert cc.line_of_sight(obj_a, obj_b, [obj_c]) is False


class TestFileLockClass:
    @pytest.fixture(autouse=True)
    def mock_file_lock(self):
        # Override the global autouse fixture so we test the real FileLock class
        pass

    def test_file_lock_acquire_success(self):
        from observer.file_lock import FileLock
        with mock.patch('fcntl.lockf') as mock_lockf, \
             mock.patch('builtins.open', mock.mock_open()):
            lock = FileLock()
            lock.acquire()
            mock_lockf.assert_called_once()

    def test_file_lock_acquire_already_running(self):
        from observer.file_lock import FileLock
        with mock.patch('fcntl.lockf', side_effect=IOError), \
             mock.patch('builtins.open', mock.mock_open()), \
             mock.patch('sys.exit') as mock_exit:
            lock = FileLock()
            lock.acquire()
            mock_exit.assert_called_once_with(1)

    def test_file_lock_release(self):
        from observer.file_lock import FileLock
        with mock.patch('fcntl.lockf') as mock_lockf, \
             mock.patch('builtins.open', mock.mock_open()):
            lock = FileLock()
            lock.release()
            mock_lockf.assert_called_once()
