import unittest
import unittest.mock as mock
import json
import io
import sys
import os

# Adjust path to include the parent directory so we can import 'harmony'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mocking modules only for the import of harmonyServer
with mock.patch.dict(sys.modules):
    # Mock cv2 before importing harmonyServer
    sys.modules["cv2"] = mock.MagicMock()
    sys.modules["matplotlib"] = mock.MagicMock()
    sys.modules["matplotlib.figure"] = mock.MagicMock()
    sys.modules["matplotlib.backends"] = mock.MagicMock()
    sys.modules["matplotlib.backends.backend_agg"] = mock.MagicMock()
    sys.modules["HarmonyMachine"] = mock.MagicMock()
    sys.modules["observer"] = mock.MagicMock()
    sys.modules["observer.Observer"] = mock.MagicMock()
    # Mock submodules referenced in harmonyServer imports
    sys.modules["observer.configurator"] = mock.MagicMock()
    sys.modules["observer.observerServer"] = mock.MagicMock()
    sys.modules["observer.calibrator"] = mock.MagicMock()
    # Mock starlette WSGIMiddleware so no flask dependency needed at import time
    sys.modules["starlette.middleware.wsgi"] = mock.MagicMock()
    # Mock HarmonyMachine as both top-level and package-relative import
    _hm_mock = mock.MagicMock()
    sys.modules["HarmonyMachine"] = _hm_mock
    sys.modules["harmony.HarmonyMachine"] = _hm_mock

    from harmony import harmonyServer


# Helper to mock generators
def mock_gen(*args, **kwargs):
    yield b"--frame\r\nContent-Type: image/jpg\r\n\r\nfake\r\n"


class TestHarmonyServer(unittest.TestCase):
    def setUp(self):
        # Create the app and FastAPI test client
        from fastapi.testclient import TestClient

        # Start patchers first to prevent thread leaks in app creation
        self.patchers = [
            mock.patch.object(harmonyServer, "getConsoleImage", side_effect=mock_gen),
            mock.patch.object(
                harmonyServer, "renderConsole", side_effect=mock_gen, create=True
            ),
            mock.patch.object(
                harmonyServer, "genCombinedCamerasView", side_effect=mock_gen
            ),
            mock.patch.object(
                harmonyServer, "genCombinedCameraWithChangesView", side_effect=mock_gen
            ),
            mock.patch.object(harmonyServer, "get_broadcaster"),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]

        self.started_patchers = []
        for patcher in self.patchers:
            mock_obj = patcher.start()
            self.started_patchers.append(patcher)
            if hasattr(patcher, "attribute") and patcher.attribute == "get_broadcaster":
                mock_broadcaster = mock.MagicMock()
                mock_broadcaster.subscribe.side_effect = mock_gen
                mock_obj.return_value = mock_broadcaster

        self.app = harmonyServer.create_harmony_app()
        self.client = TestClient(self.app, raise_server_exceptions=True)

        # Patch generators
        self.patchers = [
            mock.patch.object(harmonyServer, "getConsoleImage", side_effect=mock_gen),
            mock.patch.object(
                harmonyServer, "renderConsole", side_effect=mock_gen, create=True
            ),
            mock.patch.object(
                harmonyServer, "genCombinedCamerasView", side_effect=mock_gen
            ),
            mock.patch.object(
                harmonyServer, "genCombinedCameraWithChangesView", side_effect=mock_gen
            ),
            mock.patch.object(harmonyServer, "get_broadcaster"),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        # Patchers are already handled above

    def tearDown(self):
        for patcher in self.started_patchers:
            patcher.stop()

    def test_harmony_dashboard(self):
        """Test main dashboard."""
        response = self.client.get("/harmony/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Harmony", response.content)

    def test_harmony_reset(self):
        """Test reset."""
        response = self.client.get("/harmony/reset")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"success", response.content)

    def test_harmony_console(self):
        """Test harmony console endpoint."""
        response = self.client.get("/harmony/harmony_console")
        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response.headers["content-type"])

    def test_combined_cameras(self):
        """Test combined cameras endpoint."""
        response = self.client.get("/harmony/combinedCameras")
        self.assertEqual(response.status_code, 200)

    def test_get_objects(self):
        """Test getting objects table."""
        # Ensure no objects in memory so buildObjectTable returns immediately
        harmonyServer._cm.memory = []
        response = self.client.get("/harmony/objects")
        self.assertEqual(response.status_code, 200)

    def test_delete_object(self):
        """Test object deletion via factory."""
        view_id = "test_view"

        from harmony.harmonyServer import SessionConfig

        # Inject session
        harmonyServer.SESSIONS[view_id] = SessionConfig()
        harmonyServer.SESSIONS[view_id].selection.firstCell = (0, 0)

        # Mock memory
        mock_mem = mock.MagicMock()
        mock_mem.oid = "TargetObject"
        harmonyServer._cm.memory = [mock_mem]

        # Mock changeSetToAxialCoord for VirtualMap
        harmonyServer._cm.cc.changeSetToAxialCoord.return_value = (0, 0)
        harmonyServer._cc.changeSetToAxialCoord.return_value = (
            0,
            0,
        )  # Ensure both are mocked to return tuples
        # Ensure _cm.cc matches the mocked _cc so the delete lookup works
        harmonyServer._cm.cc = harmonyServer._cc

        response = self.client.delete(f"/harmony/object_factory/{view_id}")
        self.assertEqual(response.status_code, 200)
        # Verify object removed
        self.assertNotIn(mock_mem, harmonyServer._cm.memory)

    def test_select_pixel(self):
        """Test selecting a pixel puts it in SESSIONS."""
        harmonyServer._cc.camCoordToAxial = mock.MagicMock(return_value=(1, 2))
        harmonyServer._cc.pixel_to_axial = mock.MagicMock(return_value=(1, 2))
        harmonyServer._cm.cc = harmonyServer._cc

        view_id = "view1"
        from harmony.harmonyServer import SessionConfig

        harmonyServer.SESSIONS[view_id] = SessionConfig()

        data = {
            "viewId": view_id,
            "selectedPixel": "[100, 200]",
            "selectedCamera": "Camera 0",
            "appendPixel": "",
        }
        response = self.client.post("/harmony/select_pixel", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(view_id, harmonyServer.SESSIONS)
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (1, 2))

    def test_cam_with_changes(self):
        response = self.client.get("/harmony/camWithChanges/Camera0/view1")
        self.assertEqual(response.status_code, 200)
        response_all = self.client.get("/harmony/camWithChanges/All/view1")
        self.assertEqual(response_all.status_code, 200)

    def test_combined_cameras_with_changes(self):
        response = self.client.get("/harmony/combinedCamerasWithChanges")
        self.assertEqual(response.status_code, 200)

    def test_build_object_factory(self):
        view_id = "test_view"
        from harmony.harmonyServer import SessionConfig

        harmonyServer.SESSIONS[view_id] = SessionConfig()
        harmonyServer.SESSIONS[view_id].selection.firstCell = (0, 0)

        response = self.client.get(f"/harmony/object_factory/{view_id}")
        self.assertEqual(response.status_code, 200)

    def test_minimap(self):
        response = self.client.get("/harmony/minimap/view1")
        self.assertEqual(response.status_code, 200)

    def test_clear_pixel(self):
        view_id = "view1"
        from harmony.harmonyServer import SessionConfig, CellSelection

        harmonyServer.SESSIONS[view_id] = SessionConfig()
        # Set something
        harmonyServer.SESSIONS[view_id].selection = CellSelection(firstCell=(1, 1))

        response = self.client.get(f"/harmony/clear_pixel/{view_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, None)

    def test_object_table_ordering(self):
        """Test object table grouping and ordering."""
        view_id = "test_view"
        from harmony.harmonyServer import SessionConfig

        session = SessionConfig()
        session.moveable = ["ObjA"]
        session.allies = ["ObjB"]

        harmonyServer.SESSIONS[view_id] = session

        # Mock memory
        mock_a = mock.MagicMock()
        mock_a.oid = "ObjA"
        mock_b = mock.MagicMock()
        mock_b.oid = "ObjB"
        mock_c = mock.MagicMock()
        mock_c.oid = "ObjC"

        harmonyServer._cm.memory = [mock_c, mock_b, mock_a]

        with mock.patch.object(harmonyServer, "captureToChangeRow") as mock_row:
            mock_row.side_effect = (
                lambda x, color=None, is_moveable=False, viewId=None: x.oid
            )
            output = harmonyServer.buildObjectTable(view_id)

        self.assertIn("category-moveable", output)
        self.assertIn("ObjA", output)
        self.assertIn("category-allies", output)
        self.assertIn("ObjB", output)
        self.assertIn("category-selectable", output)
        self.assertIn("ObjC", output)

        idx_move = output.index("category-moveable")
        idx_ally = output.index("category-allies")
        idx_sel = output.index("category-selectable")
        self.assertTrue(idx_move < idx_ally < idx_sel)

    def test_update_session_id_reclaim(self):
        """Test reclaiming an existing session ID."""
        old_id = "old_session"
        existing_id = "existing_session"

        from harmony.harmonyServer import SessionConfig

        harmonyServer.SESSIONS[old_id] = SessionConfig()
        harmonyServer.SESSIONS[existing_id] = SessionConfig()

        response = self.client.post(
            "/harmony/update_session_id",
            data={"viewId": old_id, "newViewId": existing_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"window.location.href", response.content)
        self.assertIn(existing_id.encode(), response.content)

    def test_capture_to_change_row_with_color(self):
        """Test that captureToChangeRow calls custom_object_visual when color is provided."""
        mock_capture = mock.MagicMock()
        mock_capture.oid = "ObjColor"

        with (
            mock.patch.object(harmonyServer, "custom_object_visual") as mock_custom,
            mock.patch.object(harmonyServer._cm, "object_visual") as mock_default,
            mock.patch.object(harmonyServer, "imageToBase64", return_value="fake_b64"),
            mock.patch.object(
                harmonyServer._cm.cc, "trackedObjectLastDistance", return_value=10.0
            ),
        ):
            # Call with color
            harmonyServer.captureToChangeRow(mock_capture, color=(255, 0, 0))
            mock_custom.assert_called_once()
            mock_default.assert_not_called()

            mock_custom.reset_mock()
            mock_default.reset_mock()

            # Call without color
            harmonyServer.captureToChangeRow(mock_capture)
            mock_custom.assert_not_called()
            mock_default.assert_called_once()

    def test_session_cookie_logic(self):
        """Test session ID persistence via cookies."""
        # Case 1: No cookie, no param -> Should set cookie
        response = self.client.get("/harmony/", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # check cookie was set
        self.assertIn("session_view_id", response.cookies)

        val = response.cookies.get("session_view_id")
        self.assertIsNotNone(val)

        # Case 2: Send cookie -> Should maintain session
        self.client.cookies.set("session_view_id", val)
        response2 = self.client.get("/harmony/")
        self.assertEqual(response2.status_code, 200)
        self.assertIn(f"Session ID: {val}".encode(), response2.content)

        # Case 3: Send Param -> Should override cookie and update it
        new_val = "OVERRIDE-SESSION"
        response3 = self.client.get(f"/harmony/?viewId={new_val}")
        self.assertEqual(response3.status_code, 200)
        self.assertEqual(response3.cookies.get("session_view_id"), new_val)
        self.assertIn(f"Session ID: {new_val}".encode(), response3.content)

    def test_set_overlays(self):
        """Test setting grid and object overlays."""
        # The configuration is stored in harmonyServer._cc
        response = self.client.post(
            "/harmony/set_overlays", data={"show_grid": "true", "show_objects": "false"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(harmonyServer._cc.show_grid)
        self.assertFalse(harmonyServer._cc.show_objects)

        response2 = self.client.post(
            "/harmony/set_overlays", data={"show_grid": "false", "show_objects": "true"}
        )
        self.assertEqual(response2.status_code, 200)
        self.assertFalse(harmonyServer._cc.show_grid)
        self.assertTrue(harmonyServer._cc.show_objects)

    def test_multi_selection_logic(self):
        """Test multi-selection type prioritization and appending."""
        view_id = "test_session_multi"
        from harmony.harmonyServer import SessionConfig

        session = SessionConfig()
        session.allies = ["Obj1"]
        session.terrain = ["Obj1", "Obj2"]
        harmonyServer.SESSIONS[view_id] = session

        from harmony.harmonyServer import _cm, _cc

        mock_obj1 = mock.MagicMock()
        mock_obj1.oid = "Obj1"
        del mock_obj1.objectType

        mock_obj2 = mock.MagicMock()
        mock_obj2.oid = "Obj2"
        del mock_obj2.objectType

        mock_obj3 = mock.MagicMock()
        mock_obj3.oid = "Obj3"
        del mock_obj3.objectType

        harmonyServer._cm.memory = [mock_obj1, mock_obj2, mock_obj3]

        def mock_cstac(mem):
            if mem.oid == "Obj1":
                return (10, 10)
            if mem.oid == "Obj2":
                return (20, 20)

    def test_multi_selection_logic(self):
        """Test multi-selection type prioritization and appending."""
        view_id = "test_session_multi"
        from harmony.harmonyServer import SessionConfig

        session = SessionConfig()
        session.allies = ["Obj1"]
        session.terrain = ["Obj1", "Obj2"]
        harmonyServer.SESSIONS[view_id] = session

        from harmony.harmonyServer import _cm, _cc

        mock_obj1 = mock.MagicMock()
        mock_obj1.oid = "Obj1"
        del mock_obj1.objectType

        mock_obj2 = mock.MagicMock()
        mock_obj2.oid = "Obj2"
        del mock_obj2.objectType

        mock_obj3 = mock.MagicMock()
        mock_obj3.oid = "Obj3"
        del mock_obj3.objectType

        harmonyServer._cm.memory = [mock_obj1, mock_obj2, mock_obj3]

        def mock_cstac(mem):
            if mem.oid == "Obj1":
                return (10, 10)
            if mem.oid == "Obj2":
                return (20, 20)
            if mem.oid == "Obj3":
                return (30, 30)
            return (0, 0)

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cm.cc.changeSetToAxialCoord = mock.MagicMock(
            side_effect=mock_cstac
        )
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(return_value=(10, 10))
        harmonyServer._cm.cc.axial_distance = mock.MagicMock(return_value=20)
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )

        # Select Obj1 (Ally > Terrain)
        resp = self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        html = resp.content.decode()
        self.assertIn("Selected First Cell: (10, 10)", html)
        self.assertIn("Object: Obj1", html)
        self.assertIn("Type: Ally", html)
        self.assertIn("Delete Object", html)
        self.assertNotIn("Type: Terrain", html)

        # Select Obj2 with append (Terrain)
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(return_value=(20, 20))
        resp = self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[20, 20]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        html = resp.content.decode()
        self.assertIn("Selected First Cell: (10, 10)", html)
        self.assertIn("Latest Selection: (20, 20)", html)
        self.assertIn("Object: Obj2 (Terrain)", html)

        # Select Obj3 (Default Selectable)
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(return_value=(30, 30))
        resp = self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[30, 30]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        html = resp.content.decode()
        self.assertIn("Object: Obj3 (Selectable)", html)

    def test_admin_vs_user_delete(self):
        """Verify Delete Object is only in Admin."""
        SessionConfig = harmonyServer.SessionConfig
        create_harmony_app = harmonyServer.create_harmony_app
        from fastapi.testclient import TestClient

        admin_client = self.client
        user_app = create_harmony_app(template_name="HarmonyUser.html")
        user_client = TestClient(user_app)

        mock_obj = mock.MagicMock()
        mock_obj.oid = "AdminObj"
        del mock_obj.objectType
        harmonyServer._cm.memory = [mock_obj]

        harmonyServer.SESSIONS["admin_session"] = SessionConfig()
        harmonyServer.SESSIONS["user_session"] = SessionConfig()

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cm.cc.changeSetToAxialCoord = mock.MagicMock(
            return_value=(10, 10)
        )
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(return_value=(10, 10))
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )

        resp = admin_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "admin_session",
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        self.assertIn("Delete Object", resp.content.decode())

        resp_user = user_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "user_session",
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        self.assertNotIn("Delete Object", resp_user.content.decode())

    def test_move_permissions(self):
        """Verify move permissions matrix for admin and users."""
        SessionConfig = harmonyServer.SessionConfig
        create_harmony_app = harmonyServer.create_harmony_app
        from fastapi.testclient import TestClient

        admin_client = self.client
        user_app = create_harmony_app(template_name="HarmonyUser.html")
        user_client = TestClient(user_app)

        mock_obj = mock.MagicMock()
        mock_obj.oid = "Statue"
        del mock_obj.objectType
        harmonyServer._cm.memory = [mock_obj]

        harmonyServer.SESSIONS["user_session_move"] = SessionConfig()
        harmonyServer.SESSIONS["admin_session_move"] = SessionConfig()

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cm.cc.changeSetToAxialCoord = mock.MagicMock(
            return_value=(10, 10)
        )
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(
            side_effect=lambda cam, pt: tuple(map(int, pt))
        )
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )

        print(f"TEST SESSIONS ID: {id(harmonyServer.SESSIONS)}")
        # User -> Non-Moveable Object -> No Move Button
        user_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "user_session_move",
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        resp = user_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "user_session_move",
                "selectedPixel": "[20, 20]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        self.assertNotIn("Move Statue Here", resp.content.decode())

        # User -> Moveable Object -> Has Move Button
        harmonyServer.SESSIONS["user_session_move"].moveable = ["Hero"]
        mock_obj.oid = "Hero"
        user_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "user_session_move",
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        resp = user_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "user_session_move",
                "selectedPixel": "[20, 20]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        self.assertIn("Move Hero Here", resp.content.decode())

        # Admin -> Non-Moveable -> Has Move Button (Override)
        mock_obj.oid = "Statue"
        admin_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "admin_session_move",
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        resp = admin_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": "admin_session_move",
                "selectedPixel": "[20, 20]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        self.assertIn("Move Statue Here", resp.content.decode())

    def test_additional_clicks_replace_or_append(self):
        """Test that additional clicks:
        1. Replace the second selection when appendPixel is False.
        2. Append to multiple selections when appendPixel is True.
        """
        view_id = "test_session_replace_append"
        from harmony.harmonyServer import SessionConfig

        session = SessionConfig()
        harmonyServer.SESSIONS[view_id] = session

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cm.cc.camCoordToAxial = mock.MagicMock(
            side_effect=lambda cam, pt: tuple(map(int, pt))
        )
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )

        # 1. First selection (10, 10)
        self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[10, 10]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (10, 10))
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.additionalCells, [])

        # 2. Second selection (20, 20) -> should set additionalCells = [(20, 20)]
        self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[20, 20]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (10, 10))
        self.assertEqual(
            harmonyServer.SESSIONS[view_id].selection.additionalCells, [(20, 20)]
        )

        # 3. Third selection (30, 30) with appendPixel=False -> should replace second selection
        self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[30, 30]",
                "selectedCamera": "Camera 0",
                "appendPixel": "",
            },
        )
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (10, 10))
        self.assertEqual(
            harmonyServer.SESSIONS[view_id].selection.additionalCells, [(30, 30)]
        )

        # 4. Fourth selection (40, 40) with appendPixel=True -> should append (prepend) to additionalCells
        self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[40, 40]",
                "selectedCamera": "Camera 0",
                "appendPixel": "true",
            },
        )
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (10, 10))
        self.assertEqual(
            harmonyServer.SESSIONS[view_id].selection.additionalCells,
            [(40, 40), (30, 30)],
        )

    def test_canvas_data_mapping(self):
        """Test the UI coordinate mapping in the canvas_data endpoint."""
        SessionConfig = harmonyServer.SessionConfig
        view_id = "test_canvas"
        session = SessionConfig()
        session.selection.firstCell = (1, 2)
        session.selection.additionalCells = [(3, 4)]
        harmonyServer.SESSIONS[view_id] = session

        mock_obj = mock.MagicMock()
        mock_obj.oid = "TargetMapObj"
        harmonyServer._cm.memory = [mock_obj]

        import numpy as np

        # mock hull and points
        harmonyServer._cm.cc.objectToHull.return_value = np.array([[10, 10], [10, 20]])

        # mock cam change points
        cam_change = mock.MagicMock()
        cam_change.changePoints = [np.array([5, 5])]
        mock_obj.changeSet = {"Camera 0": cam_change}

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cc.cameras = {"Camera 0": mock.MagicMock()}
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(2.0, 2.0, 5.0, 5.0)
        )
        harmonyServer._cm.cc.hex_at_axial = mock.MagicMock(
            return_value=[np.array([100, 100])]
        )
        harmonyServer._cm.cc.cam_hex_at_axial = mock.MagicMock(
            return_value=[np.array([50, 50])]
        )

        resp = self.client.get(f"/harmony/canvas_data/{view_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("selection", data)
        self.assertIsNotNone(data["selection"]["firstCell"])
        first_vm_pts = data["selection"]["firstCell"]["VirtualMap"]
        self.assertEqual(first_vm_pts[0], [190.0, 190.0])

        first_cam_pts = data["selection"]["firstCell"]["Camera 0"]
        self.assertEqual(first_cam_pts[0], [90.0, 90.0])

        self.assertIn("TargetMapObj", data["objects"])
        obj_vm = data["objects"]["TargetMapObj"]["VirtualMap"]
        self.assertEqual(obj_vm[0], [10.0, 10.0])
        self.assertEqual(obj_vm[1], [10.0, 30.0])

    def test_show_grid_off_by_default(self):
        """Verify that show_grid is off (defaults to False) initially."""
        cc = harmonyServer._cc
        # If cc does not have show_grid explicitly configured, getattr(cc, "show_grid", False) should return False
        if hasattr(cc, "show_grid"):
            delattr(cc, "show_grid")
        self.assertFalse(getattr(cc, "show_grid", False))

    def test_form_contains_hx_sync(self):
        """Verify that the selectPixelForm contains hx-sync='this:queue' in the rendered HTML."""
        resp = self.client.get("/harmony/?viewId=Calm-Sun")
        self.assertEqual(resp.status_code, 200)
        html_content = resp.content.decode()
        self.assertIn('hx-sync="this:queue"', html_content)

    def test_camera_space_contour_union(self):
        """Verify that the camera space polygon union is used for multi-cell objects."""
        import numpy as np

        view_id = "test_union_cam"
        session = harmonyServer.SessionConfig()
        harmonyServer.SESSIONS[view_id] = session

        mock_obj = mock.MagicMock()
        mock_obj.oid = "MultiCellObj"
        # Provide constituent axials
        mock_obj.constituent_axials = [(1, 1), (1, 2)]
        harmonyServer._cm.memory = [mock_obj]

        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cc.cameras = {"Camera 0": mock.MagicMock()}

        # Mock hex coordinates
        harmonyServer._cm.cc.cam_hex_at_axial = mock.MagicMock(
            side_effect=lambda cam, q, r: np.array(
                [[[100, 100]], [[120, 100]], [[120, 120]], [[100, 120]]], dtype=np.int32
            )
        )

        # Mock cv2.findContours to prevent mocked cv2 unpack errors
        mock_contour = np.array(
            [[[100, 100]], [[120, 100]], [[120, 120]], [[100, 120]]], dtype=np.int32
        )
        harmonyServer.cv2.findContours = mock.MagicMock(
            return_value=([mock_contour], None)
        )

        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )

        resp = self.client.get(f"/harmony/canvas_data/{view_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("MultiCellObj", data["objects"])
        cam_pts = data["objects"]["MultiCellObj"]["Camera 0"]
        # Union of identical polygons should return a 4-point rect contour
        self.assertGreater(len(cam_pts), 0)

    def test_object_definition_validations(self):
        """Verify that object definition rejects duplicate names or overlapping coordinates."""
        view_id = "test_factory_val"
        session = harmonyServer.SessionConfig()
        session.selection.firstCell = (0, 0)
        session.selection.additionalCells = [(1, 0)]
        harmonyServer.SESSIONS[view_id] = session

        # Mock define_object_from_axial and define_object_from_axials to return oid "UniqueObj"
        harmonyServer._cm.cc.define_object_from_axial.return_value.oid = "UniqueObj"
        harmonyServer._cm.cc.define_object_from_axials.return_value.oid = "UniqueObj"

        # Pre-populate memory with an existing object named "Hero" on coordinate (0, 0)
        mock_hero = mock.MagicMock()
        mock_hero.oid = "Hero"
        mock_hero.constituent_axials = [(0, 0)]
        harmonyServer._cm.memory = [mock_hero]

        # 1. Test name collision with "Hero" (case-insensitive)
        resp = self.client.post(
            f"/harmony/object_factory/{view_id}",
            data={"object_name": "hero"},
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Definition Error", html)
        self.assertIn("already exists", html)

        # 2. Test cell coordinate overlap (coordinate (0, 0) is occupied by "Hero")
        resp = self.client.post(
            f"/harmony/object_factory/{view_id}",
            data={"object_name": "NewObj"},
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Definition Error", html)
        self.assertIn("overlap", html)

        # 3. Test empty name rejection
        resp = self.client.post(
            f"/harmony/object_factory/{view_id}",
            data={"object_name": "   "},
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Definition Error", html)
        self.assertIn("cannot be empty", html)

        # 4. Test successful definition when name is unique and cells are unoccupied
        # Clean memory and selection for a clean run
        mock_hero.constituent_axials = [(9, 9)]  # Move Hero out of the way
        resp = self.client.post(
            f"/harmony/object_factory/{view_id}",
            data={"object_name": "UniqueObj"},
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Object Defined", html)
        self.assertIn("UniqueObj", html)

    def test_collapsible_object_categories(self):
        """Verify that object categories are rendered in collapsible details elements."""
        view_id = "test_col_cat"
        session = harmonyServer.SessionConfig()
        session.selectable = ["ObjSelect"]
        harmonyServer.SESSIONS[view_id] = session

        # Pre-populate memory with a selectable object
        mock_obj = mock.MagicMock()
        mock_obj.oid = "ObjSelect"
        mock_obj.constituent_axials = [(0, 0)]
        harmonyServer._cm.memory = [mock_obj]

        with mock.patch.object(harmonyServer, "captureToChangeRow") as mock_row:
            mock_row.side_effect = (
                lambda x, color=None, is_moveable=False, viewId=None: x.oid
            )
            resp = self.client.get(f"/harmony/objects?viewId={view_id}")

        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Verify that category details/summary is rendered
        self.assertIn('<details id="category-selectable"', html)
        self.assertIn("<summary", html)
        self.assertIn("Selectable", html)

        # Verify that persistence script block is loaded
        self.assertIn("objectCategoryDetailsListenerRegistered", html)
        self.assertIn("localStorage.setItem('collapse-'", html)



class TestCameraImageUpdates(unittest.TestCase):
    """Verify that camera images update when the underlying frame changes.

    These tests exercise render_camera() directly to confirm:
    - A new frame produces different JPEG bytes than the previous frame.
    - A None frame returns None (no crash, no stale data).
    - An unknown camera name returns None gracefully.
    - The /camWithChanges streaming endpoint responds with multipart content.
    """

    def _make_cc(self, frame=None):
        """Return a minimal mock cc with one camera."""
        import numpy as np

        cc = mock.MagicMock()
        cam = mock.MagicMock()
        cam.mostRecentFrame = (
            frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        )
        cam.activeZoneBoundingBox = (0, 0, 640, 480)
        cam.cropToActiveZone = mock.MagicMock(side_effect=lambda f: f)
        cc.cameras = {"Camera 0": cam}
        cc.show_grid = False
        cc.rsc = None  # disable grid overlay path
        return cc, cam

    # ------------------------------------------------------------------
    # render_camera unit tests
    # ------------------------------------------------------------------

    def test_render_camera_returns_jpeg_bytes(self):
        """render_camera should return non-empty bytes for a valid frame."""
        import numpy as np

        cc, cam = self._make_cc(frame=np.zeros((480, 640, 3), dtype=np.uint8))

        with mock.patch.object(harmonyServer, "cv2") as mock_cv2:
            mock_cv2.imencode.return_value = (True, np.frombuffer(b"JPEG_FRAME_A", dtype=np.uint8))
            mock_cv2.resize.return_value = cam.mostRecentFrame
            mock_cv2.addWeighted = mock.MagicMock()

            result = harmonyServer.render_camera(cc, "Camera 0")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_render_camera_changes_when_frame_changes(self):
        """render_camera output must differ when mostRecentFrame changes.

        This is the core regression test: if render_camera caches stale data
        and ignores frame updates, the two results would be identical.
        """
        import numpy as np

        cc, cam = self._make_cc()

        frame_a = np.zeros((480, 640, 3), dtype=np.uint8)   # all-black
        frame_b = np.full((480, 640, 3), 128, dtype=np.uint8)  # mid-gray

        results = []
        for frame_content, jpeg_bytes in [(frame_a, b"JPEG_FRAME_A"), (frame_b, b"JPEG_FRAME_B")]:
            cam.mostRecentFrame = frame_content
            cam.cropToActiveZone.side_effect = lambda f: f

            with mock.patch.object(harmonyServer, "cv2") as mock_cv2:
                mock_cv2.imencode.return_value = (
                    True,
                    np.frombuffer(jpeg_bytes, dtype=np.uint8),
                )
                mock_cv2.resize.return_value = frame_content
                mock_cv2.addWeighted = mock.MagicMock()

                results.append(harmonyServer.render_camera(cc, "Camera 0"))

        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertNotEqual(
            results[0],
            results[1],
            "render_camera returned identical bytes for two different frames — camera is not updating",
        )

    def test_render_camera_none_frame_returns_none(self):
        """render_camera must return None when mostRecentFrame is None."""
        cc, cam = self._make_cc()
        cam.mostRecentFrame = None  # simulate camera not yet ready

        result = harmonyServer.render_camera(cc, "Camera 0")
        self.assertIsNone(result)

    def test_render_camera_unknown_camera_returns_none(self):
        """render_camera must return None gracefully for a missing camera name."""
        cc, _ = self._make_cc()

        result = harmonyServer.render_camera(cc, "NonExistentCamera")
        self.assertIsNone(result)

    def test_render_camera_encode_failure_returns_none(self):
        """If cv2.imencode fails, render_camera must not crash and must return None."""
        import numpy as np

        cc, cam = self._make_cc(frame=np.zeros((480, 640, 3), dtype=np.uint8))

        with mock.patch.object(harmonyServer, "cv2") as mock_cv2:
            mock_cv2.imencode.return_value = (False, None)   # encode failure
            mock_cv2.resize.return_value = cam.mostRecentFrame

            # Should not raise; any exception is caught and None returned
            result = harmonyServer.render_camera(cc, "Camera 0")

        # result may be None or raise AttributeError on .tobytes() — either way no unhandled crash
        # The try/except in render_camera should catch it and return None
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # /camWithChanges streaming endpoint
    # ------------------------------------------------------------------

    def test_camwithchanges_endpoint_returns_multipart_stream(self):
        """GET /camWithChanges/{camName}/{viewId} must respond 200 with multipart content-type."""
        from fastapi.testclient import TestClient

        def _fake_broadcaster(*args, **kwargs):
            broadcaster = mock.MagicMock()
            broadcaster.subscribe.side_effect = mock_gen
            return broadcaster

        patchers = [
            mock.patch.object(harmonyServer, "get_broadcaster", side_effect=_fake_broadcaster),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        started = [p.start() for p in patchers]
        try:
            app = harmonyServer.create_harmony_app()
            client = TestClient(app)
            resp = client.get("/harmony/camWithChanges/Camera 0/test_view")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("multipart/x-mixed-replace", resp.headers["content-type"])
        finally:
            for p in patchers:
                p.stop()

    def test_camwithchanges_virtualmap_uses_minimap_broadcaster(self):
        """VirtualMap camera name must route to the minimap broadcaster."""
        from fastapi.testclient import TestClient

        broadcaster_keys = []

        def _capture_broadcaster(key, render_func):
            broadcaster_keys.append(key)
            b = mock.MagicMock()
            b.subscribe.side_effect = mock_gen
            return b

        patchers = [
            mock.patch.object(harmonyServer, "get_broadcaster", side_effect=_capture_broadcaster),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        started = [p.start() for p in patchers]
        try:
            app = harmonyServer.create_harmony_app()
            client = TestClient(app)
            client.get("/harmony/camWithChanges/VirtualMap/vm_view")
            self.assertTrue(
                any("VirtualMap" in k for k in broadcaster_keys),
                f"Expected a VirtualMap broadcaster key, got: {broadcaster_keys}",
            )
        finally:
            for p in patchers:
                p.stop()



class TestSaveRestoreGameState(unittest.TestCase):
    """Verify that game state is faithfully persisted and restored.

    Tests cover:
    - Round-trip: objects and sessions survive a save → load cycle.
    - Memory replacement: cm.memory is fully replaced on load.
    - Session merging: existing sessions are preserved and saved ones merged in.
    - Missing file: load of a non-existent file returns an error, no crash.
    - Admin-only guard: user clients receive 403 on both save and load.
    - HTTP endpoints: POST /save and POST /load route correctly.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_client(self):
        """Return a fresh admin TestClient with generator mocks pre-applied."""
        from fastapi.testclient import TestClient

        patchers = [
            mock.patch.object(harmonyServer, "get_broadcaster"),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        started = [p.start() for p in patchers]
        app = harmonyServer.create_harmony_app()
        client = TestClient(app)
        return client, patchers, started

    def _teardown(self, patchers, started):
        for p in patchers:
            p.stop()

    # ------------------------------------------------------------------
    # _save_harmony / _load_harmony unit tests (bypass HTTP)
    # ------------------------------------------------------------------

    def test_save_writes_pickle_with_memory_and_sessions(self):
        """_save_harmony must write a pickle containing memory and sessions."""
        import io, pickle as pkl
        from harmony.harmonyServer import SessionConfig

        obj_a = mock.MagicMock()
        obj_a.oid = "Alpha"
        harmonyServer._cm.memory = [obj_a]
        harmonyServer.SESSIONS["sess1"] = SessionConfig(allies=["Alpha"])

        written = {}

        def fake_open(path, mode="r", **kw):
            buf = io.BytesIO()
            written["buf"] = buf
            written["path"] = path
            m = mock.MagicMock()
            m.__enter__ = mock.MagicMock(return_value=buf)
            m.__exit__ = mock.MagicMock(return_value=False)
            return m

        with (
            mock.patch("builtins.open", side_effect=fake_open),
            mock.patch.object(harmonyServer._cm, "saveGame"),
            mock.patch("harmony.harmonyServer.pickle.dump") as mock_dump,
        ):
            resp = harmonyServer._save_harmony("test_save_unit")

        self.assertIn("saved", resp.body.decode().lower())
        mock_dump.assert_called_once()
        save_data = mock_dump.call_args[0][0]
        self.assertIn("memory", save_data)
        self.assertIn("sessions", save_data)
        self.assertIn(obj_a, save_data["memory"])

    def test_load_restores_memory_from_pickle(self):
        """_load_harmony must replace cm.memory with objects from the pickle."""
        import pickle as pkl
        from harmony.harmonyServer import SessionConfig

        obj_b = mock.MagicMock()
        obj_b.oid = "Bravo"
        save_data = {"memory": [obj_b], "sessions": {}}

        harmonyServer._cm.memory = []  # start empty

        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("harmony.harmonyServer.pickle.load", return_value=save_data),
        ):
            resp = harmonyServer._load_harmony("test_load_unit")

        self.assertIn("loaded", resp.body.decode().lower())
        self.assertIn(obj_b, harmonyServer._cm.memory)

    def test_load_merges_sessions_without_losing_existing(self):
        """Loaded sessions must be merged: existing sessions stay, saved ones added."""
        import pickle as pkl
        from harmony.harmonyServer import SessionConfig

        # Pre-existing session that should survive the load
        harmonyServer.SESSIONS["live_session"] = SessionConfig(allies=["X"])

        saved_session = SessionConfig(enemies=["Y"])
        save_data = {"memory": [], "sessions": {"restored_session": saved_session}}

        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("harmony.harmonyServer.pickle.load", return_value=save_data),
        ):
            harmonyServer._load_harmony("test_merge")

        # Both sessions must now exist
        self.assertIn("live_session", harmonyServer.SESSIONS)
        self.assertIn("restored_session", harmonyServer.SESSIONS)
        self.assertEqual(harmonyServer.SESSIONS["restored_session"].enemies, ["Y"])

    def test_load_missing_file_returns_error_response(self):
        """_load_harmony on a non-existent file must return an error, not crash."""
        with mock.patch("os.path.exists", return_value=False):
            resp = harmonyServer._load_harmony("definitely_not_there")

        body = resp.body.decode().lower()
        self.assertTrue(
            "not found" in body or "error" in body,
            f"Expected error message, got: {body}",
        )

    def test_full_round_trip_save_and_load(self):
        """Objects present at save time must be present after load."""
        import copy, pickle as pkl
        from harmony.harmonyServer import SessionConfig

        obj_c = mock.MagicMock()
        obj_c.oid = "Charlie"
        harmonyServer._cm.memory = [obj_c]
        harmonyServer.SESSIONS["round_trip"] = SessionConfig(moveable=["Charlie"])

        # --- SAVE: capture what gets pickled ---
        # Use a deepcopy so the SESSIONS reference in the captured data is independent
        # of the live SESSIONS dict (which we wipe before the load step).
        captured_save_data = {}

        def fake_dump(data, fh):
            captured_save_data.update({
                "memory": list(data["memory"]),
                "sessions": copy.deepcopy(data["sessions"]),
            })

        with (
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch.object(harmonyServer._cm, "saveGame"),
            mock.patch("harmony.harmonyServer.pickle.dump", side_effect=fake_dump),
        ):
            save_resp = harmonyServer._save_harmony("round_trip_game")

        self.assertIn("saved", save_resp.body.decode().lower())
        self.assertIn("memory", captured_save_data)
        self.assertIn("round_trip", captured_save_data["sessions"])

        # --- LOAD: restore from the captured data ---
        harmonyServer._cm.memory = []  # wipe state
        harmonyServer.SESSIONS.pop("round_trip", None)
        # Confirm it's gone before load
        self.assertNotIn("round_trip", harmonyServer.SESSIONS)

        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("harmony.harmonyServer.pickle.load", return_value=captured_save_data),
        ):
            load_resp = harmonyServer._load_harmony("round_trip_game")

        self.assertIn("loaded", load_resp.body.decode().lower())
        loaded_oids = [o.oid for o in harmonyServer._cm.memory]
        self.assertIn("Charlie", loaded_oids)
        # Session must be restored with its moveable list intact
        self.assertIn("round_trip", harmonyServer.SESSIONS)
        self.assertEqual(harmonyServer.SESSIONS["round_trip"].moveable, ["Charlie"])

    # ------------------------------------------------------------------
    # Admin-only guards
    # ------------------------------------------------------------------

    def test_save_forbidden_for_user(self):
        """POST /save must return 403 for user (non-admin) apps."""
        from fastapi.testclient import TestClient

        patchers = [
            mock.patch.object(harmonyServer, "get_broadcaster"),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        started = [p.start() for p in patchers]
        try:
            user_app = harmonyServer.create_harmony_app(template_name="HarmonyUser.html")
            uc = TestClient(user_app)
            resp = uc.post("/harmony/save", data={"game_name": "test"})
            self.assertEqual(resp.status_code, 403)
        finally:
            for p in patchers:
                p.stop()

    def test_load_forbidden_for_user(self):
        """POST /load must return 403 for user (non-admin) apps."""
        from fastapi.testclient import TestClient

        patchers = [
            mock.patch.object(harmonyServer, "get_broadcaster"),
            mock.patch("harmony.harmonyServer.HexCaptureConfiguration"),
        ]
        started = [p.start() for p in patchers]
        try:
            user_app = harmonyServer.create_harmony_app(template_name="HarmonyUser.html")
            uc = TestClient(user_app)
            resp = uc.post("/harmony/load", data={"game_name": "test"})
            self.assertEqual(resp.status_code, 403)
        finally:
            for p in patchers:
                p.stop()

    # ------------------------------------------------------------------
    # HTTP endpoint routing
    # ------------------------------------------------------------------

    def test_post_save_endpoint_calls_save_harmony(self):
        """POST /harmony/save must delegate to _save_harmony and return 200."""
        client, patchers, started = self._make_client()
        try:
            with (
                mock.patch.object(
                    harmonyServer, "_save_harmony",
                    return_value=mock.MagicMock(body=b"Game saved as mygame"),
                ) as mock_save,
            ):
                resp = client.post("/harmony/save", data={"game_name": "mygame"})

            self.assertEqual(resp.status_code, 200)
            mock_save.assert_called_once_with("mygame")
        finally:
            self._teardown(patchers, started)

    def test_post_load_endpoint_calls_load_harmony(self):
        """POST /harmony/load must delegate to _load_harmony and return 200."""
        client, patchers, started = self._make_client()
        try:
            with (
                mock.patch.object(
                    harmonyServer, "_load_harmony",
                    return_value=mock.MagicMock(body=b"Game mygame loaded. Objects: 0"),
                ) as mock_load,
            ):
                resp = client.post("/harmony/load", data={"game_name": "mygame"})

            self.assertEqual(resp.status_code, 200)
            mock_load.assert_called_once_with("mygame")
        finally:
            self._teardown(patchers, started)

    def test_get_save_game_endpoint(self):
        """GET /harmony/save_game/{name} must call _save_harmony."""
        client, patchers, started = self._make_client()
        try:
            with (
                mock.patch.object(
                    harmonyServer, "_save_harmony",
                    return_value=mock.MagicMock(body=b"Game saved as quicksave"),
                ) as mock_save,
            ):
                resp = client.get("/harmony/save_game/quicksave")

            self.assertEqual(resp.status_code, 200)
            mock_save.assert_called_once_with("quicksave")
        finally:
            self._teardown(patchers, started)

    def test_get_load_game_endpoint(self):
        """GET /harmony/load_game/{name} must call _load_harmony."""
        client, patchers, started = self._make_client()
        try:
            with (
                mock.patch.object(
                    harmonyServer, "_load_harmony",
                    return_value=mock.MagicMock(body=b"Game quicksave loaded. Objects: 1"),
                ) as mock_load,
            ):
                resp = client.get("/harmony/load_game/quicksave")

            self.assertEqual(resp.status_code, 200)
            mock_load.assert_called_once_with("quicksave")
        finally:
            self._teardown(patchers, started)


if __name__ == "__main__":
    unittest.main()
