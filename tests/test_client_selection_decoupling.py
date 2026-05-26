import unittest
import unittest.mock as mock
import json
import sys
import os
import time

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


class TestClientSelectionDecoupling(unittest.TestCase):
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

    def tearDown(self):
        for patcher in self.started_patchers:
            patcher.stop()

    def _setup_select_pixel(self, view_id="view1"):
        """Common setup for select_pixel tests: mock coordinate conversion and session."""
        harmonyServer._cc.camCoordToAxial = mock.MagicMock(return_value=(1, 2))
        harmonyServer._cc.pixel_to_axial = mock.MagicMock(return_value=(1, 2))
        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cm.cc.axial_distance = mock.MagicMock(return_value=0)
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
        )
        harmonyServer._cm.memory = []

        from harmony.harmonyServer import SessionConfig

        harmonyServer.SESSIONS[view_id] = SessionConfig()
        return view_id

    def _post_select_pixel(self, view_id, pixel="[100, 200]", append=""):
        """Helper to POST to select_pixel with sensible defaults."""
        return self.client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": pixel,
                "selectedCamera": "Camera 0",
                "appendPixel": append,
            },
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_select_pixel_response_omits_canvas_update_script(self):
        """select_pixel calls render_interactor with skip_canvas_update=True,
        so the response must NOT contain harmonyEditor.updateData."""
        view_id = self._setup_select_pixel()

        response = self._post_select_pixel(view_id)
        self.assertEqual(response.status_code, 200)

        html = response.content.decode()
        # Must NOT contain the canvas update script
        self.assertNotIn(
            "harmonyEditor.updateData",
            html,
            "select_pixel response should not include canvas update script "
            "(client owns selection state)",
        )
        # Must still contain HTML (the interactor panel)
        self.assertIn("<", html, "Response should contain HTML content")

    def test_select_pixel_server_state_still_tracks_selection(self):
        """Server-side SESSIONS should still record the selection even though
        the canvas update script is omitted."""
        view_id = self._setup_select_pixel()

        response = self._post_select_pixel(view_id)
        self.assertEqual(response.status_code, 200)

        # Server tracks the selection for move, delete, etc.
        self.assertIn(view_id, harmonyServer.SESSIONS)
        self.assertEqual(
            harmonyServer.SESSIONS[view_id].selection.firstCell, (1, 2)
        )

    def test_clear_pixel_sends_client_clear_script(self):
        """clear_pixel should return a script that explicitly nulls the client
        selection, NOT a harmonyEditor.updateData call."""
        view_id = "view_clear"
        from harmony.harmonyServer import SessionConfig, CellSelection

        harmonyServer.SESSIONS[view_id] = SessionConfig()
        harmonyServer.SESSIONS[view_id].selection = CellSelection(firstCell=(1, 1))

        response = self.client.get(f"/harmony/clear_pixel/{view_id}")
        self.assertEqual(response.status_code, 200)

        html = response.content.decode()

        # Should contain the explicit client-side selection clear
        self.assertIn(
            "firstCell: null",
            html,
            "clear_pixel should include explicit client-side firstCell: null",
        )
        self.assertIn(
            "additionalCells: []",
            html,
            "clear_pixel should include explicit client-side additionalCells: []",
        )

        # Should NOT contain updateData (since updateData no longer overwrites selection)
        self.assertNotIn(
            "harmonyEditor.updateData",
            html,
            "clear_pixel should not use updateData to clear selection",
        )

        # Server-side selection should also be cleared
        self.assertIsNone(harmonyServer.SESSIONS[view_id].selection.firstCell)

    def test_canvas_data_includes_selection(self):
        """canvas_data endpoint still returns selection in JSON
        (server tracks it; client is responsible for ignoring it)."""
        import numpy as np

        view_id = "view_canvas_sel"
        from harmony.harmonyServer import SessionConfig

        session = SessionConfig()
        session.selection.firstCell = (1, 2)
        harmonyServer.SESSIONS[view_id] = session

        # Mock the objects and cameras needed by _get_canvas_data_dict
        harmonyServer._cm.memory = []
        harmonyServer._cm.cc = harmonyServer._cc
        harmonyServer._cc.cameras = {"Camera 0": mock.MagicMock()}
        harmonyServer.get_conversion_params = mock.MagicMock(
            return_value=(1.0, 1.0, 0, 0)
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
        self.assertIsNotNone(
            data["selection"]["firstCell"],
            "canvas_data should still include selection.firstCell "
            "(server provides it; client ignores it)",
        )

    def test_select_pixel_response_time(self):
        """select_pixel should respond without hanging (generous 2 s bound)."""
        view_id = self._setup_select_pixel()

        start = time.monotonic()
        response = self._post_select_pixel(view_id)
        elapsed = time.monotonic() - start

        self.assertEqual(response.status_code, 200)
        self.assertLess(
            elapsed,
            2.0,
            f"select_pixel took {elapsed:.3f}s, expected < 2.0s",
        )

    def test_select_pixel_no_rerender_coupling(self):
        """Two rapid select_pixel calls should both succeed without deadlocking,
        and server-side state should reflect the final selection."""
        view_id = self._setup_select_pixel()

        resp1 = self._post_select_pixel(view_id, pixel="[100, 200]")
        self.assertEqual(resp1.status_code, 200)

        # Second call with a different pixel (appended as additionalCell)
        resp2 = self._post_select_pixel(view_id, pixel="[300, 400]")
        self.assertEqual(resp2.status_code, 200)

        # Server-side state should show firstCell from the first click
        # and the second click in additionalCells
        session = harmonyServer.SESSIONS[view_id]
        self.assertEqual(session.selection.firstCell, (1, 2))
        self.assertIsNotNone(session.selection.additionalCells)


if __name__ == "__main__":
    unittest.main()
