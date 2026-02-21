
import unittest
import unittest.mock as mock
import json
import io
import sys
import os

# Adjust path to include the parent directory so we can import 'harmony'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking modules only for the import of harmonyServer
with mock.patch.dict(sys.modules):
    # Mock cv2 before importing harmonyServer
    sys.modules['cv2'] = mock.MagicMock()
    sys.modules['matplotlib'] = mock.MagicMock()
    sys.modules['matplotlib.backends'] = mock.MagicMock()
    sys.modules['matplotlib.backends.backend_agg'] = mock.MagicMock()
    sys.modules['matplotlib.figure'] = mock.MagicMock()

    # Mock ipynb and observer
    sys.modules['ipynb'] = mock.MagicMock()
    sys.modules['ipynb.fs'] = mock.MagicMock()
    sys.modules['ipynb.fs.full'] = mock.MagicMock()
    sys.modules['ipynb.fs.full.HarmonyMachine'] = mock.MagicMock()
    sys.modules['observer'] = mock.MagicMock()
    sys.modules['observer.Observer'] = mock.MagicMock()
    # Mock submodules referenced in harmonyServer imports
    sys.modules['observer.configurator'] = mock.MagicMock()
    sys.modules['observer.observerServer'] = mock.MagicMock()
    sys.modules['observer.calibrator'] = mock.MagicMock()

    from harmony import harmonyServer

# Helper to mock generators
def mock_gen(*args, **kwargs):
    yield b'--frame\r\nContent-Type: image/jpg\r\n\r\nfake\r\n'

class TestHarmonyServer(unittest.TestCase):

    def setUp(self):
        # Create the app and test client
        self.app = harmonyServer.create_harmony_app()
        self.app.testing = True
        self.client = self.app.test_client()
        
        # Patch generators
        # Patch generators
        self.patchers = [
            mock.patch.object(harmonyServer, 'getConsoleImage', side_effect=mock_gen),
            mock.patch.object(harmonyServer, 'renderConsole', side_effect=mock_gen, create=True),
            mock.patch.object(harmonyServer, 'genCombinedCamerasView', side_effect=mock_gen),
            mock.patch.object(harmonyServer, 'genCombinedCameraWithChangesView', side_effect=mock_gen),
            mock.patch.object(harmonyServer, 'get_broadcaster')
        ]
        for patcher in self.patchers:
            mock_obj = patcher.start()
            if patcher.attribute == 'get_broadcaster':
                # Mock broadcaster instance
                mock_broadcaster = mock.MagicMock()
                mock_broadcaster.subscribe.side_effect = mock_gen
                mock_obj.return_value = mock_broadcaster

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    def test_harmony_dashboard(self):
        """Test main dashboard."""
        response = self.client.get('/harmony/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Harmony", response.data)

    def test_harmony_reset(self):
        """Test reset."""
        response = self.client.get('/harmony/reset')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"success", response.data)

    def test_harmony_console(self):
        """Test harmony console endpoint."""
        response = self.client.get('/harmony/harmony_console')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

    def test_combined_cameras(self):
        """Test combined cameras endpoint."""
        response = self.client.get('/harmony/combinedCameras')
        self.assertEqual(response.status_code, 200)

    def test_get_objects(self):
        """Test getting objects table."""
        with mock.patch.object(self.app.cc, 'changeSetToAxialCoord', return_value=(0, 0)):
             response = self.client.get('/harmony/objects')
             self.assertEqual(response.status_code, 200)

    def test_delete_object(self):
        """Test object deletion via factory."""
        view_id = "test_view"
        mock_selection = mock.MagicMock()
        mock_selection.firstCell = (0, 0)
        
        # Create a mock object in memory
        from harmony.harmonyServer import SessionConfig
        
        # We need to ensure simple_id_generator or similar doesn't collide or we force the viewId
        # Inject session
        harmonyServer.SESSIONS[view_id] = SessionConfig()
        harmonyServer.SESSIONS[view_id].selection.firstCell = (0, 0) # Mimic selection
        
        # Mock memory
        mock_mem = mock.MagicMock()
        mock_mem.oid = "TargetObject"
        self.app.cm.memory = [mock_mem]
        
        # Mock cc to match coordinates
        self.app.cc.changeSetToAxialCoord = mock.MagicMock(return_value=(0, 0))

        response = self.client.delete(f'/harmony/object_factory/{view_id}')
        self.assertEqual(response.status_code, 200)
        # Verify object removed
        self.assertNotIn(mock_mem, self.app.cm.memory)


    def test_select_pixel(self):
        """Test selecting a pixel puts it in SESSIONS."""
        self.app.cm.cc.camCoordToAxial.return_value = (1, 2)
        # Also mock for VirtualMap path if needed, though 'selectedCamera' below is 'Camera 0'
        self.app.cm.cc.pixel_to_axial.return_value = (1, 2)
        
        view_id = 'view1'
        # Pre-seed session
        from harmony.harmonyServer import SessionConfig
        harmonyServer.SESSIONS[view_id] = SessionConfig()
        
        data = {'viewId': view_id, 'selectedPixel': '[100, 200]', 'selectedCamera': 'Camera 0', 'appendPixel': ''}
        response = self.client.post('/harmony/select_pixel', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(view_id, harmonyServer.SESSIONS)
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, (1, 2))


    def test_cam_with_changes(self):
        response = self.client.get('/harmony/camWithChanges/Camera0/view1')
        self.assertEqual(response.status_code, 200)
        
    def test_combined_cameras_with_changes(self):
        response = self.client.get('/harmony/combinedCamerasWithChanges')
        self.assertEqual(response.status_code, 200)


    def test_build_object_factory(self):
        view_id = "test_view"
        from harmony.harmonyServer import SessionConfig
        harmonyServer.SESSIONS[view_id] = SessionConfig()
        harmonyServer.SESSIONS[view_id].selection.firstCell = (0, 0)
        
        response = self.client.get(f'/harmony/object_factory/{view_id}')
        self.assertEqual(response.status_code, 200)

    def test_minimap(self):
        response = self.client.get('/harmony/minimap/view1')
        self.assertEqual(response.status_code, 200)
        
    def test_clear_pixel(self):
        view_id = "view1"
        from harmony.harmonyServer import SessionConfig, CellSelection
        harmonyServer.SESSIONS[view_id] = SessionConfig()
        # Set something
        harmonyServer.SESSIONS[view_id].selection = CellSelection(firstCell=(1,1))
        
        response = self.client.get(f'/harmony/clear_pixel/{view_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(harmonyServer.SESSIONS[view_id].selection.firstCell, None)

    # Updated test to handle context and mocks correctly
    def test_object_table_ordering(self):
        """Test object table grouping and ordering."""
        view_id = "test_view"
        from harmony.harmonyServer import SessionConfig
        
        session = SessionConfig()
        session.moveable = ["ObjA"]
        session.allies = ["ObjB"]
        # ObjC is general
        
        harmonyServer.SESSIONS[view_id] = session
        
        with self.app.app_context():
            # Mock memory
            mock_a = mock.MagicMock()
            mock_a.oid = "ObjA"
            mock_b = mock.MagicMock()
            mock_b.oid = "ObjB"
            mock_c = mock.MagicMock()
            mock_c.oid = "ObjC"
            
            # Use current_app.cm since we are in context
            self.app.cm.memory = [mock_c, mock_b, mock_a] 
            
            with mock.patch.object(harmonyServer, 'captureToChangeRow') as mock_row:
                mock_row.side_effect = lambda x, color=None: x.oid
                output = harmonyServer.buildObjectTable(view_id)
            
            # Expected order: Moveable(A), Allies(B), Others(C)
            # Now with headers
            self.assertIn("<h4>Moveable</h4>", output)
            self.assertIn("ObjA", output)
            self.assertIn("<h4>Allies</h4>", output)
            self.assertIn("ObjB", output)
            self.assertIn("<h4>Selectable</h4>", output)
            self.assertIn("ObjC", output)
            
            # Simple check for order by checking indices
            idx_move = output.index("<h4>Moveable</h4>")
            idx_ally = output.index("<h4>Allies</h4>")
            idx_sel = output.index("<h4>Selectable</h4>")
            self.assertTrue(idx_move < idx_ally < idx_sel)

    def test_update_session_id_reclaim(self):
        """Test reclaiming an existing session ID."""
        old_id = "old_session"
        existing_id = "existing_session"
        
        from harmony.harmonyServer import SessionConfig
        
        # Setup sessions
        harmonyServer.SESSIONS[old_id] = SessionConfig()
        harmonyServer.SESSIONS[existing_id] = SessionConfig()
        
        # Test reclaiming existing_id
        response = self.client.post('/harmony/update_session_id', data={'viewId': old_id, 'newViewId': existing_id})
        # Should now allow it and return a redirect script
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"window.location.href", response.data)
        self.assertIn(b"window.location.href", response.data)
        self.assertIn(existing_id.encode(), response.data)

    def test_capture_to_change_row_with_color(self):
        """Test that captureToChangeRow calls custom_object_visual when color is provided."""
        with self.app.app_context():
            mock_capture = mock.MagicMock()
            mock_capture.oid = "ObjColor"
            
            # Mock custom_object_visual and object_visual
            # Mock custom_object_visual and object_visual
            with mock.patch.object(harmonyServer, 'custom_object_visual') as mock_custom, \
                 mock.patch('harmony.harmonyServer.current_app.cm.object_visual') as mock_default, \
                 mock.patch.object(harmonyServer, 'imageToBase64', return_value="fake_b64"), \
                 mock.patch.object(harmonyServer, 'url_for', return_value="/fake_url"), \
                 mock.patch('harmony.harmonyServer.current_app.cm.cc.trackedObjectLastDistance', return_value=10.0):
                
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
        response = self.client.get('/harmony/')
        self.assertEqual(response.status_code, 200)
        cookie = None
        # Werkzeug 2.x/3.x cookie handling in test client
        # response.headers['Set-Cookie'] might contain it
        self.assertTrue(any('session_view_id=' in h[1] for h in response.headers if h[0] == 'Set-Cookie'))
        
        # Extract the set cookie value
        cookie_header = [h[1] for h in response.headers if h[0] == 'Set-Cookie'][0]
        # format: session_view_id=VALUE; Path=/
        val = cookie_header.split(';')[0].split('=')[1]
        
        # Case 2: Send cookie -> Should maintain session
        self.client.set_cookie('session_view_id', val)
        response2 = self.client.get('/harmony/')
        self.assertEqual(response2.status_code, 200)
        # Verify the viewId in the page matches our cookie
        self.assertIn(f"Session ID: {val}".encode(), response2.data)
        
        # Case 3: Send Param -> Should override cookie and update it
        new_val = "OVERRIDE-SESSION"
        response3 = self.client.get(f'/harmony/?viewId={new_val}')
        self.assertEqual(response3.status_code, 200)
        self.assertTrue(any(f'session_view_id={new_val}' in h[1] for h in response3.headers if h[0] == 'Set-Cookie'))
        self.assertIn(f"Session ID: {new_val}".encode(), response3.data)

if __name__ == '__main__':
    unittest.main()
