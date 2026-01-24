
import unittest
import unittest.mock as mock
import json
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../observer')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../harmony')))


# Mock cv2 before importing harmonyServer
mock_cv2 = mock.MagicMock()
sys.modules['cv2'] = mock_cv2

# Mock matplotlib
mock_mpl = mock.MagicMock()
sys.modules['matplotlib'] = mock_mpl
sys.modules['matplotlib.backends'] = mock_mpl
sys.modules['matplotlib.backends.backend_agg'] = mock_mpl
sys.modules['matplotlib.figure'] = mock_mpl

# Mock ipynb module structure
mock_ipynb = mock.MagicMock()
sys.modules['ipynb'] = mock_ipynb
sys.modules['ipynb.fs'] = mock_ipynb
sys.modules['ipynb.fs.full'] = mock_ipynb


# Mock specific ipynb modules that are imported
sys.modules['ipynb.fs.full.HarmonyMachine'] = mock_ipynb
sys.modules['ipynb.fs.full.Observer'] = mock_ipynb
sys.modules['ipynb.fs.full.CalibratedObserver'] = mock_ipynb
sys.modules['ipynb.fs.full.HexObserver'] = mock_ipynb
sys.modules['ipynb.fs.full.HexGridConfiguration'] = mock_ipynb
sys.modules['ipynb.fs.full.HexCaptureConfiguration'] = mock_ipynb

# Mock observer.calibrator
mock_calibrator = mock.MagicMock()
sys.modules['observer.calibrator'] = mock_calibrator
# Set attributes that are imported from it
mock_calibrator.calibrator = mock.MagicMock()
mock_calibrator.CalibratedCaptureConfiguration = mock.MagicMock()
mock_calibrator.registerCaptureService = mock.MagicMock()
mock_calibrator.DATA_LOCK = mock.MagicMock()
mock_calibrator.CONSOLE_OUTPUT = "Mock Console"

# Mock observer.observerServer
mock_obs_server = mock.MagicMock()
sys.modules['observer.observerServer'] = mock_obs_server
mock_obs_server.observer = mock.MagicMock()
mock_obs_server.configurator = mock.MagicMock()

# Mock observer.configurator
mock_configurator = mock.MagicMock()
sys.modules['observer.configurator'] = mock_configurator
mock_configurator.configurator = mock.MagicMock() # Ensure blueprint exists

from flask import Flask

class TestMultiSelection(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.patcher_cc = mock.patch('observer.calibrator.CalibratedCaptureConfiguration')
        self.mock_cc_cls = self.patcher_cc.start()
        
        self.patcher_hex = mock.patch('observer.HexGridConfiguration')
        self.patcher_hex.start()
        
        self.patcher_hcapt = mock.patch('observer.HexCaptureConfiguration')
        self.patcher_hcapt.start()
        

        self.patcher_reg = mock.patch('harmonyServer.registerCaptureService')
        self.patcher_reg.start()
        
        self.patcher_hm = mock.patch('harmonyServer.HarmonyMachine')
        self.mock_hm_cls = self.patcher_hm.start()
        
        # Setup CC mock
        self.mock_cc = self.mock_cc_cls.return_value
        self.mock_cc.cameras = {"Camera 0": mock.MagicMock()}
        
        # Setup HM mock
        def harmony_machine_side_effect(cc):
            m = mock.MagicMock()
            m.cc = cc
            m.memory = []
            return m
        self.mock_hm_cls.side_effect = harmony_machine_side_effect
        
        # Create App
        from harmonyServer import create_harmony_app, SESSIONS, SessionConfig
        self.app = create_harmony_app() # This clears SESSIONS in some versions or not?
        # SESSIONS is global in module, create_harmony_app resets it?
        # harmonyServer.start_servers clears APPS but create_harmony_app appends.
        # Check resetHarmony().
        
        # Manually add test session
        SESSIONS['test_session'] = SessionConfig()
        
        self.app.cc = self.mock_cc
        if hasattr(self.app, 'cm'):
             self.app.cm.cc = self.mock_cc
             
        self.client = self.app.test_client()
        
        # Common mocks for logic
        self.app.cc.camCoordToAxial.side_effect = lambda cam, pt: (int(pt[0]), int(pt[1]))
        self.app.cc.pixel_to_axial.side_effect = lambda x, y: (int(x), int(y))
        self.app.cc.axial_distance.side_effect = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

    def tearDown(self):
        mock.patch.stopall()

    def test_multi_selection_logic(self):
        # Mock Memory Objects
        mock_obj1 = mock.MagicMock()
        mock_obj1.oid = "Obj1"
        # Remove hardcoded objectType, rely on session config
        del mock_obj1.objectType 
        
        mock_obj2 = mock.MagicMock()
        mock_obj2.oid = "Obj2"
        del mock_obj2.objectType
        
        self.app.cm.memory = [mock_obj1, mock_obj2]
        
        # Configure Session Types
        from harmonyServer import SESSIONS
        session = SESSIONS['test_session']
        # Obj1 matches both Ally and Terrain -> Should be Ally (higher priority)
        session.allies = ["Obj1"]
        session.terrain = ["Obj1", "Obj2"] 
        # Obj2 is only in Terrain -> Terrain
        # Add Obj3 (fallback)
        
        mock_obj3 = mock.MagicMock()
        mock_obj3.oid = "Obj3"
        del mock_obj3.objectType
        self.app.cm.memory.append(mock_obj3)
        
        # Mock changeSetToAxialCoord
        def mock_cstac(mem):
            if mem.oid == "Obj1":
                return (10, 10)
            if mem.oid == "Obj2":
                return (20, 20)
            if mem.oid == "Obj3":
                return (30, 30)
            return (0, 0)
        self.app.cc.changeSetToAxialCoord.side_effect = mock_cstac
        
        with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 0, 0)):
            # Step 1: Select (10, 10) - Obj1 (Ally > Terrain)
            resp = self.client.post('/harmony/select_pixel', data={
                "viewId": "test_session",
                "selectedPixel": json.dumps([10, 10]),
                "selectedCamera": "Camera 0",
                "appendPixel": "" 
            })
            
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Selected First Cell: (10, 10)", html)
            self.assertIn("Object: Obj1", html)
            self.assertIn("Type: Ally", html)
            
            # Ensure "Delete Object" IS present (Default is Admin)
            self.assertIn("Delete Object", html)

            self.assertNotIn("Type: Terrain", html) # Ensure only highest priority is shown

            
            # Step 2: Select (20, 20) with appendPixel - Obj2 (Terrain)
            resp = self.client.post('/harmony/select_pixel', data={
                "viewId": "test_session",
                "selectedPixel": json.dumps([20, 20]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true"
            })
            
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Selected First Cell: (10, 10)", html)
            self.assertIn("Latest Selection: (20, 20)", html)
            self.assertIn("Object: Obj2 (Terrain)", html)
            self.assertIn("Dist to First: 20 cells", html)
            
            # Step 3: Select (15, 15) with appendPixel - No Object
            resp = self.client.post('/harmony/select_pixel', data={
                "viewId": "test_session",
                "selectedPixel": json.dumps([15, 15]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true"
            })
            
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Latest Selection: (15, 15)", html)
            # Check for delimiter
            self.assertIn("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", html)
            self.assertIn("Selection 2: (20, 20)", html)
            
            # Step 4: Select (30, 30) - Obj3 (No explicit group -> Default Selectable)
            resp = self.client.post('/harmony/select_pixel', data={
                "viewId": "test_session",
                "selectedPixel": json.dumps([30, 30]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true"
            })
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Object: Obj3 (Selectable)", html)

            self.assertIn("Object: Obj3 (Selectable)", html)

    def test_admin_vs_user_delete(self):
         # Helper to create client with specific template config
         def create_client(template_name):
             from harmonyServer import create_harmony_app
             app = create_harmony_app(template_name=template_name)
             app.cc = self.mock_cc
             if hasattr(app, 'cm'):
                 app.cm.cc = self.mock_cc
             # Re-mock dependencies for new app instance if needed
             app.cc.camCoordToAxial.side_effect = lambda cam, pt: (int(pt[0]), int(pt[1]))
             return app.test_client()

         # 1. Test Admin (Harmony.html) -> Should have Delete
         admin_client = create_client("Harmony.html")
         
         # Initialize Admin Session
         from harmonyServer import SESSIONS, SessionConfig
         SESSIONS['admin_session'] = SessionConfig()
         
         with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 0, 0)):
             # Setup object
             mock_obj = mock.MagicMock()
             mock_obj.oid = "AdminObj"
             del mock_obj.objectType
             self.app.cm.memory = [mock_obj]
             
             # Mock axial
             self.app.cc.changeSetToAxialCoord.side_effect = lambda mem: (10, 10)
             
             resp = admin_client.post('/harmony/select_pixel', data={
                "viewId": "admin_session",
                "selectedPixel": json.dumps([10, 10]),
                "selectedCamera": "Camera 0",
                "appendPixel": "" 
             })
             
             html = resp.data.decode()
             self.assertIn("Delete Object", html)

         # 2. Test User (HarmonyUser.html) -> Should NOT have Delete
         user_client = create_client("HarmonyUser.html")
         
         # Initialize User Session
         SESSIONS['user_session'] = SessionConfig()
         
         with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 0, 0)):
             # Setup object (shared memory in real app, mocked here)
             mock_obj = mock.MagicMock()
             mock_obj.oid = "UserObj"
             del mock_obj.objectType
             self.app.cm.memory = [mock_obj]
             
             # Mock axial
             self.app.cc.changeSetToAxialCoord.side_effect = lambda mem: (20, 20)
             
             resp = user_client.post('/harmony/select_pixel', data={
                "viewId": "user_session",
                "selectedPixel": json.dumps([20, 20]),
                "selectedCamera": "Camera 0",
                "appendPixel": "" 
             })
             
             html = resp.data.decode()
             self.assertNotIn("Delete Object", html)

             self.assertNotIn("Delete Object", html)

    def test_move_permissions(self):
         # Helper to create client with specific template config
         def create_client(template_name):
             from harmonyServer import create_harmony_app
             app = create_harmony_app(template_name=template_name)
             app.cc = self.mock_cc
             if hasattr(app, 'cm'):
                 app.cm.cc = self.mock_cc
             app.cc.camCoordToAxial.side_effect = lambda cam, pt: (int(pt[0]), int(pt[1]))
             return app.test_client()

         # 1. User: Non-Moveable Object -> No Move Button
         user_client = create_client("HarmonyUser.html")
         # Init session
         from harmonyServer import SESSIONS, SessionConfig
         SESSIONS['user_session_move'] = SessionConfig()
         
         with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 0, 0)):
             self.app.cm.memory = []
             mock_obj = mock.MagicMock()
             mock_obj.oid = "Statue"
             del mock_obj.objectType
             self.app.cm.memory.append(mock_obj)
             
             self.app.cc.changeSetToAxialCoord.side_effect = lambda mem: (10, 10)
             
             # Need two selections to trigger move option
             resp = user_client.post('/harmony/select_pixel', data={
                "viewId": "user_session_move",
                "selectedPixel": json.dumps([10, 10]),
                "selectedCamera": "Camera 0",
                "appendPixel": "" 
             })
             resp = user_client.post('/harmony/select_pixel', data={
                "viewId": "user_session_move",
                "selectedPixel": json.dumps([20, 20]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true" 
             })
             html = resp.data.decode()
             self.assertIn("Selected First Cell: (10, 10)", html)
             self.assertIn("Latest Selection: (20, 20)", html)
             self.assertNotIn("Move Statue Here", html)

         # 2. User: Moveable Object -> Has Move Button
         SESSIONS['user_session_move'].moveable = ["Hero"]
         mock_obj.oid = "Hero"
         
         with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 40, 40)):
             # Re-select
             resp = user_client.post('/harmony/select_pixel', data={
                "viewId": "user_session_move",
                "selectedPixel": json.dumps([10, 10]), 
                "selectedCamera": "Camera 0",
                "appendPixel": ""
             })
             resp = user_client.post('/harmony/select_pixel', data={
                "viewId": "user_session_move",
                "selectedPixel": json.dumps([20, 20]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true"
             })
             html = resp.data.decode()
             self.assertIn("Move Hero Here", html)

         # 3. Admin: Non-Moveable Object -> Has Move Button (Override)
         admin_client = create_client("Harmony.html")
         SESSIONS['admin_session_move'] = SessionConfig() # Empty config, nothing moveable
         
         with mock.patch('harmonyServer.get_conversion_params', return_value=(1.0, 1.0, 0, 0)):
             self.app.cm.memory = [] # Clear previous
             mock_obj = mock.MagicMock()
             mock_obj.oid = "Statue"
             del mock_obj.objectType
             self.app.cm.memory.append(mock_obj)
             
             resp = admin_client.post('/harmony/select_pixel', data={
                "viewId": "admin_session_move",
                "selectedPixel": json.dumps([10, 10]),
                "selectedCamera": "Camera 0",
                "appendPixel": "" 
             })
             resp = admin_client.post('/harmony/select_pixel', data={
                "viewId": "admin_session_move",
                "selectedPixel": json.dumps([20, 20]),
                "selectedCamera": "Camera 0",
                "appendPixel": "true" 
             })
             html = resp.data.decode()
             self.assertIn("Move Statue Here", html)

if __name__ == '__main__':
    unittest.main()
