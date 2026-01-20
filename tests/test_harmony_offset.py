
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from flask import Flask

import sys
import os

# Adjust path to import harmony modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../harmony')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../observer')))

from harmonyServer import harmony, create_harmony_app, perspective_res

@pytest.fixture
def app():
    # Mocking dependencies
    with patch('harmonyServer.HexCaptureConfiguration') as MockCC, \
         patch('harmonyServer.HarmonyMachine') as MockCM, \
         patch('harmonyServer.HexGridConfiguration') as MockHexGrid, \
         patch('harmonyServer.registerCaptureService'):
        
        app = create_harmony_app()
        app.secret_key = 'test_secret'
        app.config['TESTING'] = True
        
        # Setup mock cameras
        mock_cam = MagicMock()
        # Set a small resolution to verify scaling (e.g., 100x200)
        # perspective_res is 1920x1080
        # Expected scale: x=19.2, y=5.4
        mock_frame = np.zeros((200, 100, 3), dtype=np.uint8) 
        mock_cam.mostRecentFrame = mock_frame
        
        app.cc.cameras = {'Camera 1': mock_cam}
        
        # Setup mock memory object
        mock_obj = MagicMock()
        mock_obj.oid = "TestObject"
        
        # Point at (10, 10) in raw coordinates
        mock_change_set = MagicMock()
        mock_change_set.changePoints = [[10, 10]]
        
        mock_obj.changeSet = {'Camera 1': mock_change_set}
        
        # Mock changeSetToAxialCoord for VirtualMap
        app.cm.cc.changeSetToAxialCoord.return_value = (0, 0)
        
        app.cm.memory = [mock_obj]
        
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_get_canvas_data_scaling(client, app):
    """
    Test that getCanvasData scales coordinates from raw camera resolution 
    to perspective_res (1920x1080).
    """
    # Create a session
    with client.session_transaction() as sess:
        # We might need to initialize session in the server manually or via endpoint
        pass

    # Call the endpoint
    # We need a valid game/view ID. harmonyServer uses SESSIONS global.
    # We can inject a session directly into the global SESSIONS for testing.
    from harmonyServer import SESSIONS, SessionConfig
    SESSIONS['test_view'] = SessionConfig()
    
    response = client.get('/harmony/canvas_data/test_view')
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'objects' in data
    assert 'TestObject' in data['objects']
    assert 'Camera 1' in data['objects']['TestObject']
    
    points = data['objects']['TestObject']['Camera 1']
    assert len(points) == 1
    
    x, y = points[0]
    
    # Calculate expected scaling
    # Raw: (10, 10)
    # Resolution: 100x200 (w=100, h=200)
    # Target: 1920x1080
    # Scale X: 1920 / 100 = 19.2
    # Scale Y: 1080 / 200 = 5.4
    
    expected_x = 10 * (1920 / 100)
    expected_y = 10 * (1080 / 200)
    
    # Check if scaled (allowing for small float errors)
    # NOTE: This test is EXPECTED TO FAIL until the fix is implemented.
    # Currently it returns raw coordinates (10, 10)
    
    print(f"Received: ({x}, {y}), Expected: ({expected_x}, {expected_y})")
    
    assert abs(x - expected_x) < 0.1
    assert abs(y - expected_y) < 0.1
