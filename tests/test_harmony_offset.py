
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import json

import sys
import os

# Adjust path to import harmony modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../harmony')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../observer')))

from harmonyServer import harmony, create_harmony_app, perspective_res
import harmonyServer

@pytest.fixture
def app():
    # Mocking dependencies
    with patch('harmonyServer.HexCaptureConfiguration') as MockCC, \
         patch('harmonyServer.HarmonyMachine') as MockCM, \
         patch('harmonyServer.HexGridConfiguration') as MockHexGrid, \
         patch('harmonyServer.registerCaptureService'):
        
        app = create_harmony_app()
        
        # Setup mock cameras
        mock_cam = MagicMock()
        # Set a small resolution to verify scaling (e.g., 100x200)
        # perspective_res is 1920x1080
        # Expected scale: x=19.2, y=5.4
        mock_frame = np.zeros((200, 100, 3), dtype=np.uint8) 
        mock_cam.mostRecentFrame = mock_frame
        mock_cam.activeZoneBoundingBox = (0, 0, 100, 200) # x, y, w, h
        
        # Wire cameras into module-level state
        harmonyServer._cc.cameras = {'Camera 1': mock_cam}
        
        # Setup mock memory object
        mock_obj = MagicMock()
        mock_obj.oid = "TestObject"
        
        # Point at (10, 10) in raw coordinates
        mock_change_set = MagicMock()
        mock_change_set.changePoints = [[10, 10]]
        
        mock_obj.changeSet = {'Camera 1': mock_change_set}
        
        # Mock changeSetToAxialCoord for VirtualMap
        harmonyServer._cm.cc.changeSetToAxialCoord.return_value = (0, 0)
        harmonyServer._cc.changeSetToAxialCoord.return_value = (0, 0)
        
        # Mock realSpaceBoundingBox
        harmonyServer._cm.cc.realSpaceBoundingBox.return_value = (0, 0, 800, 800)
        harmonyServer._cm.cc.hex.width = 1600
        harmonyServer._cm.cc.hex.height = 1600
        
        harmonyServer._cm.memory = [mock_obj]
        
        yield app

@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)

def test_get_canvas_data_scaling(client, app):
    """
    Test that getCanvasData scales coordinates from raw camera resolution 
    to perspective_res (1920x1080).
    """
    from harmonyServer import SESSIONS, SessionConfig
    SESSIONS['test_view'] = SessionConfig()
    
    response = client.get('/harmony/canvas_data/test_view')
    assert response.status_code == 200
    
    data = response.json()
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
    print(f"Received: ({x}, {y}), Expected: ({expected_x}, {expected_y})")
    
    assert abs(x - expected_x) < 0.1
    assert abs(y - expected_y) < 0.1

def test_virtualmap_crop_scaling(client, app):
    """
    Test that getCanvasData scales and offsets VirtualMap coordinates 
    correctly when cropped with a 20px margin.
    """
    from harmonyServer import SESSIONS, SessionConfig
    SESSIONS['test_view'] = SessionConfig()
    
    harmonyServer._cm.cc.realSpaceBoundingBox.return_value = (500, 500, 800, 800)
    
    # We mock objectToHull to return a point at (500, 500)
    harmonyServer._cm.cc.objectToHull.return_value = np.array([[[500, 500]]], dtype=np.float32)
    
    response = client.get('/harmony/canvas_data/test_view')
    assert response.status_code == 200
    
    data = response.json()
    assert 'TestObject' in data['objects']
    assert 'VirtualMap' in data['objects']['TestObject']
    
    pts = data['objects']['TestObject']['VirtualMap']
    x, y = pts[0]
    
    # Logic: Offset = 500 - 20 = 480. Scale = 1200 / (800 + 40) = 1.4285714
    expected_x = (500 - 480) * (1200 / 840)
    expected_y = (500 - 480) * (1200 / 840)
    
    assert abs(x - expected_x) < 0.1
    assert abs(y - expected_y) < 0.1
