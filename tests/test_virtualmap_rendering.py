import os
import json
import pytest
from unittest.mock import MagicMock
from harmony.harmonyServer import render_minimap, render_camera
from harmony.HarmonyMachine import HarmonyMachine

def test_virtualmap_rendering_success(monkeypatch):
    """
    Tests that the virtualmap generator (render_minimap) successfully generates
    a valid JPEG payload without throwing exceptions or returning None.
    Uses mocks to prevent real camera initialization and ensure fast execution.
    """
    # Mock _get_cc to prevent real global state access during render_minimap
    mock_cc = MagicMock()
    mock_cc.show_grid = False
    mock_cc.show_objects = False
    mock_cc.realSpaceBoundingBox.return_value = (0, 0, 1000, 1000)
    mock_cc.hex.width = 1600
    mock_cc.hex.height = 1600
    
    monkeypatch.setattr("harmony.harmonyServer._get_cc", lambda: mock_cc)

    # Initialize a fast mock HarmonyMachine
    cm = MagicMock()
    cm.cc = mock_cc
    cm.show_grid = False
    cm.objectsAndColors = []
    
    # We need buildMiniMap to actually return a valid numpy image
    import numpy as np
    cm.buildMiniMap.return_value = np.zeros([1600, 1600, 3], dtype="uint8")
    
    # Test 1: Render with grid OFF (default)
    result_no_grid = render_minimap(cm)
    
    assert result_no_grid is not None, "render_minimap returned None instead of JPEG bytes"
    assert len(result_no_grid) > 0, "render_minimap returned an empty byte array"
    
    # Test 2: Render with grid ON
    mock_cc.show_grid = True
    result_with_grid = render_minimap(cm)
    
    assert result_with_grid is not None, "render_minimap returned None when show_grid=True"
    assert len(result_with_grid) > 0, "render_minimap returned an empty byte array when show_grid=True"

    # Both should be valid image bytes (JPEG starts with \xff\xd8)
    assert result_no_grid.startswith(b"\xff\xd8"), "Output is not a valid JPEG"
    assert result_with_grid.startswith(b"\xff\xd8"), "Output is not a valid JPEG"

def test_render_camera_fallback_no_frame(monkeypatch):
    """
    Ensures that render_camera returns a valid fallback JPEG instead of None
    when the camera has not yet captured a frame (e.g., during RTSP initialization).
    """
    mock_cc = MagicMock()
    mock_cc.cameras = {}
    
    # Create a mock camera with NO frame
    mock_cam = MagicMock()
    mock_cam.mostRecentFrame = None
    mock_cc.cameras[0] = mock_cam
    
    monkeypatch.setattr("harmony.harmonyServer._get_cm", lambda: MagicMock())
    
    # "0" is passed as a string, but the dict uses int key. 
    # Our recent fix ensures this works correctly.
    result = render_camera(mock_cc, "0")
    
    assert result is not None, "render_camera returned None when frame was missing"
    assert result.startswith(b"\xff\xd8"), "Fallback is not a valid JPEG"

def test_render_camera_fallback_missing_camera(monkeypatch):
    """
    Ensures that render_camera returns a valid fallback JPEG instead of None
    when the requested camera doesn't exist in the configuration.
    """
    mock_cc = MagicMock()
    mock_cc.cameras = {}  # Empty cameras dict
    
    monkeypatch.setattr("harmony.harmonyServer._get_cm", lambda: MagicMock())
    
    result = render_camera(mock_cc, "99")
    
    assert result is not None, "render_camera returned None when camera was missing"
    assert result.startswith(b"\xff\xd8"), "Fallback is not a valid JPEG"

def test_render_minimap_fallback_no_frame(monkeypatch):
    """
    Ensures that render_minimap returns a valid fallback JPEG instead of None
    when buildMiniMap returns None (e.g., if calibration fails).
    """
    mock_cc = MagicMock()
    monkeypatch.setattr("harmony.harmonyServer._get_cc", lambda: mock_cc)

    cm = MagicMock()
    cm.cc = mock_cc
    cm.buildMiniMap.return_value = None  # Force failure
    
    result = render_minimap(cm)
    
    assert result is not None, "render_minimap returned None when buildMiniMap failed"
    assert result.startswith(b"\xff\xd8"), "Fallback is not a valid JPEG"
