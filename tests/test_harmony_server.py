
import pytest
import unittest.mock as mock
import json
import numpy as np
import numpy as np

# Helper to mock generators
def mock_gen(*args, **kwargs):
    yield b'--frame\r\nContent-Type: image/jpg\r\n\r\nfake\r\n'

@pytest.fixture
def harmony_client_patched(harmony_app):
    """Client with patched generators."""
    with mock.patch('harmony.harmonyServer.renderConsole', side_effect=mock_gen), \
         mock.patch('harmony.harmonyServer.genCombinedCamerasView', side_effect=mock_gen), \
         mock.patch('harmony.harmonyServer.genCameraWithChangesView', side_effect=mock_gen), \
         mock.patch('harmony.harmonyServer.fullCam', side_effect=mock_gen), \
         mock.patch('harmony.harmonyServer.genCombinedCameraWithChangesView', side_effect=mock_gen), \
         mock.patch('harmony.harmonyServer.minimapGenerator', side_effect=mock_gen):
        yield harmony_app.test_client()

def test_harmony_dashboard(harmony_client):
    """Test main dashboard."""
    response = harmony_client.get('/harmony/')
    assert response.status_code == 200
    assert b"Harmony" in response.data

def test_harmony_reset(harmony_client):
    """Test reset."""
    response = harmony_client.get('/harmony/reset')
    assert response.status_code == 200
    assert b"success" in response.data

def test_harmony_console(harmony_client_patched):
    """Test harmony console endpoint."""
    response = harmony_client_patched.get('/harmony/harmony_console')
    assert response.status_code == 200
    assert response.mimetype == 'multipart/x-mixed-replace'

def test_combined_cameras(harmony_client_patched):
    """Test combined cameras endpoint."""
    response = harmony_client_patched.get('/harmony/combinedCameras')
    assert response.status_code == 200

def test_get_objects(harmony_client):
    """Test getting objects table."""
    app = harmony_client.application
    app.cc.changeSetToAxialCoord.return_value = (0, 0)
    response = harmony_client.get('/harmony/objects')
    assert response.status_code == 200


def test_delete_object(harmony_client):
    """Test object deletion via factory."""
    view_id = "test_view"
    mock_selection = mock.MagicMock()
    mock_selection.firstCell = (0, 0)
    mock_mem = mock.MagicMock()
    app = harmony_client.application
    app.cc.changeSetToAxialCoord.return_value = (0, 0)
    app.cc.changeSetToAxialCoord.return_value = (0, 0)
    app.cm.memory = [mock_mem]
    from harmony import harmonyServer
    
    with mock.patch.dict(harmonyServer.SELECTED_CELLS, {view_id: mock_selection}, clear=True):
        response = harmony_client.delete(f'/harmony/object_factory/{view_id}')
        assert response.status_code == 200
        assert mock_mem not in app.cm.memory

def test_select_pixel(harmony_client):
    """Test selecting a pixel puts it in SELECTED_CELLS."""
    app = harmony_client.application
    app.cc.camCoordToAxial.return_value = (1, 2)
    app = harmony_client.application
    app.cc.camCoordToAxial.return_value = (1, 2)
    from harmony import harmonyServer
    with mock.patch.dict(harmonyServer.SELECTED_CELLS, {}, clear=True):
        data = {'viewId': 'view1', 'selectedPixel': '[100, 200]', 'selectedCamera': 'Camera 0', 'appendPixel': ''}
        response = harmony_client.post('/harmony/select_pixel', data=data)
        assert response.status_code == 200
        assert 'view1' in harmonyServer.SELECTED_CELLS
        assert harmonyServer.SELECTED_CELLS['view1'].firstCell == (1, 2)




def test_cam_with_changes(harmony_client_patched):
    response = harmony_client_patched.get('/harmony/camWithChanges/Camera0/view1')
    assert response.status_code == 200

def test_full_cam(harmony_client_patched):
    response = harmony_client_patched.get('/harmony/fullCam/Camera0')
    assert response.status_code == 200

def test_combined_cameras_with_changes(harmony_client_patched):
    response = harmony_client_patched.get('/harmony/combinedCamerasWithChanges')
    assert response.status_code == 200


def test_build_object_factory(harmony_client):
    view_id = "test_view"
    mock_selection = mock.MagicMock()
    mock_selection.firstCell = (0, 0)
    view_id = "test_view"
    mock_selection = mock.MagicMock()
    mock_selection.firstCell = (0, 0)
    from harmony import harmonyServer
    with mock.patch.dict(harmonyServer.SELECTED_CELLS, {view_id: mock_selection}, clear=True):
        response = harmony_client.get(f'/harmony/object_factory/{view_id}')
        assert response.status_code == 200

def test_minimap(harmony_client_patched):
    response = harmony_client_patched.get('/harmony/minimap/view1')
    assert response.status_code == 200
    

def test_clear_pixel(harmony_client):
    view_id = "view1"
    from harmony import harmonyServer
    with mock.patch.dict(harmonyServer.SELECTED_CELLS, {view_id: mock.Mock()}, clear=True):
        response = harmony_client.get(f'/harmony/clear_pixel/{view_id}')
        assert response.status_code == 200
        assert view_id not in harmonyServer.SELECTED_CELLS
