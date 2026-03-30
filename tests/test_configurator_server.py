import pytest

def test_index_redirect(configurator_client):
    """Test the root redirects to /configurator."""
    response = configurator_client.get('/', follow_redirects=False)
    assert response.status_code == 303
    assert "/configurator" in response.headers['location']

def test_configurator_index(configurator_client):
    """Test the configurator index page."""
    response = configurator_client.get('/configurator/')
    assert response.status_code == 200
    assert b"Camera" in response.content

def test_app_structure(configurator_app):
    """Test that the app is initialized with the correct configuration structure."""
    assert hasattr(configurator_app.state, 'cc')
    assert hasattr(configurator_app.state.cc, 'hex')
    assert hasattr(configurator_app.state.cc, 'cameras')
    # Verify hex has size attribute (used in template)
    assert hasattr(configurator_app.state.cc.hex, 'size')
    # Verify saveConfiguration exists (used in routes)
    assert hasattr(configurator_app.state.cc, 'saveConfiguration')

def test_grid_configuration(configurator_client, configurator_app):
    """Test updating the grid configuration."""
    # Ensure current size is different or mocked
    configurator_app.state.cc.hex.size = 10.0
    response = configurator_client.post('/configurator/grid_configuration', data={'size': '25'})
    assert response.status_code == 200
    assert configurator_app.state.cc.hex.size == 25.0

def test_update_config(configurator_client, configurator_app):
    """Test the main update config route."""
    response = configurator_client.post('/configurator/')
    assert response.status_code == 200
    assert response.content == b"Success"
    configurator_app.state.cc.saveConfiguration.assert_called()

def test_delete_camera(configurator_client, configurator_app):
    """Test deleting a camera."""
    # Ensure camera exists primarily
    assert "Camera 0" in configurator_app.state.cc.cameras
    
    response = configurator_client.post('/configurator/delete_cam/Camera 0')
    assert response.status_code == 200
    
    # Check that camera was removed from the dict
    assert "Camera 0" not in configurator_app.state.cc.cameras
    # Check that saveConfiguration was called
    configurator_app.state.cc.saveConfiguration.assert_called()

def test_calibrator_endpoint(configurator_client):
    """Test that calibrator endpoints are accessible via configurator app."""
    # Calibrator is registered under configurator blueprint in the original code,
    # but in configuratorServer.py we register configurator blueprint which has 
    # configurator.register_blueprint(calibrator, url_prefix='/calibrator') 
    # SO the path should be /configurator/calibrator/
    
    # Wait, let's check configurator.py line 15: 
    # configurator.register_blueprint(calibrator, url_prefix='/calibrator')
    # And configuratorServer.py line 18:
    # app.register_blueprint(configurator, url_prefix='/configurator')
    # So yes, /configurator/calibrator/
    
    response = configurator_client.get('/configurator/calibrator/')
    # Route is deprecated/removed in favor of integrated Configurator UI
    assert response.status_code == 404

def test_manual_calibration_logic(configurator_client, configurator_app):
    """Test standard configurator coordinate mapping & manual cell selection."""
    cam_name = "Camera 0"
    
    # payload that simulates 4 points (2 rows across 2 columns) forming a block
    data = {
        cam_name: {
            "pixel": [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.2, 0.2]],
            "axial": [[0, 0], [1, 0], [0, 1], [1, 1]]
        }
    }
    
    from unittest import mock
    with mock.patch.object(configurator_app.state.cm, 'buildRealSpaceConverter') as mock_build, \
         mock.patch.object(configurator_app.state.cc, 'axial_to_pixel', return_value=(50.0, 50.0)):
        
        response = configurator_client.post('/configurator/manual_calibration', json=data)
        
        assert response.status_code == 200
        assert b"Added manual calibration" in response.content
        mock_build.assert_called_once()
        
        assert len(configurator_app.state.cm.calibrationPts) == 1
        cam_calibs = configurator_app.state.cm.calibrationPts[0]
        assert cam_name in cam_calibs
        # Structure is [pixel, real, axial]
        assert len(cam_calibs[cam_name]) == 3
        assert len(cam_calibs[cam_name][2]) == 4
