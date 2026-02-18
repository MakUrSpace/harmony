import pytest

def test_index_redirect(configurator_client):
    """Test the root redirects to /configurator."""
    response = configurator_client.get('/')
    # Flask test client follow_redirects defaults to False.
    # Expect 303 redirect to /configurator
    assert response.status_code == 303
    assert "/configurator" in response.headers['Location']

def test_configurator_index(configurator_client):
    """Test the configurator index page."""
    response = configurator_client.get('/configurator/')
    assert response.status_code == 200
    assert b"Camera" in response.data

def test_app_structure(configurator_app):
    """Test that the app is initialized with the correct configuration structure."""
    assert hasattr(configurator_app, 'cc')
    assert hasattr(configurator_app.cc, 'hex')
    assert hasattr(configurator_app.cc, 'cameras')
    # Verify hex has size attribute (used in template)
    assert hasattr(configurator_app.cc.hex, 'size')
    # Verify saveConfiguration exists (used in routes)
    assert hasattr(configurator_app.cc, 'saveConfiguration')

def test_grid_configuration(configurator_client, configurator_app):
    """Test updating the grid configuration."""
    # Ensure current size is different or mocked
    configurator_app.cc.hex.size = 10.0
    response = configurator_client.post('/configurator/grid_configuration', data={'size': '25'})
    assert response.status_code == 200
    assert configurator_app.cc.hex.size == 25.0

def test_update_config(configurator_client, configurator_app):
    """Test the main update config route."""
    response = configurator_client.post('/configurator/')
    assert response.status_code == 200
    assert response.data == b"Success"
    configurator_app.cc.saveConfiguration.assert_called()

def test_delete_camera(configurator_client, configurator_app):
    """Test deleting a camera."""
    # Ensure camera exists primarily
    assert "Camera 0" in configurator_app.cc.cameras
    
    response = configurator_client.post('/configurator/delete_cam/Camera 0')
    assert response.status_code == 200
    
    # Check that camera was removed from the dict
    assert "Camera 0" not in configurator_app.cc.cameras
    # Check that saveConfiguration was called
    configurator_app.cc.saveConfiguration.assert_called()

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
    assert response.status_code == 200
