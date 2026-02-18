
import pytest
import sys
import os
import unittest.mock as mock
import numpy as np

# Ensure we can import from root modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../observer')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../harmony')))

@pytest.fixture(scope="session", autouse=True)
def mock_cv2():
    """Global mock for cv2 interactions"""
    mock_cv2_module = mock.MagicMock()
    mock_cap = mock.MagicMock()
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.isOpened.return_value = True
    
    mock_cv2_module.VideoCapture.return_value = mock_cap
    mock_cv2_module.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cv2_module.imencode.return_value = (True, b'fake_image_data')
    
    # Patch sys.modules so import cv2 works everywhere
    with mock.patch.dict(sys.modules, {'cv2': mock_cv2_module}):
        yield mock_cv2_module

@pytest.fixture(autouse=True)
def mock_file_lock():
    with mock.patch('observer.file_lock.FileLock') as MockLock:
        mock_lock = MockLock.return_value
        mock_lock.acquire.return_value = None
        mock_lock.release.return_value = None
        yield MockLock

@pytest.fixture
def observer_app(mock_cv2):
    """Fixture to create the Observer app with mocked hardware."""
    with mock.patch('observer.calibrator.CalibratedCaptureConfiguration') as MockCC:
        mock_cc = MockCC.return_value
        mock_cc.cameras = {"Camera 0": mock.MagicMock()}
        mock_cc.cameras["Camera 0"].mostRecentFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cc.cameras["Camera 0"].cropToActiveZone.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        from flask import Flask
        from observer.observerServer import observer
        from observer.calibrator import CalibratedObserver
        
        # Reset global app variable in observerServer module if needed
        import observer.observerServer as obs_module
        
        app = Flask(__name__)
        app.cc = mock_cc
        app.register_blueprint(observer, url_prefix='/observer')
        # Configurator removed from observer
        app.cm = CalibratedObserver(app.cc)
        obs_module.app = app
        
        yield app

@pytest.fixture
def configurator_app(mock_cv2):
    """Fixture to create the Configurator app with mocked hardware."""
@pytest.fixture
def configurator_app(mock_cv2):
    """Fixture to create the Configurator app with mocked hardware."""
    # Patch where it is used in configuratorServer
    with mock.patch('observer.configuratorServer.HexCaptureConfiguration') as MockCC:
        mock_cc = MockCC.return_value
        mock_cc.cameras = {"Camera 0": mock.MagicMock()}
        mock_cc.cameras["Camera 0"].mostRecentFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock activeZone to be serializable
        mock_cc.cameras["Camera 0"].activeZone = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        
        # Mock hex configuration
        mock_cc.hex = mock.MagicMock()
        mock_cc.hex.size = 10.0
        
        from observer.configuratorServer import create_configurator_app
        
        app = create_configurator_app()
        # Ensure the app uses our mock (create_configurator_app calls HexCaptureConfiguration() 
        # which returns mock_cc because of the patch)
        
        # Re-assign cm with new cc if needed, but create_configurator_app already does:
        # app.cm = CalibrationObserver(app.cc)
        
        # However, CalibrationObserver might need to be mocked if it does heavy lifting
        # But we let it run with mocked cc for now.
        
        # We need to make sure the global 'app' variables in modules are updated
        from observer.configurator import setConfiguratorApp
        from observer.calibrator import setCalibratorApp
        setConfiguratorApp(app)
        setCalibratorApp(app)
        
        yield app

@pytest.fixture
def configurator_client(configurator_app):
    return configurator_app.test_client()

@pytest.fixture
def harmony_app(mock_cv2):
    """Fixture to create the Harmony app with mocked hardware."""
    with mock.patch('observer.calibrator.CalibratedCaptureConfiguration') as MockCC, \
         mock.patch('observer.HexGridConfiguration') as MockHex, \
         mock.patch('observer.HexCaptureConfiguration'), \
         mock.patch('harmony.harmonyServer.registerCaptureService'), \
         mock.patch('harmony.harmonyServer.HarmonyMachine') as MockHM:
        
        mock_cc = MockCC.return_value
        mock_cc.cameras = {"Camera 0": mock.MagicMock()}
        mock_cc.cameras["Camera 0"].mostRecentFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Configure HarmonyMachine mock
        def harmony_machine_side_effect(cc):
            m = mock.MagicMock()
            m.cc = cc
            m.memory = []
            return m
        MockHM.side_effect = harmony_machine_side_effect

        from harmony.harmonyServer import create_harmony_app
        app = create_harmony_app()
        app.cc = mock_cc
        # Ensure app.cm uses the same mock cc
        if hasattr(app, 'cm'):
            app.cm.cc = mock_cc
        yield app

@pytest.fixture
def client(observer_app):
    return observer_app.test_client()

@pytest.fixture
def harmony_client(harmony_app):
    return harmony_app.test_client()
