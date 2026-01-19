
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

@pytest.fixture
def observer_app(mock_cv2):
    """Fixture to create the Observer app with mocked hardware."""
    with mock.patch('observer.calibrator.CalibratedCaptureConfiguration') as MockCC:
        mock_cc = MockCC.return_value
        mock_cc.cameras = {"Camera 0": mock.MagicMock()}
        mock_cc.cameras["Camera 0"].mostRecentFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cc.cameras["Camera 0"].cropToActiveZone.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        from flask import Flask
        from observer.observerServer import observer, configurator, setConfiguratorApp
        from observer.calibrator import CalibratedObserver
        
        # Reset global app variable in observerServer module if needed
        import observer.observerServer as obs_module
        
        app = Flask(__name__)
        app.cc = mock_cc
        app.register_blueprint(observer, url_prefix='/observer')
        app.register_blueprint(configurator, url_prefix='/configurator')
        app.cm = CalibratedObserver(app.cc)
        setConfiguratorApp(app)
        obs_module.app = app
        
        yield app

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
