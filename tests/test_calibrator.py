
import pytest
import unittest.mock as mock
from observer.calibrator import CalibratedObserver, CalibratedCaptureConfiguration

def test_calibrator_init(mock_cv2):
    """Test initialization of CalibratedObserver."""
    with mock.patch('observer.calibrator.CalibratedCaptureConfiguration') as MockCC:
        mock_cc = MockCC.return_value
        mock_cc.cameras = {} 
        
        observer = CalibratedObserver(mock_cc)
        assert observer.cc == mock_cc
        assert observer.mode == "passive"
        assert len(observer.memory) == 0

def test_calibrator_mode_switch(mock_cv2):
    """Test switching between passive and track modes."""
    with mock.patch('observer.calibrator.CalibratedCaptureConfiguration') as MockCC:
        observer = CalibratedObserver(MockCC.return_value)
        
        observer.trackMode()
        assert observer.mode == "track"
        
        observer.passiveMode()
        assert observer.mode == "passive"
