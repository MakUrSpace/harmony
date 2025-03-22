import os
import sys

from importnb import Notebook

oldPath = os.getcwd()
try:
    observerDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, observerDirectory)
    with Notebook():
        from CalibratedObserver import (
            CalibratedCaptureConfiguration,
            CalibrationObserver,
            CalibratedObserver,
            CameraChange,
            TrackedObject,
            MiniMapObject,
            Transition,
            distanceFormula
        )
finally:
    os.chdir(oldPath)