import os
import sys

oldPath = os.getcwd()
try:
    observerDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, observerDirectory)
    from observer.CalibratedObserver import (
        CalibratedCaptureConfiguration,
        CalibrationObserver,
        CalibratedObserver,
        CameraChange,
        TrackedObject,
        MiniMapObject,
        Transition,
        distanceFormula
    )
    from observer.HexObserver import (
        HexGridConfiguration,
        HexCaptureConfiguration
    )
finally:
    os.chdir(oldPath)