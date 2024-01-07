import os
import sys

oldPath = os.getcwd()
try:
    observerDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, observerDirectory)
    from ipynb.fs.full.CalibrationMachine import CaptureConfiguration, CalibrationMachine, CalibratedMachine, TrackedObject
finally:
    os.chdir(oldPath)