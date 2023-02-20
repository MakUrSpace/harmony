import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64

from flask import Flask, render_template, Response


CAMERAS = []
app = Flask(__name__)


EXPORT_URL = os.getenv("OT_EXPORT_URL", "localhost:7000/observer")
EXPECTED_CAMS = [int(camNum) for camNum in os.getenv("OT_EXPECTED_CAMS", "0").split(",")]
EXPORT_SLEEP = min(float(os.getenv("OT_EXPORT_SLEEP", "10")), 5)


def identify_cameras():
    cameraOutputs = os.listdir("output")
    cameras = [fn.split(".")[0][3:] for fn in cameraOutputs]
    return cameras


def getCameraImage(camera_idx):
        with open(f"output/cam{camera_idx}.jpg", "rb") as f:
            jpg = f.read()
        base64Image = base64.b64encode(jpg)
        return base64Image


def exportLoop():
    while True:
        for cam in CAMERAS:
            print(f"{datetime.utcnow()}: Exporting {cam}")
            getCameraImage(cam)
        print(f"{datetime.utcnow()}: Resting...")
        sleep(EXPORT_SLEEP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    for expectedCam in EXPECTED_CAMS:
        assert expectedCam in CAMERAS
    print(f"Supporting Cameras: {CAMERAS}")
    exportLoop()

