import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64
from time import sleep
import requests as req

from flask import Flask, render_template, Response


CAMERAS = []
app = Flask(__name__)


EXPORT_URL = os.getenv("OBS_EXPORT_URL", "http://localhost:7000/observer/foobar")
EXPECTED_CAMS = os.getenv("OBS_EXPECTED_CAMS", "0").split(",")
EXPORT_SLEEP = min(float(os.getenv("OBS_EXPORT_SLEEP", "10")), 5)


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
        imagePacket = {}
        for cam in CAMERAS:
            print(f"{datetime.utcnow()}: Exporting {cam}")
            imagePacket[cam] = getCameraImage(cam)
        resp = req.post(EXPORT_URL, json=imagePacket)
        assert resp.status_code == 200, f"Failed to export images: {resp.text}"
        print(f"{datetime.utcnow()}: Resting...")
        sleep(EXPORT_SLEEP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    for expectedCam in EXPECTED_CAMS:
        assert expectedCam in CAMERAS, f"Expected {expectedCam} in {CAMERAS}"
    print(f"Supporting Cameras: {CAMERAS}")
    exportLoop()

