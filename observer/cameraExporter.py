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
EXPORT_SLEEP = max(float(os.getenv("OBS_EXPORT_SLEEP", "30")), 5)


def identify_cameras():
    global LAST_IMAGE, DELTAS
    cameraOutputs = os.listdir("output")
    cameras = list(set([int(fn[3:4]) for fn in cameraOutputs if "changeTracked" not in fn]))
    LAST_IMAGE = {c:None for c in cameras}
    DELTAS = {c:[] for c in cameras}
    return cameras


def getCameraImagePaths(camera_idx):
    return [f"output/{path}"
            for path in os.listdir("output")
            if f"cam{camera_idx}_" in path]


def getCameraImages(camera_idx):
    camImages = []
    for image in getCameraImagePaths(camera_idx):
        if f"cam{camera_idx}" in image:
            with open(image, "rb") as f:
                jpg = f.read()
            camImages.append(base64.b64encode(jpg))
    return camImages


def exportLoop():
    while True:
        imagePacket = {}
        for cam in CAMERAS:
            print(f"{datetime.utcnow()}: Capturing Cam {cam}")
            imagePacket[cam] = getCameraImages(cam)
        print(f"{datetime.utcnow()}: Sending Image Packet...")
        resp = req.post(EXPORT_URL, json=imagePacket)
        assert resp.status_code == 200, f"Failed to export images: {resp.text}"
        print(f"{datetime.utcnow()}: Resting...")
        sleep(EXPORT_SLEEP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    for expectedCam in EXPECTED_CAMS:
        assert int(expectedCam) in CAMERAS, f"Expected {expectedCam} in {CAMERAS}"
    print(f"Supporting Cameras: {CAMERAS}")
    exportLoop()

