import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64
import multiprocessing
from time import sleep


from flask import Flask, render_template, Response


CAMERAS = []
CAMERA_FRAMES = {}
app = Flask(__name__)


MAX_CAM_ID = int(os.getenv("OBS_MAX_CAM_ID", '6'))
CAPTURE_TIME = float(os.getenv("OBS_CAPTURE_TIME", '5'))
EXPECTED_CAMS = [int(camNum) for camNum in os.getenv("OBS_EXPECTED_CAMS", '0').split(',')]


def capture_camera(cam_num):
    try:
        cam = cv2.VideoCapture(cam_num)
        retval, image = cam.read()
    finally:
        cam.release()
    retval, buff = cv2.imencode('.jpg', image)
    b64jpg = base64.b64encode(buff)
    return b64jpg


def identify_cameras(device_numbers=list(range(MAX_CAM_ID))):
    functional = []
    for dn in device_numbers:
        try:
            img = capture_camera(dn)
            functional.append(dn)
        except Exception as e:
            continue
    return functional


def collectFromCameras():
    global CAMERA_FRAMES
    while True:
        for camNum in CAMERAS:
            print(f"Capturing {camNum}")
            start = datetime.utcnow()
            try:
                cap = cv2.VideoCapture(camNum)
                print("capture created")
                while (datetime.utcnow() - start).total_seconds() < CAPTURE_TIME:
                    ret, cv2_im = cap.read()

                    retval, buff = cv2.imencode('.jpg', cv2_im)
                    with open(f"output/cam{camNum}_new.jpg", "wb") as f:
                        f.write(buff)
                    os.rename(f"output/cam{camNum}_new.jpg", f"output/cam{camNum}.jpg")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    sleep(0.05)
            except Exception as e:
                print(f"Failed to capture {camNum}: {e}")
            finally:
                cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    for eC in EXPECTED_CAMS:
        assert eC in CAMERAS, f"Expected to find Camera: {ec}"
    print(f"Supporting Cameras: {CAMERAS}")
    collectFromCameras()

