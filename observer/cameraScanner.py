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


MAX_CAM_ID = int(os.getenv("OBS_MAX_CAM_ID", '10'))
CAPTURE_TIME = float(os.getenv("OBS_CAPTURE_TIME", '5'))
CAPTURE_FRAMES = int(os.getenv("OBS_CAPTURE_FRAMES", '5'))
EXPECTED_CAMS = [int(camNum) for camNum in os.getenv("OBS_EXPECTED_CAMS", '0').split(',') if camNum != '']


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


def frameDifferences(image0, image1):
    img_height = image0.shape[0]
    diff = cv2.absdiff(cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            # Calculate bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            if x < centerPoint[0] < x+w and y < centerPoint[1] < y+w:
                # Draw rectangle - bounding box on both images
                # cv2.rectangle(image0, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.rectangle(image1, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(coreImage, f'{x}-{x+w}, {y}-{y+w}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 0), 2, cv2.LINE_AA)
    return image1


def collectFromCamera(camNum, frames=CAPTURE_FRAMES, delay=0.5):
    cap = cv2.VideoCapture(camNum)
    try:
        sleep(delay)
        for f in range(CAPTURE_FRAMES):
            for frame in range(CAPTURE_FRAMES):
                ret, cv2_im = cap.read()
                sleep(delay)
            retval, buff = cv2.imencode('.jpg', cv2_im)
            with open(f"output/cam{camNum}_f{f}_new.jpg", "wb") as f:
                f.write(buff)
            os.rename(f"output/cam{camNum}_f{f}_new.jpg", f"output/cam{camNum}_f{frame}.jpg")
            sleep(delay)
    finally:
        cap.release()


def collectFromCameras():
    global CAMERA_FRAMES
    lastFrames = {camNum: None for camNum in CAMERAS}
    while True:
        for camNum in CAMERAS:
            collectFromCamera(camNum, CAPTURE_FRAMES) 
        sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    for eC in EXPECTED_CAMS:
        assert eC in CAMERAS, f"Expected to find Camera: {eC} in {CAMERAS}"
    print(f"Supporting Cameras: {CAMERAS}")
    collectFromCameras()
