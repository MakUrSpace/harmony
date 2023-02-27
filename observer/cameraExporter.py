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


LAST_IMAGE = {}
DELTAS = {}


def changeBetween(im0, im1):
    img_height = im0.shape[0]
    print(img_height)
    diff = cv2.absdiff(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            # Calculate bounding box around contour
            boxes.append(cv2.boundingRect(contour))
    return boxes


def drawBoxesOnImage(image, boxes):
    imageWithBoxes = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(imageWithBoxes, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(imageWithBoxes, f'{x}-{x+w}, {y}-{y+w}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 0, 0), 2, cv2.LINE_AA)
    return imageWithBoxes


def identify_cameras():
    cameraOutputs camera_idx= os.listdir("output")
    cameras = list(set([int(fn[3:4]) for fn in cameraOutputs]))
    return cameras


def getCameraImagePaths(camera_idx):
    return [f"output/{path}"
            for path in os.listdir("output")
            if f"cam{camera_idx}_" in path]


def getCameraImages(camera_idx):
    for image in getCameraImagePaths(camera_idx):
        if f"cam{camera_idx}" in image:
            with open(f"output/{image}", "rb") as f:
                jpg = f.read()
            camImages.append(base64.b64encode(jpg))
    newImage = cv2.imread(f"output/{image}")
    updateCameraState(newImage)
    return camImages


def updateCameraState(camera_idx, newImage):
    diffBoxes = changeBetween(LAST_IMAGE[camera_idx], newImage)
    diffImage = drawBoxesOnImage(newImage, diffBoxes)
    
    deltas= []
    for x,y,w,h in diffBoxes:
        subImage = newImage[y:y+h,x:x+w]
        retval, buffer = cv2.imencode('.jpg', subImage)
        deltas.append(base64.b64encode(buffer))
    DELTAS[camera_idx] = deltas

    cv2.imwrite(f"output/changeTracked_c{camera_idx}.jpg", diffImage)
    LAST_IMAGE[camera_idx] = newImage


def exportLoop():
    while True:
        imagePacket = {}
        for cam in CAMERAS:
            print(f"{datetime.utcnow()}: Capturing Cam {cam}")
            imagePacket[cam] = getCameraImages(cam)
        imagePacket["DELTAS"] = DELTAS
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

