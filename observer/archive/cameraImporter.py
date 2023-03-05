import argparse
import cv2
import numpy as np
import os
import imutils
import json
from random import choice
from datetime import datetime, timedelta
import base64

from flask import Flask, render_template, Response, request


CAMERAS = []
app = Flask(__name__)
PORT = int(os.getenv("OBS_IMPORT_PORT", "7000"))


def identify_cameras():
    cameraOutputs = os.listdir("output")
    cameras = [fn.split(".")[0][3:] for fn in cameraOutputs]
    return cameras


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


def trackChanges(observer_name, camera_idx, newImage):
    lastImage = LAST_IMAGE[camera_idx]
    diffBoxes = changeBetween(lastImage, newImage)
    print(f"Found {len(diffBoxes)} changes")
    diffImage = drawBoxesOnImage(newImage, diffBoxes)
    os.makedirs(f"imported/{observer_name}/changeTracking", exist_ok=True)
    cv2.imwrite(
        f"imported/{observer_name}/changeTracking/ct_cam{camera_idx}_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}.jpg",
        diffImage)
    deltas = [newImage[y:y+h,x:x+w] for x,y,w,h in diffBoxes]
    deltaPath = f"imported/{observer_name}/changeTracking/deltas_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}"
    os.makedirs(deltaPath, exist_ok=True)
    for idx, d in enumerate(deltas):
        cv2.imwrite(f"{deltaPath}/delta{idx}_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}.jpg", d)


@app.route('/observer/<observer_name>', methods=['POST'])
def receive_images(observer_name):
    imported_images = request.get_json()
    for cam, frames in imported_images.items():
        os.makedirs(f"imported/{observer_name}", exist_ok=True)
        for idx, image in enumerate(frames):
            imageBin = base64.b64decode(image)
            with open(f"imported/{observer_name}/{observer_name}_cam{cam}_frame{idx}_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}.jpg", "wb") as f:
               f.write(imageBin)
        newImage = cv2.imdecode(np.frombuffer(imageBin, np.uint8), cv2.IMREAD_ANYCOLOR)
        if cam not in LAST_IMAGE:
            LAST_IMAGE[cam] = newImage
        else:
            trackChanges(observer_name, cam, newImage)
    return "Success!!"


@app.route('/camera_list')
def camera_list():
    return json.dumps(CAMERAS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=PORT)

