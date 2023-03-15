import json
import cv2
import base64
from flask import Flask, make_response
import asyncio
import argparse
from time import sleep

eyesApp = Flask(__name__)
CAM_NUM = None


sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN_x4.pb")
sr.setModel("fsrcnn", 4);


def capture_camera():
    try:
        cam = cv2.VideoCapture(CAM_NUM)
        retval, image = cam.read()
        assert retval is True
    finally:
        cam.release()
    image = sr.upsample(image)
    retval, buff = cv2.imencode('.jpg', image)
    return buff


@eyesApp.route('/capture', methods=['GET'])
def trackedObjects():
    try:
        image = capture_camera()
    except:
        return ("UH-OH", 500, {})
    image_bytes = image.tobytes()
    return image_bytes, {'content-type': 'image/jpg'}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ObserverEyes',
        description='Hosts a single USB webcam')
    parser.add_argument('-c', '--cam')
    args = parser.parse_args()
    CAM_NUM = int(args.cam)
    port = f"720{CAM_NUM}"
    eyesApp.run(host="0.0.0.0", port=port)
