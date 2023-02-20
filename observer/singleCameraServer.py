import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64

from flask import Flask, render_template, Response


CAMERA = None
app = Flask(__name__)


def capture_camera(cam_num):
    cam = cv2.VideoCapture(cam_num)
    retval, image = cam.read()
    cam.release()
    retval, buff = cv2.imencode('.jpg', image)
    b64jpg = base64.b64encode(buff)
    return b64jpg


def identify_cameras(device_numbers=list(range(8))):
    functional = []
    for dn in device_numbers:
        try:
            img = capture_camera(dn)
            functional.append(dn)
        except Exception:
            continue
    return functional


def gen_frames():
    cap = cv2.VideoCapture(CAMERA)
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if not ret:
            break

        retval, buff = cv2.imencode('.jpg', cv2_im)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


@app.route('/')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = None)
    args = parser.parse_args()
    CAMERA = args.camera_idx
    if CAMERA is None:
        print(identify_cameras())
    else:
        port = int(f"500{1 + CAMERA}")
        app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)

