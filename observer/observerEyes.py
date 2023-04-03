import json
import cv2
import base64
from flask import Flask, make_response, request, Response
import asyncio
import argparse
from time import sleep
from traceback import format_exc


eyesApp = Flask(__name__)
CAM_NUM = None
CAM_SETTINGS = {}


sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN_x4.pb")
sr.setModel("fsrcnn", 4)


def capture_camera():
    upsample = False
    sampleSize = 5
    try:
        cam = cv2.VideoCapture(CAM_NUM)
        for setName, setValue in CAM_SETTINGS.items():
            if setName == 'CAP_PROP_UPSAMPLE':
                upsample = True
            elif setName == 'CAP_PROP_SAMPLE_SIZE':
                sampleSize = int(setValue)
            else:
                if "WIDTH" in setName or "HEIGHT" in setName:
                    setValue = int(setValue)
                else:
                    setValue = float(setValue)
                print(f"Setting {setName} to {setValue}")
                cam.set(getattr(cv2, setName), setValue)

        for i in range(5):
            retval, image = cam.read()
            sleep(0.001)
        assert retval is True
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    finally:
        cam.release()
    if upsample:
        image = sr.upsample(image)
    retval, buff = cv2.imencode('.jpg', image)
    return buff


def setSettings(newCamSettings):
    global CAM_SETTINGS
    for key in newCamSettings:
        assert key in [
            "CAP_PROP_UPSAMPLE", "CAP_PROP_SAMPLE_SIZE",
            "CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT", "CAP_PROP_BRIGHTNESS", "CAP_PROP_CONTRAST",
            "CAP_PROP_SATURATION", "CAP_PROP_EXPOSURE", "CAP_PROP_AUTO_EXPOSURE", "CAP_PROP_FPS"]
    CAM_SETTINGS = newCamSettings


@eyesApp.route('/capture', methods=['GET'])
def capture():
    setSettings(request.args.to_dict())
    try:
        image = capture_camera()
    except:
        print(format_exc())
        return ("UH-OH", 500, {})
    image_bytes = image.tobytes()
    return image_bytes, {'content-type': 'image/jpg'}


@eyesApp.route('/stream', methods=['GET'])
def stream():
    def gen_frames():
        while True:
            image = capture_camera()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ObserverEyes',
        description='Hosts a single USB webcam')
    parser.add_argument('-c', '--cam')
    args = parser.parse_args()
    CAM_NUM = int(args.cam)
    port = f"720{CAM_NUM}"
    eyesApp.run(host="0.0.0.0", port=port)
