import os
import cv2
from math import ceil
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import argparse
import json
from io import BytesIO
from ipynb.fs.full.Observer import cm, cameras, pov, Camera, hStackImages, vStackImages

import threading
import atexit
from flask import Flask, render_template, Response, request, make_response
from traceback import format_exc
from ultralytics import YOLO

CONSOLE_OUTPUT = "No Output Yet"


humanInferenceModel = YOLO("yolov8n.pt")


POOL_TIME = 0.1 #Seconds
PORT = int(os.getenv("OBSERVER_PORT", "7000"))

ENABLE_CYCLE = True

data_lock = threading.Lock()

captureTimer = threading.Timer(0,lambda x: None,())    
def createCaptureApp():
    app = Flask(__name__)

    def interrupt():
        global captureTimer
        captureTimer.cancel()

    def cycleMachine():
        with data_lock:
            if ENABLE_CYCLE:
                cm.cycle()
        # Set the next timeout to happen
        captureTimer = threading.Timer(POOL_TIME, cycleMachine, ())
        captureTimer.start()   

    def initialize():
        global captureTimer
        captureTimer = threading.Timer(POOL_TIME, cycleMachine, ())
        captureTimer.start()

    initialize()
    # When you kill Flask (SIGTERM), cancels the timer
    atexit.register(interrupt)
    return app


observerApp = createCaptureApp()


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

def renderConsole():
    while True:
        cam = list(cameras.values())[0]
        shape = (400, 400)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")
        consoleImage = cv2.putText(zeros, f'CapMac--{cm.state}',
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Mode: {cm.mode}',
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'ID: {cm.interactionDetected} || DC: {cm.debounceCounter}',
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)     
        ret, consoleImage = cv2.imencode('.jpg', consoleImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@observerApp.route('/console', methods=['GET'])
def getConsoleImage():
    return Response(renderConsole(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/bootstrap.min.css', methods=['GET'])
def getBSCSS():
    with open("templates/bootstrap.min.css", "r") as f:
        bscss = f.read()
    return Response(bscss, mimetype="text/css")


@observerApp.route('/bootstrap.min.js', methods=['GET'])
def getBSJS():
    with open("templates/bootstrap.min.js", "r") as f:
        bsjs = f.read()
    return Response(bsjs, mimetype="application/javascript")


def genCameraFullViewWithActiveZone(camNum):
    while True:
        cam = cameras[int(camNum)]
        camImage = cam.drawActiveZone(cam.mostRecentFrame)
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(camImage)
        camFrame = BytesIO()
        FigureCanvas(fig).print_png(camFrame)
        camFrame.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camFrame.read() + b'\r\n')
    
    
@observerApp.route('/config/camera/<camNum>')
def cameraActiveZoneWithObjects(camNum):
    return Response(genCameraFullViewWithActiveZone(int(camNum)), mimetype='multipart/x-mixed-replace; boundary=frame')
    return response


def buildConfigurator():
    cameraConfigRows = []
    clickSubs = []
    for cam in cameras.values():
        activeZone = json.dumps(cam.activeZone.tolist())
        currentCalibration = f"Calibrated" if cam.M is not None else "Not Calibrated"
        cameraConfigRows.append(f"""
            <div class="row justify-content-center">
                <h3 class="mt-5">Camera {cam.camNum}</h3>
                <div class="container">
                    <img src="/config/camera/{cam.camNum}" title="{cam.camNum} Capture" width="100%" height="500" id="cam{cam.camNum}" onclick="cam{cam.camNum}ClickListener(event)"></iframe>
                </div>
                <form method="post">
                  <label>{currentCalibration}</label><br>
                  <label for="az">Active Zone</label><br>
                  <input type="text" name="az" id="cam{cam.camNum}_ActiveZone" value="{activeZone}" size="50"><br>
                  <input type="hidden" id="camNum" name="camNum" value="{cam.camNum}">
                  <input type="submit" value="Update Cam {cam.camNum}">
                  <input type="hidden" id="configType" name="configType" value="activeZone">
                </form>
                <br>
            </div>""")
        clickSubs.append(f"""
            function cam{cam.camNum}ClickListener(event) {{
                const imgElem = document.getElementById("cam{cam.camNum}")
                bounds=imgElem.getBoundingClientRect();
                const left=bounds.left;
                const top=bounds.top;
                const x = event.x - left;
                const y = event.y - top;
                const cw=imgElem.clientWidth
                const ch=imgElem.clientHeight
                const iw=imgElem.naturalWidth
                const ih=imgElem.naturalHeight
                const px=x/cw*iw
                const py=y/ch*ih
                console.log(px, px)
                const x_offset = 80
                const x_scale = 1800 / 480
                const image_x = (px - x_offset) * x_scale
                const y_offset = 100
                const y_scale = 1080 / 280
                const image_y = (py - y_offset) * y_scale
                const formField = document.getElementById("cam{cam.camNum}_ActiveZone")
                var formValue = JSON.parse(formField.value)
                formValue.push([~~image_x, ~~image_y])
                formField.value = JSON.stringify(formValue)
            }}""")
    unwarpedImage = imageToBase64(cm.cc.unwarpedOverlaidCameras())
    unwarpedImage = f'<img src="data:image/jpeg;base64,{unwarpedImage}">'
    with open("templates/Configuration.html") as f:
        template = f.read()
    return template.replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows)
    ).replace(
        "{clickSubscriptions}", "\n".join(clickSubs)
    ).replace(
        "{unwarpedImage}", unwarpedImage)

    
@observerApp.route('/config', methods=['GET'])
def config():
    cm.cc.capture()
    return buildConfigurator()
    
    
@observerApp.route('/config', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = ""
    print(f"Received update config request")
    
    if (config_type := request.form.get("config_type")) == 'save_state':
        cm.cc.saveConfiguration()
        CONSOLE_OUTPUT = "State Saved!"
        print("State Saved!")
    elif config_type == 'load_state':
        cm.cc.recoverConfiguration()
        CONSOLE_OUTPUT = "State Recovered!"
        print("State Recovered!")
    elif config_type == 'calibrate_cameras':
        cm.calibrate()
        CONSOLE_OUTPUT = "Cameras calibrated!"
        print("Cameras Calibrated")
    else:
        camNum = int(request.form.get("camNum"))
        if (configType := request.form.get("configType")) == "activeZone":
            az = np.float32(json.loads(request.form.get(f"az")))

            cam = cameras[camNum]
            cam.setActiveZone(az)
        elif configType == "calibrate":
            xPosition = float(request.form.get('calibrate_x'))
            yPosition = float(request.form.get('calibrate_y'))
            try:
                qrCodes = cameras[camNum].calibrate((xPosition, yPosition))
                CONSOLE_OUTPUT = f"Camera {camNum} Calibrated to ({xPosition}, {yPosition}). Saw {qrCodes} calibration boxes"
            except Exception as e:
                print(f"Failed Calibration: {e}")
                CONSOLE_OUTPUT = f"Failed Calibration: {e}"

    print(f"Rebuilding configurator")
    return buildConfigurator()


def genCombinedCamerasView():
    while True:
        camImages = []
        for camNum in cameras.keys():
            camImage = cameras[camNum].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        print(f"Num memories: {len(cm.memory)}")
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        for camNum in cameras.keys():
            camImage = cameras[camNum].mostRecentFrame.copy()
            # Paint known objects blue
            for memObj in cm.memory:
                if memObj.changeSet[camNum].changeType not in ['delete', None]:
                    memContour = np.array([memObj.changeSet[camNum].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
            # Paint last changes red
            if cm.lastChanges is not None and not cm.lastChanges.empty:
                lastChange = cm.lastChanges.changeSet[camNum]
                if lastChange is not None and lastChange.changeType not in ['delete', None]:
                    lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
            # Paint classification green
            if cm.lastClassification is not None and not cm.lastClassification.empty:
                lastClass = cm.lastClassification.changeSet[camNum]
                if lastClass is not None and lastClass.changeType not in ['delete', None]:
                    lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def minimapGenerator():
    while True:
        camImage = cameras[0].mostRecentFrame
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
    

@observerApp.route('/control/minimap')
def minimapResponse():
    return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/control')
def buildController():
    with open("templates/Controller.html", "r") as f:
        template = f.read()

    cameraCaptures = """    <div class="container" width="100%">
        <div class="row" width="100%">
            <h3 class="mt-5">Virtual Map</h3>
            <img src="/control/minimap" height=200px>
        </div>
        <div class="row" width="100%">
            <h3 class="mt-5">Live Cameras</h3>
            <img src="/control/combinedCamerasWithChanges" height=500px>
        </div>
    </div>"""
    template = template.replace("{cameraCaptures}", cameraCaptures)
    return template


def buildObjectTable():
    changeRows = []
    print(f"Aware of {len(cm.memory)} objects")
    for cid, capture in enumerate(cm.memory[::-1]):
        encodedBA = imageToBase64(capture.visual())
        changeRow = f"""
            <div class="row">
                <div class="col">
                    <form method=post">
                      <label for="object_name">Object Name</label><br>
                      <input type="text" name="object_name" id="object_name" value="{cid}" size="50"><br>
                      <input type="hidden" id="orig_name" name="orig_name" value="{cid}">
                      <input type="submit" value="Update {cid} Name">
                    </form>
                </div>
                <div class="col">
                    <p>{capture.realCenter}</p>
                </div> 
                <div class="col">
                   <div class="row" name="captureVisual">
                     <img alt="{cid} Capture" src="data:image/jpg;base64,{encodedBA}">
                   </div>
                </div>
                <div class="col">
                    <form method="post">
                      <input type="submit" value="Delete {cid}">
                    </form>
                </div>
            </div>"""
        changeRows.append(changeRow)
    
    topRow = """
        <div class="container justify-content-center">
            <div class="row">
                <div class="col">
                    <h3>Object Name</h3>
                </div>
                <div class="col">
                    <h3>Object Coordinates</h3>
                </div>
                <div class="col">
                    <h3>Object Visual</h3>
                </div>
                <div class="col">
                    <h3>Delete Object</h3>
                </div>
            </div>
            {changeRows}
        </div>""".format(
        changeRows="".join(changeRows)
    )
    with open("templates/ChangeTable.html", "r") as f:
        template = f.read()
    return template.replace("{changeTableBody}", topRow)


@observerApp.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


if __name__ == "__main__":
    cm.cycle()
    print(f"Launching Observer Server on {PORT}")
    observerApp.run(host="0.0.0.0", port=PORT)
