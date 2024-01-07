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

from ipynb.fs.full.CalibratedObserver import CalibratedCaptureConfiguration, CalibrationObserver

import threading
import atexit
from flask import Blueprint, render_template, Response, request, make_response, redirect
from traceback import format_exc


CONSOLE_OUTPUT = "No Output Yet"

POOL_TIME = 0.1 #Seconds
PORT = int(os.getenv("OBSERVER_PORT", "7000"))

ENABLE_CYCLE = True

data_lock = threading.Lock()

captureTimer = threading.Timer(0,lambda x: None,())    
def registerCaptureService(app):
    def interrupt():
        global captureTimer
        captureTimer.cancel()

    def cycleMachine():
        global CONSOLE_OUTPUT
        with data_lock:
            if ENABLE_CYCLE:
                try:
                    cm.cycle()
                    CONSOLE_OUTPUT = ""
                except Exception as e:
                    print(f"Unrecognized error: {e}")
                    CONSOLE_OUTPUT = e
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


observerApp = Blueprint('config', __name__, template_folder='templates')


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(cc.cameras.values())[0]
        shape = (400, 400)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")

        consoleImage = cv2.putText(zeros, f'Cycle {cm.cycleCounter}',
            (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'LO: {CONSOLE_OUTPUT}',
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Mode: {cm.mode:7} {"(" + cm.actionState + ")" if cm.mode == "action" else ""}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {cm.state:10}',
            (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        
        consoleImage = cv2.putText(zeros, f'Last Change',
            (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        if cm.lastMemory is not None:
            if 'newObject' in cm.lastMemory:
                obj = cm.lastMemory['newObject']
                currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(obj)]
                objectIdentifier = f'New Object Added'
                objectLocation = f'at {currentLocation}'
                consoleImage = cv2.putText(zeros, objectIdentifier,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, objectLocation,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            elif 'changedObject' in cm.lastMemory:
                obj = cm.lastMemory['changedObject']
                currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(obj)]
                objectIdentifier = f'{obj.objectType} {obj.name}'
                objectLocation = f'at {currentLocation}'
                consoleImage = cv2.putText(zeros, objectIdentifier,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, objectLocation,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                lastLocation = [f"{d:7.2f}" for d in cc.rsc.changeSetToRealCenter(obj.previousVersion())]
                distanceMoved = cc.rsc.trackedObjectLastDistance(obj)
                consoleImage = cv2.putText(zeros, f'Moved {distanceMoved:6.2f} mm',
                    (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, f'From {lastLocation}',
                    (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            elif 'annotatedObject' in cm.lastMemory:
                obj = cm.lastMemory['annotatedObject']
                currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(obj)]
                objectIdentifier = f'{obj.name} is {obj.objectType} '
                objectLocation = f'at {currentLocation}'
                consoleImage = cv2.putText(zeros, objectIdentifier,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, objectLocation,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
   
        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@observerApp.route('/observer_console', methods=['GET'])
def getConsoleImage():
    return Response(renderConsole(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCamerasView():
    while True:
        camImages = []
        for camName in cc.cameras.keys():
            camImage = cc.cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraWithChangesView(camName):
    camName = str(camName)
    cam = cc.cameras[camName]
    while True:
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        camImage = cam.mostRecentFrame.copy()
        # Paint known objects blue
        for memObj in cm.memory:
            if memObj.changeSet[camName].changeType not in ['delete', None]:
                memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
        # Paint last changes red
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            lastChange = cm.lastChanges.changeSet[camName]
            if lastChange is not None and lastChange.changeType not in ['delete', None]:
                lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
        # Paint classification green
        if cm.lastClassification is not None and not cm.lastClassification.empty:
            lastClass = cm.lastClassification.changeSet[camName]
            if lastClass is not None and lastClass.changeType not in ['delete', None]:
                lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/camWithChanges/<camName>')
def cameraViewWithChangesResponse(camName):
    return Response(genCameraWithChangesView(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def fullCam(camName):
    camName = str(camName)
    cam = cc.cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/fullCam/<camName>')
def genFullCam(camName):
    return Response(fullCam(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        for camName in cc.cameras.keys():
            camImage = cc.cameras[camName].mostRecentFrame.copy()
            # Paint known objects blue
            for memObj in cm.memory:
                if memObj.changeSet[camName].changeType not in ['delete', None]:
                    memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
            # Paint last changes red
            if cm.lastChanges is not None and not cm.lastChanges.empty:
                lastChange = cm.lastChanges.changeSet[camName]
                if lastChange is not None and lastChange.changeType not in ['delete', None]:
                    lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
            # Paint classification green
            if cm.lastClassification is not None and not cm.lastClassification.empty:
                lastClass = cm.lastClassification.changeSet[camName]
                if lastClass is not None and lastClass.changeType not in ['delete', None]:
                    lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/set_passive')
def controlSetPassive():
    with data_lock:
        cm.passiveMode()
    return "success"
        

@observerApp.route('/set_track')
def controlSetTrack():
    with data_lock:
        cm.trackMode()
    return "success"


@observerApp.route('/')
def buildController():
    with open("templates/Controller.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in cc.cameras.keys()])
    defaultCam = [camName for camName, cam in cc.cameras.items()][0]
    return template.replace("{defaultCamera}", defaultCam).replace("{cameraButtons}", cameraButtons)


def captureToChangeRow(capture):
    encodedBA = imageToBase64(capture.visual())
    name = capture.name if type(capture) == QuantumObject else "None"
    objType = capture.objectType if type(capture) == QuantumObject else "None"
    center = ", ".join([f"{pt:.2f}" for pt in cc.rsc.changeSetToRealCenter(capture)])
    health = ""
    if type(capture) == QuantumObject:
        numHits = sum([True for act in capture.actionHistory.values() if act[0] == 'hit'])
    else:
        numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "
    changeRow = f"""
        <div class="row mb-1">
            <div class="col">
                <div class="row">
                    <div class="col">
                        <p>{objType}</p>
                    </div>
                    <div class="col">
                        <p>{name}</p>
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <p>({center})</p>
                    </div>
                    <div class="col">
                        <p>{health}</p>
                    </div>
                </div>
                <div class="row">
                    <button class="btn btn-primary" onclick="window.location.href='/control/objectsettings/{capture.oid}'">Edit</button>
                </div>
            </div>
            <div class="col">
                <img class="img-fluid border border-secondary" alt="Capture Image" src="data:image/jpg;base64,{encodedBA}" style="border-radius: 10px;">
            </div>
        </div>
        <hr class="mt-2 mb-3"/>"""
    return changeRow


def buildObjectTable():
    changeRows = []
    print(f"Aware of {len(cm.memory)} objects")
    for capture in cm.memory:
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@observerApp.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


@observerApp.route('/objectsettings/<objectId>', methods=['GET'])
def getObjectInfo(objectId):
    capture = None
    for c in cm.memory:
        if c.oid == objectId:
            capture = c

    with open("templates/ChangeTable.html", "r") as f:
        template = f.read()

    if capture is None:
        return template.replace("{changeTableBody}", f"Object {objectId} Not Found")
    
    objOptions = "\n".join([f'<option value="{objType}">' for objType in mc.ObjectFactories.keys()])
    encodedBA = imageToBase64(capture.visual())
    center = ", ".join([f"{pt:.2f}" for pt in cc.rsc.changeSetToRealCenter(capture)])
    health = ""
    if type(capture) == QuantumObject:
        numHits = sum([True for act in capture.actionHistory.values() if act[0] == 'hit'])
        name = capture.name
        objectType = capture.objectType
    else:
        numHits = 0
        name = ""
        objectType = ""
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "
    body =  f"""
    <div class="col">
        <button class="btn btn-secondary" hx-delete="/control/objectsettings/{objectId}" hx-swap="outerHTML">Delete</button>
        <form hx-post="/control/objectsettings/{objectId}">
            <div class="row">
              <datalist id="objectTypeDataList">
                {objOptions}
              </datalist>
              <div class="form-group col">
                <label for="objectType">Object Type</label>
                <input list="objectTypeDataList" class="form-control" name="objectType" value="{objectType}">
              </div>
              <div class="form-group col">
                <label for="objectName">Object Name</label>
                <input type="text" class="form-control" name="objectName" placeholder="{name}">
              </div>
            </div>
            <div class="row">
                <div class="col">
                    <p>({center})</p>
                </div>
                <div class="col">
                    <p>{health}</p>
                </div>
            </div>
            <div class="row">
                <button class="btn btn-primary">Update</button>
            </div>
        </form>
    </div>
    <div class="col">
        <img class="img-fluid border border-secondary" alt="Capture Image" src="data:image/jpg;base64,{encodedBA}" style="border-radius: 10px;">
    </div>"""
    template = template.replace("{changeTableBody}", body)
    return template
    
    
@observerApp.route('/objectsettings/<objectId>', methods=['POST'])
def postObjectSettings(objectId):
    objName = request.form.get("objectName")
    objType = request.form.get("objectType")
    cm.annotateObject(objectId, objName, objType)
    return redirect("/control", code=303)
    
    
@observerApp.route('/objectsettings/<objectId>', methods=['DELETE'])
def deleteObjectSettings(objectId):
    cm.deleteObject(objectId)
    return redirect("/control", code=303)


if __name__ == "__main__":
    from flask import Flask
    cc = CaptureConfiguration()
    cc.capture()
    cm = Observer(cc)
    cm.cycle()
    app = Flask(__name__)
    app.register_blueprint(observerApp, url_prefix='/control')

    @app.route('/<page>')
    def getPage(page):
        try:
            with open(f"templates/{page}") as page:
                page = page.read()
        except Exception as e:
            print(f"Failed to find page: {e}")
            page = "Not found!"
        return page

    registerCaptureService(app)
    print(f"Launching Observer Server on {PORT}")
    app.run(debug=True, host="0.0.0.0", port=PORT)
