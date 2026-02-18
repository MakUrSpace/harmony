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

import threading
import atexit
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for
from traceback import format_exc

# from configurator import configurator, setConfiguratorApp # Moved to configuratorServer.py
from calibrator import calibrator, CalibratedObserver, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, setCalibratorApp
from file_lock import FileLock

app = None

CONSOLE_OUTPUT = "No Output Yet"
POOL_TIME = 0.1 #Seconds
PORT = int(os.getenv("OBSERVER_PORT", "7000"))


observer = Blueprint('observer', __name__, template_folder='templates')


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(app.cc.cameras.values())[0]
        shape = (170, 400)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")

        consoleImage = cv2.putText(zeros, f'Cycle {app.cm.cycleCounter}',
            (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        mode = f'Mode: {app.cm.mode:7}'
        # If it's CalibrationObserver, it might have dowel_position
        if hasattr(app.cm, 'dowel_position') and app.cm.dowel_position:
             mode += f'-{app.cm.dowel_position}'
        
        consoleImage = cv2.putText(zeros, mode,
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {app.cm.state:10}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'LO: {CONSOLE_OUTPUT}',
            (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@observer.route('/observer_console', methods=['GET'])
def getConsoleImage():
    return Response(renderConsole(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCamerasView():
    while True:
        camImages = []
        for camName in app.cc.cameras.keys():
            camImage = app.cc.cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observer.route('/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraWithChangesView(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            print("Has changes")
        if app.cm.lastClassification is not None:
            print("Has class")
        camImage = cam.cropToActiveZone(cam.mostRecentFrame.copy())
        # Paint known objects blue
        for memObj in app.cm.memory:
            if memObj.changeSet[camName].changeType not in ['delete', None]:
                memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
        # Paint last changes red
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            lastChange = app.cm.lastChanges.changeSet[camName]
            if lastChange is not None and lastChange.changeType not in ['delete', None]:
                lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
        # Paint classification green
        if app.cm.lastClassification is not None and not app.cm.lastClassification.empty:
            lastClass = app.cm.lastClassification.changeSet[camName]
            if lastClass is not None and lastClass.changeType not in ['delete', None]:
                lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_LINEAR)
        ret, camImage = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observer.route('/camWithChanges/<camName>')
def cameraViewWithChangesResponse(camName):
    if camName == "VirtualMap":
        return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(genCameraWithChangesView(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def fullCam(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observer.route('/fullCam/<camName>')
def genFullCam(camName):
    return Response(fullCam(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            print("Has changes")
        if app.cm.lastClassification is not None:
            print("Has class")
        for camName in app.cc.cameras.keys():
            camImage = app.cc.cameras[camName].mostRecentFrame.copy()
            # Paint known objects blue
            for memObj in app.cm.memory:
                if memObj.changeSet[camName].changeType not in ['delete', None]:
                    memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
            # Paint last changes red
            if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
                lastChange = app.cm.lastChanges.changeSet[camName]
                if lastChange is not None and lastChange.changeType not in ['delete', None]:
                    lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
            # Paint classification green
            if app.cm.lastClassification is not None and not app.cm.lastClassification.empty:
                lastClass = app.cm.lastClassification.changeSet[camName]
                if lastClass is not None and lastClass.changeType not in ['delete', None]:
                    lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observer.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observer.route('/reset')
def resetObserver():
    with DATA_LOCK:
        app.cm = CalibratedObserver(app.cc)
    return 'success'


@observer.route('/set_passive')
def controlSetPassive():
    with DATA_LOCK:
        app.cm.passiveMode()
    return buildModeController()
        

@observer.route('/set_track')
def controlSetTrack():
    with DATA_LOCK:
        app.cm.trackMode()
    return buildModeController()


def buildModeController():
    return """  <div class="btn-group" role="group" aria-label="Observer Capture Mode Control Buttons">
                  <input type="radio" class="btn-check" name="btnradio" id="passive" autocomplete="off" {passiveChecked}hx-get="{observerURL}set_passive" hx-target="#modeController">
                  <label class="btn btn-outline-primary" for="passive">Passive</label>
                  <input type="radio" class="btn-check" name="btnradio" id="track" autocomplete="off" {activeChecked}hx-get="{observerURL}set_track" hx-target="#modeController">
                  <label class="btn btn-outline-primary" for="track">Track</label>
                </div>""".replace(
        "{observerURL}", url_for(".buildObserver")).replace(
        "{passiveChecked}", 'checked=""' if app.cm.mode == "passive" else '').replace(
        "{activeChecked}", 'checked=""' if app.cm.mode == "track" else '')


@observer.route('/get_mode_controller')
def getModeController():
    return buildModeController()


@observer.route('/')
def buildObserver():
    if type(app.cm) is not CalibratedObserver:
        with DATA_LOCK:
            app.cm = CalibratedObserver(app.cc)
    with open(f"{os.path.dirname(__file__)}/templates/Observer.html", "r") as f:
        template = f.read()
    cameraButtons = '<input type="button" value="Virtual Map" onclick="liveCameraClick(\'VirtualMap\')">' + ' '.join([f'''<input type="button" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    defaultCam = [camName for camName, cam in app.cc.cameras.items()][0]
    return template.replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{observerURL}", url_for('.buildObserver')).replace(
        "{configuratorURL}", '/configurator')


def captureToChangeRow(capture):
    encodedBA = imageToBase64(capture.visual())
    name = "None"
    objType = "None"
    center = ", ".join(["0", "0"])
    health = ""
    numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "
    with open(f"{os.path.dirname(__file__)}/templates/TrackedObjectRow.html") as f:
        changeRowTemplate = f.read()
    moveDistance = app.cm.cc.rsc.trackedObjectLastDistance(capture)
    moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f} mm"
    changeRow = changeRowTemplate.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in app.cm.cc.rsc.changeSetToRealCenter(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{observerURL}", url_for(".buildObserver")).replace(
        "{encodedBA}", imageToBase64(capture.visual()))
    return changeRow


def buildObjectTable():
    changeRows = []
    print(f"Aware of {len(app.cm.memory)} objects")
    for capture in app.cm.memory:
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@observer.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()
    
    
@observer.route('/objects/<objectId>', methods=['GET'])
def getObjectSettings(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    objectName = cap.oid
    observerURL = url_for(".buildObserver")
    with open(f"{os.path.dirname(__file__)}/templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return template.replace(
        "{observerURL}", observerURL).replace(
        "{objectName}", cap.oid)


@observer.route('/objects/<objectId>', methods=['POST'])
def updateObjectSettings(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404
    newName = request.form["objectName"]
    if newName != cap.oid:
        cap.oid = newName
    return f"""<div id="objectTable" hx-get="{url_for(".buildObserver")}/objects" hx-trigger="every 1s"></div>"""
    
    
@observer.route('/objects/<objectId>', methods=['DELETE'])
def deleteObjectSettings(objectId):
    app.cm.deleteObject(objectId)
    return f"""<div id="objectTable" hx-get="{url_for(".buildObserver")}/objects" hx-trigger="every 1s"></div>"""


@observer.route('/object_distances/<objectId>', methods=['GET'])
def getObjectDistances(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    with open(f"{os.path.dirname(__file__)}/templates/ObjectDistanceCard.html") as f:
        cardTemplate = f.read()
    objDistCards = []
    for target in app.cm.memory:
        if target.oid == cap.oid:
            continue
        else:
            objDistCards.append(cardTemplate.replace(
                "{targetName}", target.oid).replace(
                "{encodedBA}", imageToBase64(target.visual())).replace(
                "{objectDistance}", f"{app.cm.cc.rsc.distanceBetweenObjects(cap, target):6.0f} mm"))
                
    with open(f"{os.path.dirname(__file__)}/templates/ObjectDistanceTable.html") as f:
        template = f.read()
    return template.replace(
        "{observerURL}", url_for(".buildObserver")).replace(
        "{objectName}", cap.oid).replace(
        "{objectDistanceCards}", "\n".join(objDistCards))


def minimapGenerator():
    while True:
        camImage = app.cm.cc.buildMiniMap(
            blueObjects=app.cm.memory,
            greenObjects=[app.cm.lastClassification] if app.cm.lastClassification is not None else None)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
    

@observer.route('/minimap')
def minimapResponse():
    return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def setObserverApp(newApp):
    global app
    app = newApp


def main():
    lock = FileLock()
    lock.acquire()

    app = Flask(__name__)
    # Register shutdown hook
    atexit.register(lock.release)
    
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(observer, url_prefix='/observer')
    app.register_blueprint(calibrator, url_prefix='/calibrator')
    # Configurator is now a separate app
    # app.register_blueprint(configurator, url_prefix='/configurator')
    app.cm = CalibratedObserver(app.cc)
    
    # setConfiguratorApp(app) # No longer needed in observer
    setCalibratorApp(app)

    @app.route('/')
    def index():
        return redirect('/observer', code=303)

    @app.route('/bootstrap.min.css', methods=['GET'])
    def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
            bscss = f.read()
        return Response(bscss, mimetype="text/css")
    
    @app.route('/bootstrap.min.js', methods=['GET'])
    def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            bsjs = f.read()
        return Response(bsjs, mimetype="application/javascript")
    
    @app.route('/htmx.min.js', methods=['GET'])
    def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            htmx = f.read()
        return Response(htmx, mimetype="application/javascript")
    
    registerCaptureService(app)
    print(f"Launching Observer Server on {PORT}")
    try:
        app.run(host="0.0.0.0", port=PORT)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
