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
from ipynb.fs.full.HarmonyEye import cc, cm, mc, cameras, hStackImages, vStackImages, HarmonyConfiguration, HarmonyMachine, HarmonyCamera, QuantumObject

import threading
import atexit
from flask import Flask, render_template, Response, request, make_response
from traceback import format_exc


CONSOLE_OUTPUT = "No Output Yet"

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
        global CONSOLE_OUTPUT
        with data_lock:
            if ENABLE_CYCLE:
                try:
                    cm.cycle()
                    CONSOLE_OUTPUT = ""
                except AssertionError as ae:
                    diceAEStr = "Unable to distinguish 0 or 2 dice"
                    if diceAEStr == str(ae)[:len(diceAEStr)]:
                        CONSOLE_OUTPUT = "Shake dice tray"
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


observerApp = createCaptureApp()


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(cameras.values())[0]
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
            elif 'selectedObject' in cm.lastMemory:
                obj = cm.lastMemory['selectedObject']
                currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(obj)]
                action = f"{obj.name} selected"
                line2 = "as actor" if len(cm.selectedObjects) == 1 else "as target"
                consoleImage = cv2.putText(zeros, action,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, line2,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            elif 'confirmedAction' in cm.lastMemory:
                roll, sel0, sel1 = cm.lastMemory['confirmedAction']
                action = f"{sel0.name} attacking"
                line2 = f"{sel1.name} confirmed ({roll})"
                consoleImage = cv2.putText(zeros, action,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, line2,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            elif 'actedOn' in cm.lastMemory:
                obj = cm.lastMemory['actedOn']
                currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(obj)]
                result, roll, actor = obj.actionHistory[list(obj.actionHistory.keys())[-1]]
                action = f"{obj.name} {result} by {actor} ({roll})"
                objectIdentifier = f'{obj.objectType} {obj.name}'
                objectLocation = f'at {currentLocation}'
                consoleImage = cv2.putText(zeros, objectIdentifier,
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, objectLocation,
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, action,
                    (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
   
        ret, consoleImage = cv2.imencode('.jpg', zeros)
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


@observerApp.route('/htmx.min.js', methods=['GET'])
def getHTMX():
    with open("templates/htmx.min.js", "r") as f:
        htmx = f.read()
    return Response(htmx, mimetype="application/javascript")


def genDiceCameraView(camName):
    while True:
        try:
            cam = cameras[str(camName)]
            assert cam.camType == "dice"
            try:
                diceRoll = cam.collectDiceRoll()
            except:
                diceRoll = "Failed"
            img = cam.drawActiveZone(cam.mostRecentFrame)
            cv2.putText(img, f"Result: {diceRoll}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 168, 255), 3, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Failed genCameraFullViewWithActiveZone for {camName} -- {e}")
            yield (b'--frame\r\nContent-Type: image/jpg\r\n\r\n\r\n')


@observerApp.route('/config/dicewatcher', methods=['GET'])
def dicewatcher():
    return Response(genDiceCameraView("Dice"), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraFullViewWithActiveZone(camName):
    while True:
        try:
            cam = cameras[str(camName)]
            if cam.camType == "dice":
                cam.capture()
            img = cam.drawActiveZone(cam.mostRecentFrame)
            ret, img = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Failed genCameraFullViewWithActiveZone for {camName} -- {e}")
            yield (b'--frame\r\nContent-Type: image/jpg\r\n\r\n\r\n')
    
    
@observerApp.route('/config/camera/<camName>')
def cameraActiveZoneWithObjects(camName):
    return Response(genCameraFullViewWithActiveZone(str(camName)), mimetype='multipart/x-mixed-replace; boundary=frame')


def buildCalibrationPlan():
    calibCamHeaders = "".join([f"<th>Cam {i}</th>" for i in cameras.keys()])
    calibFormFields = []
    for ptNum, (activeCameras, calibCoordinates) in cm.cc.calibrationPlan.items():
        checkboxes = "".join([f'<td><input type="checkbox" {"checked=true" if i in activeCameras else ""} id="calib{ptNum}Cam{i}" name="calib{ptNum}Cam{i}"></td>'
                              for i, cam in cameras.items()])
        coordinates = f'<td><input class="form-control" type="text" id="calib{ptNum}Coord" size="35" name="calib{ptNum}Coord" value="{calibCoordinates}"></td>'
        calibFormFields.append("".join(["".join(checkboxes), coordinates]))

    with open("templates/CalibrationForm.html") as f:
        template = f.read()

    for i, calib in enumerate(calibFormFields):
        template = template.replace(f"{{calib{i}}}", calib)

    return template.replace("{calibCamHeaders}", calibCamHeaders)


def buildConfigurator():
    cameraConfigRows = []
    clickSubs = []
    for cam in cameras.values():
        if cam is None:
            continue
        activeZone = json.dumps(cam.activeZone.tolist())
        optionValues = f"""<option value="field" selected>Game Field</option><option value="dice">Dice</option>""" if cam.camType == "field" else \
            f"""<option value="field">Game Field</option><option value="dice" selected>Dice</option>"""
        cameraConfigRows.append(f"""
            <div class="row justify-content-center text-center">
                <h3 class="mt-5">Camera {cam.camName} <input type="button" value="Delete" class="btn-error" hx-post="/config/delete_cam/{cam.camName}" hx-swap="outerHTML"></h3>
                <img src="/config/camera/{cam.camName}" title="{cam.camName} Capture" height="375" id="cam{cam.camName}" onclick="camClickListener('{cam.camName}', event)">
                <label for="az">Active Zone</label><br>
                <div class="container">
                    <div class="row">
                        <div class="col">
                            <input type="text" name="az" id="cam{cam.camName}_ActiveZone" value="{activeZone}" size="50" hx-post="/config/cam{cam.camName}_activezone" hx-swap="none">   
                            <input type="button" name="clearCam{cam.camName}AZ" value="Clear AZ" onclick="clearCamAZ('{cam.camName}', event)">
                        </div>
                        <div class="col">    
                            <label>Camera Type</label>
                            <select name="camType" hx-post="/config/cam{cam.camName}_type" hx-swap="none">
                              {optionValues}
                            </select>
                        </div>
                    </div>
                </div>
            </div>""")
    calibrationForm = buildCalibrationPlan()

    with open("templates/Configuration.html") as f:
        template = f.read()
    
    return template.replace(
        "{calibrationPlanForm}", calibrationForm).replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows))

    
@observerApp.route('/config', methods=['GET'])
def config():
    return buildConfigurator()


@observerApp.route('/config', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = "Saved Configuration"
    cc.saveConfiguration()
    return "success"


@observerApp.route('/config/calibration_plan', methods=['POST'])
def updateCalibrationPlan():
    newCalibPlan = {i: [[], []] for i in range(6)}
    for key, value in request.form.items():
        calibPtNum = int(key[5])
        if 'Cam' in key:
            camName = str(key[9])
            newCalibPlan[calibPtNum][0].append(camName)
        elif 'Coord' in key:
            newCalibPlan[calibPtNum][1] = json.loads(value)
    cm.cc.calibrationPlan = newCalibPlan
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = "Saved Calib Plan"
    return buildCalibrationPlan()


@observerApp.route('/config/cam<camName>_activezone', methods=['POST'])
def updateCamActiveZone(camName):
    global CONSOLE_OUTPUT
    print(f"Received update Active Zone request for {camName}")
    try:
        az = np.float32(json.loads(request.form.get(f"az")))
        cam = cameras[camName]
        cam.setActiveZone(az)
    except:
        print(f"Unrecognized data: {camName} - {az}")
    CONSOLE_OUPUT = f"Updated {camName} AZ"
    return "success"


@observerApp.route('/config/cam<camName>_type', methods=['POST'])
def updateCamType(camName):
    global CONSOLE_OUTPUT
    print(f"Received update Active Zone request for {camName}")
    try:
        camType = str(request.form.get(f"camType"))
        cam = cameras[camName]
        cam.camType = camType 
    except:
        print(f"Unrecognized data: {camName} - {camType}")
    CONSOLE_OUPUT = f"Updated {camName} type to {camType}"
    return "success"


@observerApp.route('/config/new_camera', methods=['GET'])
def getNewCameraForm():
    with open("templates/NewCamera.html", "r") as f:
        template = f.read()
    return template


def resetHarmonyMachine():
    global cc, cm, cameras
    cc = HarmonyConfiguration()
    cm = HarmonyMachine(cc)


@observerApp.route('/config/new_camera', methods=['POST'])
def addNewCamera():
    global CONSOLE_OUTPUT
    camName = request.form.get("camName")
    camRot = request.form.get("camRot")
    camAddr = request.form.get("camAddr")
    with data_lock:
        cameras[camName] = HarmonyCamera(address=camAddr, activeZone=[[0, 0], [0, 1], [1, 1,], [1, 0]], camName=camName, rotate=camRot, camType="field")
        cm.cc.rsc = None
        cm.cc.saveConfiguration()
        resetHarmonyMachine()
    CONSOLE_OUTPUT = f"Added Camera {camName}"
    return """<script>window.location = '/config';</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/config'">"""


@observerApp.route('/config/reset', methods=['POST'])
def requestHarmonyReset():
    global CONSOLE_OUTPUT
    with data_lock: 
        resetHarmonyMachine()
        CONSOLE_OUTPUT = "Harmony was reset!"
    return """<script>window.location.reload();</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/config'">"""


@observerApp.route('/config/delete_cam/<camName>', methods=['POST'])
def deleteCamera(camName):
    global CONSOLE_OUTPUT
    with data_lock:
        cameras.pop(camName)
        cm.cc.rsc = None
        cm.cc.saveConfiguration()
        resetHarmonyMachine()
        CONSOLE_OUTPUT = f"Deleted Camera {camName}"
    return """<script>window.location.reload();</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/config'">"""


@observerApp.route('/calibrator/start_calibration')
def startCalibration():
    with data_lock:
        cm.startCalibration()
    return "success"
        
        
@observerApp.route('/calibrator/abort_calibration')
def abortCalibration():
    with data_lock:
        cm.abortCalibration()
    return "success"


@observerApp.route('/calibrator', methods=['GET'])
def buildCalibrator():
    if cm.mode != "calibrate":
        calibrationMonitor = f"""<p>Calibration in not in Progress<p>"""
    else:
        formattedPts = [f"> {pt}<br>" for pt in cm.calibrationPts]
        calibrationMonitor = f"""
            <p>Calibration in Progress: {cm.mode == "calibrate"}<br>Progress {len(cm.calibrationPts)} / {cm.cc.numCalibrationPoints}<br>{formattedPts}<p>
        """
        
    if cm.cc.rsc == None:
        calibratorResult = "Not Calibrated"
    else:
        ims = []
        for camName, cons in cm.cc.rsc.converters.items():
            if len(cons) == 1:
                camImage = cons[0].showUnwarpedImage()
            else:
                camImage = hStackImages([c.showUnwarpedImage() for c in cons])
            ims.append(camImage)
        unWarped = imageToBase64(vStackImages(ims))
        calibratorResult = f'{cm.cc.rsc}<br><img alt="Unwarped Camera Views" src="data:image/jpg;base64,{unWarped}">'

    with open("templates/Calibrator.html") as f:
        template = f.read()
    return template.replace(
        "{calibratorResult}", calibratorResult).replace(
        "{calibrationMonitor}", calibrationMonitor)

    if (start_calibration := request.form.get("calib_type")) == 'start_calibration':
        with data_lock:
            cm.startCalibration()
    elif (abort_calibration := request.form.get("calib_type")) == 'abort_calibration':
        with data_lock:
            cm.abortCalibration()
    return buildCalibrator()


def genCombinedCamerasView():
    while True:
        camImages = []
        for camName in cameras.keys():
            camImage = cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraWithChangesView(camName):
    camName = str(camName)
    cam = cameras[camName]
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
        # Paint Selected Objects Purple
        if cm.selectedObjects is not None:
            for idx, sel in enumerate(cm.selectedObjects):
                color = (255, 255, 0) if idx == 0 else (255, 0, 255)
                if sel.changeSet[camName].changeType not in ['delete', None]:
                    memContour = np.array([sel.changeSet[camName].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, color, -1)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/camWithChanges/<camName>')
def cameraViewWithChangesResponse(camName):
    return Response(genCameraWithChangesView(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def fullCam(camName):
    camName = str(camName)
    cam = cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/fullCam/<camName>')
def genFullCam(camName):
    return Response(fullCam(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        for camName in cameras.keys():
            camImage = cameras[camName].mostRecentFrame.copy()
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
            # Paint Selected Objects Purple
            if cm.selectedObjects is not None:
                for idx, sel in enumerate(cm.selectedObjects):
                    color = (255, 255, 0) if idx == 0 else (255, 0, 255)
                    if sel.changeSet[camName].changeType not in ['delete', None]:
                        memContour = np.array([sel.changeSet[camName].changePoints], dtype=np.int32)
                        camImage = cv2.drawContours(camImage, memContour, -1, color, -1)
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
        camImage = cm.buildMiniMap()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
    

@observerApp.route('/control/minimap')
def minimapResponse():
    return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/control/set_passive')
def controlSetPassive():
    with data_lock:
        cm.passiveMode()
    return "success"
        

@observerApp.route('/control/set_add')
def controlSetAdd():
    with data_lock:
        cm.addMode()
    return "success"
        

@observerApp.route('/control/set_move')
def controlSetMove():
    with data_lock:
        cm.moveMode()
    return "success"
        

@observerApp.route('/control/set_action')
def controlSetAction():
    with data_lock:
        cm.actionMode()
    return "success"


@observerApp.route('/control')
def buildController():
    with open("templates/Controller.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in cameras.keys()])
    defaultCam = [camName for camName, cam in cameras.items() if cam.camType == 'field'][0]
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
                    <button class="btn btn-primary" onclick="window.location.href='/objectsettings/{capture.oid}'">Edit</button>
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
    print(f"Aware of {len(cm.memory) + len(cm.newObjectBuffer)} objects")
    for capture in cm.newObjectBuffer:
        changeRows.append(captureToChangeRow(capture))
    changeRows.append("""<hr class="mt-4 mb-6 border-top"/>""")
    for capture in cm.memory:
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@observerApp.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


@observerApp.route('/objectsettings/<objectId>', methods=['GET'])
def getObjectInfo(objectId):
    capture = None
    for c in cm.newObjectBuffer + cm.memory:
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
        <form hx-post="/objectsettings/{objectId}">
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
    return buildController()


coords = {"real": "[]", **{cam.camName: "[]" for cam in cameras.values()}}

    
@observerApp.route('/annotatecams/<camName>/coord')
def getCamCoord(camName):
    global coords
    assert camName in coords, f"{camName} not in coords"
    return coords[camName]
    
    
@observerApp.route('/annotatecams/<camName>/coord', methods=['POST'])
def postCamCoord(camName):
    global coords
    camCoords = json.loads(coords[camName])
    newCoord = [int(d) for d in request.get_json()]
    camCoords.append(newCoord)
    coords[camName] = json.dumps(camCoords)
    realCoord = cm.cc.rsc.camCoordToRealSpace(camName, newCoord)
    for cam in cameras.values():
        if cam.camName == camName or cam.camType == "dice":
            continue
        camCoord = [int(d) for d in cm.cc.rsc.realCoordToCamSpace(cam.camName, realCoord)]
        coords[cam.camName] = json.dumps(json.loads(coords[cam.camName]) + [camCoord])
    coords['real'] = json.dumps(json.loads(coords['real']) + [[int(d) for d in realCoord]])
    return json.dumps(coords)
    
    
@observerApp.route('/annotatecams/clearcoord', methods=['POST'])
def clearCamCoord():
    global coords
    coords = {"real": "[]", **{cam.camName: "[]" for cam in cameras.values()}}
    return "Clear Coordinates"


def genAnnotatedCamera(camName):
    while True:
        try:
            cam = cameras[str(camName)]
            camCoords = json.loads(coords[str(camName)])
            if cam.camType == "dice":
                cam.capture()
            img = cam.drawActiveZone(cam.mostRecentFrame)
            for coord in camCoords:
                cv2.circle(img, coord, 8, (0, 0, 255), -1)
            ret, img = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Failed genCameraFullViewWithActiveZone for {camName} -- {e}")
            yield (b'--frame\r\nContent-Type: image/jpg\r\n\r\n\r\n')
    
    
@observerApp.route('/annotatecams/camera/<camName>')
def annotationCamera(camName):
    return Response(genAnnotatedCamera(str(camName)), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/annotatecams')
def annotateCams():
    cameraCols = []
    coordinateFields = []
    for cam in cameras.values():
        if cam is None or cam.camType == "dice":
            continue
        cameraCols.append(f"""
            <div class="col">
                <h3 class="mt-5">Camera {cam.camName}</h3>
                <img src="/annotatecams/camera/{cam.camName}" title="{cam.camName} Capture" width="100%" id="cam{cam.camName}" onclick="camClickListener('{cam.camName}', event)">
            </div>""")
        coordinateFields.append(f"""
            <div class="row">
                <h4 class="mt-5">{cam.camName} Coordinates: </h4>
                <div id="cam{cam.camName}Coord" hx-get="/annotatecams/{cam.camName}/coord" hx-trigger="every 2s"></div>
            </div>""")

    with open("templates/CamAnnotater.html") as f:
        template = f.read()
    
    return template.replace(
        "{cameraColumns}", "\n".join(cameraCols)).replace(
        "{coordinateFields}", "\n".join(coordinateFields)).replace(
        "{realCoord}", coords['real'])


if __name__ == "__main__":
    cm.cycle()
    print(f"Launching Observer Server on {PORT}")
    #observerApp.run(host="0.0.0.0", port=PORT, ssl_context="adhoc")
    observerApp.run(host="0.0.0.0", port=PORT)
