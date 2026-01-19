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

from ipynb.fs.full.CalibratedObserver import CalibratedCaptureConfiguration, CalibrationObserver, CalibratedObserver

import threading
import atexit
from flask import Blueprint, render_template, Response, request, make_response, redirect, url_for, current_app
from traceback import format_exc


CONSOLE_OUTPUT = "No Output Yet"
POOL_TIME = 0.01 #Seconds
ENABLE_CYCLE = True
DATA_LOCK = threading.Lock()

app = None
calibrator = Blueprint('calibrator', __name__, template_folder='templates')


captureTimer = threading.Timer(0,lambda x: None,())    
def registerCaptureService(app):
    def interrupt():
        global captureTimer
        captureTimer.cancel()

    def cycleMachine():
        global CONSOLE_OUTPUT
        with DATA_LOCK:
            if ENABLE_CYCLE:
                try:
                    cmOut = app.cm.cycle()
                    if cmOut is not None:
                        CONSOLE_OUTPUT = f"({app.cm.cycleCounter%100})-{cmOut}"
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
        mode =  f'Mode: {app.cm.mode:7}' + ('' if app.cm.dowel_position is None else f'-{app.cm.dowel_position}')
        consoleImage = cv2.putText(zeros, mode,
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {app.cm.state:10}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'LO: {CONSOLE_OUTPUT}',
            (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2, cv2.LINE_AA)

        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@calibrator.route('/observer_console', methods=['GET'])
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
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@calibrator.route('/combinedCameras')
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
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@calibrator.route('/camWithChanges/<camName>')
def cameraViewWithChangesResponse(camName):
    return Response(genCameraWithChangesView(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def fullCam(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@calibrator.route('/fullCam/<camName>')
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
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@calibrator.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@calibrator.route('/set_passive')
def controlSetPassive():
    with DATA_LOCK:
        app.cm.passiveMode()
    return buildModeController()
        

@calibrator.route('/set_track_<position>')
def controlSetTrack(position):
    assert position in ['first', 'top', 'hypos', 'longs', 'shorts'], f"Unrecognzed dowel position: {position}"
    with DATA_LOCK:
        app.cm.trackMode(dowel_position=position)
    return buildModeController()


def buildModeController():
    return """
            <div class="btn-group" role="group" aria-label="Observer Capture Mode Control Buttons">
              <input type="radio" class="btn-check" name="btnradio" id="passive" autocomplete="off" {passiveChecked}hx-get="{calibratorURL}set_passive">
              <label class="btn btn-outline-primary" for="passive">Passive</label>
              <input type="radio" class="btn-check" name="btnradio" id="track_first" autocomplete="off" {firstChecked}hx-get="{calibratorURL}set_track_first">
              <label class="btn btn-outline-primary" for="track_first">First (Reset)</label>
              <input type="radio" class="btn-check" name="btnradio" id="track_top" autocomplete="off" {topChecked}hx-get="{calibratorURL}set_track_top">
              <label class="btn btn-outline-primary" for="track_top">Track Top</label>
              <input type="radio" class="btn-check" name="btnradio" id="track_hypos" autocomplete="off" {hyposChecked}hx-get="{calibratorURL}set_track_hypos">
              <label class="btn btn-outline-primary" for="track_hypos">Track Hypos</label>
              <input type="radio" class="btn-check" name="btnradio" id="track_longs" autocomplete="off" {longsChecked}hx-get="{calibratorURL}set_track_longs">
              <label class="btn btn-outline-primary" for="track_longs">Track Longs</label>
              <input type="radio" class="btn-check" name="btnradio" id="track_shorts" autocomplete="off" {shortsChecked}hx-get="{calibratorURL}set_track_shorts">
              <label class="btn btn-outline-primary" for="track_shorts">Track Short</label>
            </div>""".replace(
        "{calibratorURL}", url_for(".buildCalibrator")).replace(
        "{passiveChecked}", 'checked=""' if app.cm.mode == "passive" else '').replace(
        "{firstChecked}", 'checked=""' if app.cm.mode == "track" and app.cm.dowel_position == "first" else '').replace(
        "{topChecked}", 'checked=""' if app.cm.mode == "track" and app.cm.dowel_position == "top" else '').replace(
        "{hyposChecked}", 'checked=""' if app.cm.mode == "track" and app.cm.dowel_position == "hypos" else '').replace(
        "{longsChecked}", 'checked=""' if app.cm.mode == "track" and app.cm.dowel_position == "longs" else '').replace(
        "{shortsChecked}", 'checked=""' if app.cm.mode == "track" and app.cm.dowel_position == "shorts" else '')


@calibrator.route('/get_mode_controller')
def getModeController():
    return buildModeController()


@calibrator.route('/commit_calibration')
def commitCalibration():
    app.cm.buildRealSpaceConverter()
    with DATA_LOCK:
        global CONSOLE_OUTPUT
        CONSOLE_OUTPUT = f"{app.cm.cycleCounter} - Calibration stored"
    app.cc.saveConfiguration()
    return redirect(url_for('.buildCalibrator'), code=303)


def resetCalibrationObserver():
    with DATA_LOCK:
        app.cm = CalibrationObserver(app.cc)


@calibrator.route('/reset')
def requestResetCalibrationObserver():
    resetCalibrationObserver()
    return "reset successful!", 200


@calibrator.route('/')
def buildCalibrator():
    if type(app.cm) is not CalibrationObserver:
        resetCalibrationObserver()
    with open(f"{os.path.dirname(__file__)}/templates/Calibrator.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-primary" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    defaultCam = [camName for camName, cam in app.cc.cameras.items()][0]
    return template.replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{calibratorURL}", url_for('.buildCalibrator'))


def captureToChangeRow(capture):
    encodedBA = imageToBase64(app.cm.object_visual(capture))
    name = "None"
    realTriangle = {camName: ctp[-1] for camName, ctp in capture.calibTriPts.items()}
    center = ", ".join(["0", "0"])
    health = ""
    numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "
    changeRow = f"""
        <div class="row mb-1">
            <div class="col">
                <div class="row">
                    <p>Realspace Coord:<br>{realTriangle}</p>
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
    print(f"Aware of {len(app.cm.memory)} objects")
    for capture in app.cm.memory:
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@calibrator.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


def setCalibratorApp(newApp):
    global app
    app = newApp


if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(calibrator, url_prefix='/calibrator')
    app.cm = CalibrationObserver(cc)

    @app.route('/<page>')
    def getPage(page):
        try:
            with open(f"{os.path.dirname(__file__)}/templates/{page}") as page:
                page = page.read()
        except Exception as e:
            print(f"Failed to find page: {e}")
            page = "Not found!"
        return page

    registerCaptureService(app)
    print(f"Launching Observer Server on Port 7000")
    app.run(host="0.0.0.0", port=7000)
