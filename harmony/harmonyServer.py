import os
from math import ceil
import base64
import argparse
import json
from io import BytesIO
from dataclasses import dataclass, asdict
from typing import Callable
from functools import wraps
import threading
import atexit
from traceback import format_exc
import time
from uuid import uuid4

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for, jsonify

from observer import HexGridConfiguration, HexCaptureConfiguration
from observer.configurator import configurator, setConfiguratorApp
from observer.observerServer import observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT

import os
import sys

oldPath = os.getcwd()
try:
    observerDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, observerDirectory)
    from ipynb.fs.full.HarmonyMachine import HarmonyMachine, INCHES_TO_MM
finally:
    os.chdir(oldPath)


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')

perspective_res = [1920, 1080]

def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(app.cc.cameras.values())[0]
        shape = (200, 600)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")

        consoleImage = cv2.putText(zeros, f'Cycle {app.cm.cycleCounter}',
            (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Mode: {app.cm.mode:7}',
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {app.cm.state:10}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'LO: {CONSOLE_OUTPUT}',
            (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        with DATA_LOCK:
            try:
                roundCount = app.cm.getRoundCount()
            except TypeError:
                roundCount = "N/A"
            consoleImage = cv2.putText(zeros, f'Round: {roundCount:3}-{app.cm.getPhase()}',
                (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')
        time.sleep(0.1)


@harmony.route('/harmony_console', methods=['GET'])
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


@harmony.route('/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@dataclass
class CellSelection:
    viewId: str
    firstCell: tuple
    additionalCells: list[tuple] | None = None

    @property
    def secondCell(self):
        return self.additionalCells[0] if self.additionalCells else None


SELECTED_CELLS = {}


def genCameraWithChangesView(camName, viewId=None):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        camImage = cam.cropToActiveZone(cam.mostRecentFrame.copy())

        grid = app.cc.cameraGriddle(camName)
        camImage = cam.cropToActiveZone(cv2.addWeighted(grid, 0.3, camImage, 1.0 - 0.3, 0.0))
        camImage = cv2.resize(camImage, perspective_res, interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/camWithChanges/<camName>/<viewId>')
def cameraViewWithChangesResponse(camName, viewId):
    if camName == "VirtualMap":
        return Response(minimapGenerator(viewId=viewId), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(genCameraWithChangesView(camName, viewId=viewId), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/canvas_data/<viewId>')
def getCanvasData(viewId):
    # Example structure - populate this from app.cm.memory
    objects = {}
    print(f"Hex: {app.cm.cc.hex}")
    for obj in app.cm.memory:
        objects[obj.oid] = {
            "VirtualMap": [
                (pt.tolist()[0][0], pt.tolist()[0][1])
                for pt in app.cm.cc.hex_at_axial(*app.cm.cc.changeSetToAxialCoord(obj))],
            **{
                camName: [(pt[0], pt[1]) for pt in obj.changeSet[camName].changePoints]
                for camName in app.cc.cameras.keys()}
        }
    
    data = {
        "objects": {
            obj.oid: {"VirtualMap": [pt.tolist()[0] for pt in app.cm.cc.hex_at_axial(*app.cm.cc.changeSetToAxialCoord(obj))], **{
                camName: obj.changeSet[camName].changePoints for camName in app.cc.cameras.keys()
            }} for obj in app.cm.memory
        },
        "moveable": ["M0"],
        "selectable":  ["M0", "S0", "T0", "A0"],
        "terrain": ["TO"],
        "allies": ["A0"],
        "enemies": ["E0"],
        "viewId": viewId,
        "selection": asdict(SELECTED_CELLS[viewId]) if viewId in SELECTED_CELLS else {},
        "cameraName": "VirtualMap"
    }
    return jsonify(data)


def fullCam(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/fullCam/<camName>')
def genFullCam(camName):
    return Response(fullCam(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        with DATA_LOCK:
            camImages = []
            if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
                print("Has changes")
            if app.cm.lastClassification is not None:
                print("Has class")
            camImages = app.cm.getCameraImagesWithChanges(app.cc.cameras.keys()).values()
            camImage = vStackImages(camImages)
            camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
            ret, camImage = cv2.imencode('.jpg', camImage)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
        time.sleep(0.1)


@harmony.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/reset')
def resetHarmony():
    with DATA_LOCK:
        app.cm = HarmonyMachine(app.cc)
        app.cm.reset()
        CONSOLE_OUTPUT = "Harmony reset."
    return 'success'


@harmony.route('/load', methods=["POST"])
def loadHarmony():
    game_name = request.form["game_name"]
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        CONSOLE_OUTPUT = "Loading game state..."
    time.sleep(3)
    with DATA_LOCK:
        app.cm.loadGame(gameName=game_name)        
        CONSOLE_OUTPUT = "Game state reloaded."
    return 'success'


@harmony.route('/save', methods=["POST"])
def saveHarmony():
    game_name = request.form["game_name"]
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        app.cm.saveGame(gameName=game_name)
        CONSOLE_OUTPUT = "Saved game state."
    return 'success'


@harmony.route('/')
def buildHarmony():
    if type(app.cm) is not HarmonyMachine:
        resetHarmony()
    with open(f"{os.path.dirname(__file__)}/harmony_templates/Harmony.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-info" value="Camera {camName}" onclick="gameWorldClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="gameWorldClick(\'VirtualMap\')">{cameraButtons}"""
    defaultCam = [camName for camName, cam in app.cc.cameras.items()]
    if len(defaultCam) == 0:
        defaultCam = "None"
    else:
        defaultCam = defaultCam[0]
    return template.replace(
        "{viewId}", str(uuid4())).replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{harmonyURL}", url_for('.buildHarmony')).replace(
        "{configuratorURL}", '/configurator')


def captureToChangeRow(capture):
    name = "None"
    objType = "None"
    center = ", ".join(["0", "0"])
    with open(f"{os.path.dirname(__file__)}/harmony_templates/TrackedObjectRow.html") as f:
        changeRowTemplate = f.read()
    moveDistance = app.cm.cc.trackedObjectLastDistance(capture)
    moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f}"
    changeRow = changeRowTemplate.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in app.cm.cc.changeSetToAxialCoord(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{observerURL}", url_for(".buildHarmony")).replace(
        "{encodedBA}", imageToBase64(app.cm.object_visual(capture)))
    return changeRow
    

def buildObjectTable():
    changeRows = []
    print(f"Structure object table")
    for capture in app.cm.memory:
        print(f"Cap {capture.oid}")
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@harmony.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


def getInteractor():
    return buildObjectTable()


def findObjectIdOr404(objectId_endpoint: Callable) -> Callable:
    @wraps(objectId_endpoint)
    def findOr404_endpoint(**kwargs):
        try:
            objectId = kwargs.pop("objectId")
        except KeyError as ke:
            error = f"{objectId} Not found"
            print(error)
            return error, 404
        return objectId_endpoint(cap=app.cm.findObject(objectId=objectId), **kwargs)
    return findOr404_endpoint
    
    
@harmony.route('/objects/<objectId>', methods=['GET'])
@findObjectIdOr404
def getObject(cap):
    footprint_enabled = request.args.get('footprint', "false") == "true"

    objectName = cap.oid
    with open(f"{os.path.dirname(__file__)}/harmony_templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{objectSettings}", buildObjectSettings(cap)).replace(
        "{footprintToggleState}", str(not footprint_enabled).lower()).replace(
        "{encodedBA}", imageToBase64(app.cm.object_visual(cap, withContours=footprint_enabled)))


@harmony.route('/objects/<objectId>', methods=['POST'])
@findObjectIdOr404
def updateObjectSettings(cap):
    # TODO: implement
    return buildObjectTable()
    
    
@harmony.route('/objects/<objectId>', methods=['DELETE'])
@findObjectIdOr404
def deleteObjectSettings(cap):
    with DATA_LOCK:
        app.cm.deleteObject(cap.oid)
    return buildObjectTable()


def buildObjectSettings(cap, objType=None):
    return "200"
    


@harmony.route('/objects/<objectId>/settings', methods=['GET'])
@findObjectIdOr404
def getObjectSettings(cap):
    return buildObjectSettings(cap)


@harmony.route('/objects/<objectId>/type', methods=['POST'])
@findObjectIdOr404
def updateObjectType(cap):
    newType = request.form["objectType"]
    assert newType in ["None", "Terrain", "Structure", "Unit"], f"Unrecognized object type: {newType}"
    return buildObjectSettings(cap, objType=newType)    


@harmony.route('/object_factory/<viewId>', methods=['GET'])
def buildObjectFactory(viewId):
    selectedCell = SELECTED_CELLS[viewId].firstCell
    return f"""
        <form hx-post="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#interactor">
            <label for="object_name" class="form-check-label">Object Name</label>
            <input type="text" name="object_name" value=""><br>
            <label for="selected_cells" class="form-check-label">Selected Cells</label>
            <input type="text" name="selected_cells" value="{selectedCell}"><br>
            <input type="submit" class="btn btn-primary" value="Define Object">
        </form>
    """

    
interactor_template = """
  <hr>
  <span class="border-3 border-info">
  <div class="border-3 border-info" align="left">
    {info}
  </div>
  </span>
  <span class="border-3 border-warning">
  <div class="border-3 border-warning" align="right">
    <h3>Selected Cell Actions</h3>
    {actions}
  <div>
  </span>
"""


@harmony.route('/object_factory/<viewId>', methods=['POST'])
def buildObject(viewId):
    objectName = str(request.form.get("object_name"))
    selectedAxial = SELECTED_CELLS[viewId].firstCell
    trackedObject = app.cm.cc.define_object_from_axial(objectName, *selectedAxial)
    app.cm.commitChanges(trackedObject)
    return interactor_template.format(info=f"""
        <h2>Selected cell: {selectedAxial}</h2>
        <h3>Object Name: {trackedObject.oid}</h3>
        """,
                                    actions=f"""
        <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
        <hr>
        <input type="button" class="btn btn-danger" value="Delete Object" hx-delete="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#interactor">
        """)


@harmony.route('/object_factory/<viewId>', methods=['DELETE'])
def deleteObject(viewId):
    selected = SELECTED_CELLS[viewId]
    overlap = None
    for mem in app.cm.memory:
        mem_axial = app.cm.cc.changeSetToAxialCoord(mem)
        if mem_axial == selected.firstCell:
            overlap = mem
            break
    app.cm.memory.remove(mem)
    SELECTED_CELLS.pop(viewId)
    return "Success"


@harmony.route('/request_move/<oid>/<viewId>', methods=['GET'])
def moveObjectDefinition(oid, viewId):
    try:
        selected = SELECTED_CELLS[viewId]
        firstCell = selected.firstCell
        secondCell = selected.secondCell
    except Exception as e:
        raise Exception("500") from e
    trackedObject = app.cm.cc.define_object_from_axial(oid, *secondCell)
    existing = app.cm.cc.define_object_from_axial(oid, *firstCell)
    with DATA_LOCK:
        app.cm.memory.remove(existing)
        app.cm.commitChanges(trackedObject)
        SELECTED_CELLS.pop(viewId)
    return "Success"


@harmony.route('/clear_pixel/<viewId>', methods=['GET'])
def clearPixel(viewId):
    if viewId in SELECTED_CELLS:
        with DATA_LOCK:
            SELECTED_CELLS.pop(viewId)
    return ""

@harmony.route('/select_pixel', methods=['POST'])
def selectPixel():
    global SELECTED_CELLS
    if len(SELECTED_CELLS) > 12:
        with DATA_LOCK:
            SELECTED_CELLS = {}
    
    viewId = request.form["viewId"]
    pixel = json.loads(request.form["selectedPixel"])
    x, y = pixel
    cam = request.form["selectedCamera"]
    appendPixel = bool(request.form["appendPixel"])
    if cam == "VirtualMap":
        axial_coord = app.cm.cc.pixel_to_axial(x, y)
    else:
        axial_coord = app.cm.cc.camCoordToAxial(cam, (x, y))
    print(f"viewId {viewId} || Received: Pixel {pixel} on Cam {cam} || Translated to Axial: {axial_coord}")
    with DATA_LOCK:
        existing = SELECTED_CELLS.get(viewId, None)
        if existing:
            if existing.secondCell is None or appendPixel:
                SELECTED_CELLS[viewId].additionalCells = [axial_coord]
            else:
                SELECTED_CELLS[viewId] = CellSelection(viewId, axial_coord)
        else:
            SELECTED_CELLS[viewId] = CellSelection(viewId, axial_coord)
    q, r = axial_coord

    overlaps = []
    targets = []
    selected = SELECTED_CELLS[viewId]
    for mem in app.cm.memory:
        mem_axial = app.cm.cc.changeSetToAxialCoord(mem)
        if mem_axial == selected.firstCell:
            overlaps.append(mem)
        if mem_axial == selected.secondCell:
            targets.append(mem)

    if overlaps:
        overlap = overlaps[0]
        if selected.secondCell:
            distance = app.cm.cc.axial_distance(selected.firstCell, selected.secondCell)
            distance = f"{distance} {'cell' if abs(distance) == 1 else 'cells'}"
            return interactor_template.format(info=f"""
                    <h2>Selected cell: {selected.firstCell}</h2>
                    <h2>Object name: {overlap.oid}</h2>
                    <h3>Distance to {selected.secondCell}  --  {distance}</h3>
                """,
                                                actions=f"""
                    <input type="button" class="btn btn-info" value="Move Object" hx-get="{url_for(".buildHarmony")}request_move/{overlap.oid}/{viewId}" hx-target="#interactor">
                    <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
                """)
        else:
            return interactor_template.format(info=f"""
                    <h2>Selected cell: {selected.firstCell}</h2>
                    <h2>Object name: {overlap.oid}</h2>
                """,
                                                actions=f"""
                <div id="object_factory">
                    <input type="button" class="btn btn-danger" value="Delete Object" hx-delete="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#object_factory">
                </div>
                <hr>
                <input type="button" class="btn btn-danger" value="Clear Selected Pixel" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
                """)
    else:
        if selected.secondCell:
            distance = app.cm.cc.axial_distance(selected.firstCell, selected.secondCell)
            distance = f"{distance} {'cell' if abs(distance) == 1 else 'cells'}"
            return interactor_template.format(info=f"""
                    <h2>Selected cell: {selected.firstCell}</h2>
                    <h3>Distance to {selected.secondCell}  --  {distance}</h3>
                """,
                                                actions=f"""
                    <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
                """)
        else:
            return interactor_template.format(info=f"""
                    <h2>Selected cell: {selected.firstCell}</h2>
                """,
                                                actions=f"""
                <div id="object_factory">
                    <input type="button" class="btn btn-success" value="Define Object" hx-get="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#object_factory">
                </div>
                <input type="button" class="btn btn-danger" value="Clear Selected Pixel" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
                """)


@harmony.route('/select_additional_pixel/<viewId>', methods=['POST'])
def selectAdditionalPixel(viewId):
    pass


def minimapGenerator(viewId=None):
    while True:
        camImage = app.cm.buildMiniMap()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/minimap/<viewId>')
def minimapResponse(viewId):
    return Response(minimapGenerator(viewId), mimetype='multipart/x-mixed-replace; boundary=frame')


def setHarmonyApp(newApp):
    global app
    app = newApp


def create_harmony_app():
    global app
    
    app = Flask(__name__)
    app.cc = HexCaptureConfiguration()
    if app.cc.hex is None:
        app.cc.hex = HexGridConfiguration()
    app.cc.capture()
    app.cm = HarmonyMachine(app.cc)
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.register_blueprint(harmony, url_prefix='/harmony')
    resetHarmony()
    setConfiguratorApp(app)
    setObserverApp(app)
    
    @app.route('/')
    def index():
        return redirect('/harmony', code=303)
    
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

    return app


def main():
    create_harmony_app()
    PORT = 7000
    print(f"Launching harmony Server on {PORT}")
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

