

from math import ceil
import base64
import argparse
import json
from io import BytesIO
from dataclasses import dataclass, asdict, field
from typing import Callable
from functools import wraps
import threading
import atexit
from traceback import format_exc
import time
from uuid import uuid4
import copy

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for, jsonify, current_app, stream_with_context

from observer import HexGridConfiguration, HexCaptureConfiguration
from observer.configurator import configurator, setConfiguratorApp
from observer.observerServer import observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT



import sys
import os

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


@harmony.route('/harmony_console', methods=['GET'])
def getConsoleImage():
    return Response(stream_with_context(renderConsole()), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCamerasView():
    while True:
        camImages = []
        for camName in current_app.cc.cameras.keys():
            camImage = current_app.cc.cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/combinedCameras')
def combinedCamerasResponse():
    return Response(stream_with_context(genCombinedCamerasView()), mimetype='multipart/x-mixed-replace; boundary=frame')


@dataclass
class CellSelection:
    firstCell: tuple | None = None
    additionalCells: list[tuple] | None = None

    def __post_init__(self):
        self.additionalCells = self.additionalCells or []

    @property
    def secondCell(self):
        return self.additionalCells[0] if self.additionalCells else None


@dataclass
class SessionConfig:
    moveable: list = field(default_factory=list)
    selectable: list = field(default_factory=list)
    terrain: list = field(default_factory=list)
    allies: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    targetable: list = field(default_factory=list)
    selection: CellSelection = field(default_factory=CellSelection)


SESSIONS = {}
APPS = []


def genCameraWithGrid(camName):
    camName = str(camName)
    cam = current_app.cc.cameras[camName]
    while True:
        camImage = cam.cropToActiveZone(cam.mostRecentFrame.copy())
        grid = current_app.cc.cameraGriddle(camName)
        camImage = cam.cropToActiveZone(cv2.addWeighted(grid, 0.3, camImage, 1.0 - 0.3, 0.0))
        camImage = cv2.resize(camImage, perspective_res, interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/camWithChanges/<camName>/<viewId>')
def cameraViewWithChangesResponse(camName, viewId):
    if camName == "VirtualMap":
        return Response(stream_with_context(minimapGenerator()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(stream_with_context(genCameraWithGrid(camName)), mimetype='multipart/x-mixed-replace; boundary=frame')


def safe_point(pt):
    p = pt.tolist()
    # Handle contour format [[x,y]] vs point format [x,y]
    if len(p) > 0 and isinstance(p[0], list):
        return (p[0][0], p[0][1])
    return (p[0], p[1])

def axial_to_ui_object(q, r):
    return {
        "VirtualMap": [
            safe_point(pt)
            for pt in current_app.cm.cc.hex_at_axial(q, r)],
        **{camName: [
            safe_point(pt)
            for pt in current_app.cm.cc.cam_hex_at_axial(camName, q, r)] for camName in current_app.cc.cameras.keys()}
    }


@harmony.route('/canvas_data/<viewId>')
def getCanvasData(viewId):
    objects = {}
    for obj in current_app.cm.memory:
        objects[obj.oid] = {
            "VirtualMap": [
                safe_point(pt)
                for pt in current_app.cm.cc.hex_at_axial(*current_app.cm.cc.changeSetToAxialCoord(obj))],
            **{
                camName: [(pt[0], pt[1]) for pt in obj.changeSet[camName].changePoints]
                for camName in current_app.cc.cameras.keys()}
        }
    
    session_config = SESSIONS.get(viewId, SessionConfig())

    data = {
        "objects": objects,
        **asdict(session_config),
        "selection": {
            "firstCell": axial_to_ui_object(*session_config.selection.firstCell) if session_config.selection.firstCell is not None else None,
            "additionalCells": [axial_to_ui_object(*cell) for cell in session_config.selection.additionalCells]
        }
    }
    return jsonify(data)


def genCombinedCameraWithChangesView():
    while True:
        with DATA_LOCK:
            camImages = []
            if current_app.cm.lastChanges is not None and not current_app.cm.lastChanges.empty:
                print("Has changes")
            if current_app.cm.lastClassification is not None:
                print("Has class")
            camImages = current_app.cm.getCameraImagesWithChanges(current_app.cc.cameras.keys()).values()
            camImage = vStackImages(camImages)
            camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
            ret, camImage = cv2.imencode('.jpg', camImage)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
        time.sleep(0.1)


@harmony.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(stream_with_context(genCombinedCameraWithChangesView()), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/reset')
def resetHarmony():
    with DATA_LOCK:
        new_cm = HarmonyMachine(current_app.cc)
        new_cm.reset()
        
        # Update all registered apps to share the new machine
        if APPS:
            for app_instance in APPS:
                app_instance.cm = new_cm
        else:
            # Fallback if APPS not populated (legacy)
            current_app.cm = new_cm
            
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
        current_app.cm.loadGame(gameName=game_name)        
        CONSOLE_OUTPUT = "Game state reloaded."
    return 'success'


@harmony.route('/save', methods=["POST"])
def saveHarmony():
    game_name = request.form["game_name"]
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        current_app.cm.saveGame(gameName=game_name)
        CONSOLE_OUTPUT = "Saved game state."
    return 'success'


@harmony.route('/')
def buildHarmony():
    if type(current_app.cm) is not HarmonyMachine:
        resetHarmony()
    template_name = current_app.config.get('HARMONY_TEMPLATE', 'Harmony.html')
    with open(f"{os.path.dirname(__file__)}/harmony_templates/{template_name}", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-info" value="Camera {camName}" onclick="gameWorldClick('{camName}')">''' for camName in current_app.cc.cameras.keys()])
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="gameWorldClick(\'VirtualMap\')">{cameraButtons}"""
    defaultCam = [camName for camName, cam in current_app.cc.cameras.items()]
    if len(defaultCam) == 0:
        defaultCam = "None"
    else:
        defaultCam = defaultCam[0]
        
    # Register new session
    new_view_id = str(uuid4())
    SESSIONS[new_view_id] = SessionConfig()
    
    return template.replace(
        "{viewId}", new_view_id).replace(
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
    moveDistance = current_app.cm.cc.trackedObjectLastDistance(capture)
    moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f}"
    changeRow = changeRowTemplate.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in current_app.cm.cc.changeSetToAxialCoord(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{observerURL}", url_for(".buildHarmony")).replace(
        "{encodedBA}", imageToBase64(current_app.cm.object_visual(capture)))
    return changeRow
    

def buildObjectTable():
    changeRows = []
    print(f"Structure object table")
    for capture in current_app.cm.memory:
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
        return objectId_endpoint(cap=current_app.cm.findObject(objectId=objectId), **kwargs)
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
        "{encodedBA}", imageToBase64(current_app.cm.object_visual(cap, withContours=footprint_enabled)))


@harmony.route('/objects/<objectId>', methods=['POST'])
@findObjectIdOr404
def updateObjectSettings(cap):
    # TODO: implement
    return buildObjectTable()
    
    
@harmony.route('/objects/<objectId>', methods=['DELETE'])
@findObjectIdOr404
def deleteObjectSettings(cap):
    with DATA_LOCK:
        current_app.cm.deleteObject(cap.oid)
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
    selectedCell = SESSIONS[viewId].selection.firstCell
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
    selectedAxial = SESSIONS[viewId].selection.firstCell
    trackedObject = current_app.cm.cc.define_object_from_axial(objectName, *selectedAxial)
    current_app.cm.commitChanges(trackedObject)
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
    selected = SESSIONS[viewId].selection
    overlap = None
    for mem in current_app.cm.memory:
        mem_axial = current_app.cm.cc.changeSetToAxialCoord(mem)
        if mem_axial == selected.firstCell:
            overlap = mem
            break
    current_app.cm.memory.remove(mem)
    SESSIONS[viewId].selection = CellSelection()
    return "Success"


@harmony.route('/request_move/<oid>/<viewId>', methods=['GET'])
def moveObjectDefinition(oid, viewId):
    try:
        session = SESSIONS[viewId]
        selected = session.selection 
        firstCell = selected.firstCell
        secondCell = selected.additionalCells[0]
    except Exception as e:
        raise Exception("500") from e

    if oid not in session.moveable:
        raise Exception("403")

    trackedObject = current_app.cm.cc.define_object_from_axial(oid, *secondCell)
    existing = current_app.cm.cc.define_object_from_axial(oid, *firstCell)
    with DATA_LOCK:
        current_app.cm.memory.remove(existing)
        current_app.cm.commitChanges(trackedObject)
        SESSIONS[viewId].selection = CellSelection()
    return "Success"


@harmony.route('/clear_pixel/<viewId>', methods=['GET'])
def clearPixel(viewId):
    if viewId in SESSIONS:
        with DATA_LOCK:
            SESSIONS[viewId].selection = CellSelection()
    return ""

@harmony.route('/select_pixel', methods=['POST'])
def selectPixel():
    global SESSIONS
    
    viewId = request.form["viewId"]
    pixel = json.loads(request.form["selectedPixel"])
    x, y = pixel
    cam = request.form["selectedCamera"]
    appendPixel = bool(request.form["appendPixel"])
    if cam == "VirtualMap":
        axial_coord = current_app.cm.cc.pixel_to_axial(x, y)
    else:
        axial_coord = current_app.cm.cc.camCoordToAxial(cam, (x, y))
    print(f"viewId {viewId} || Received: Pixel {pixel} on Cam {cam} || Translated to Axial: {axial_coord}")
    with DATA_LOCK:
        existing = SESSIONS.get(viewId, SessionConfig()).selection
        if existing.firstCell:
            if existing.secondCell is None:
                SESSIONS[viewId].selection.additionalCells = [axial_coord]
            elif appendPixel:
                SESSIONS[viewId].selection.additionalCells.insert(0, axial_coord)
            else:
                SESSIONS[viewId].selection = CellSelection(viewId, axial_coord)
        else:
            SESSIONS[viewId].selection = CellSelection(firstCell=axial_coord)
    q, r = axial_coord

    overlaps = []
    targets = []
    selected = SESSIONS[viewId].selection
    for mem in current_app.cm.memory:
        mem_axial = current_app.cm.cc.changeSetToAxialCoord(mem)
        if mem_axial == selected.firstCell:
            overlaps.append(mem)
        if mem_axial == selected.secondCell:
            targets.append(mem)

    if overlaps:
        overlap = overlaps[0]
        if selected.secondCell:
            distance = current_app.cm.cc.axial_distance(selected.firstCell, selected.secondCell)
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
            distance = current_app.cm.cc.axial_distance(selected.firstCell, selected.secondCell)
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


def minimapGenerator():
    while True:
        camImage = current_app.cm.buildMiniMap()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/minimap/<viewId>')
def minimapResponse(viewId):
    return Response(stream_with_context(minimapGenerator()), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/control')
def session_control_list():
    """List all active sessions"""
    return render_template('SessionList.html', sessions=SESSIONS)


@harmony.route('/control/<viewId>', methods=['GET'])
def session_control_panel(viewId):
    """Control panel for a specific session"""
    if viewId not in SESSIONS:
        return f"Session {viewId} not found", 404
        
    return render_template('ControlPanel.html', 
                         viewId=viewId, 
                         config=SESSIONS[viewId], 
                         objects=current_app.cm.memory)


@harmony.route('/control/<viewId>/update', methods=['POST'])
def update_session_config(viewId):
    """Update configuration for a specific session"""
    if viewId not in SESSIONS:
        return f"Session {viewId} not found", 404
        
    # Rebuild config from form data
    # Form data will have keys like "selectable_OID", "enemy_OID", etc.
    # or better: "OID_selectable", "OID_enemy"

    old_config = SESSIONS[viewId]
    new_config = SessionConfig()
    new_config.selection = old_config.selection
    
    # Iterate over all known objects to check their status in the form
    for obj in current_app.cm.memory:
        oid = obj.oid
        if request.form.get(f"{oid}_selectable"):
            new_config.selectable.append(oid)
        if request.form.get(f"{oid}_terrain"):
            new_config.terrain.append(oid)
        if request.form.get(f"{oid}_targetable"):
            new_config.targetable.append(oid)
        if request.form.get(f"{oid}_enemies"):
            new_config.enemies.append(oid)
        if request.form.get(f"{oid}_allies"):
            new_config.allies.append(oid)
        if request.form.get(f"{oid}_moveable"):
            new_config.moveable.append(oid)
            
    SESSIONS[viewId] = new_config
    return redirect(url_for('.session_control_panel', viewId=viewId))



def setHarmonyApp(newApp):
    global app
    app = newApp


def create_harmony_app(template_name="Harmony.html"):
    global app
    
    app = Flask(__name__)
    APPS.append(app)
    app.config['HARMONY_TEMPLATE'] = template_name
    app.cc = HexCaptureConfiguration()
    if app.cc.hex is None:
        app.cc.hex = HexGridConfiguration()
    app.cc.capture()
    app.cm = HarmonyMachine(app.cc)
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.register_blueprint(harmony, url_prefix='/harmony')
    with app.app_context():
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


def start_servers():
    global app
    APPS.clear()
    
    # Initialize shared state
    cc = HexCaptureConfiguration()
    if cc.hex is None:
        cc.hex = HexGridConfiguration()
    cc.capture()
    cm = HarmonyMachine(cc)
    
    # Helper to create configured app sharing state
    def make_app(template_name):
        new_app = Flask(__name__)
        new_app.config['HARMONY_TEMPLATE'] = template_name
        new_app.cc = cc
        new_app.cm = cm
        
        # Register blueprints
        new_app.register_blueprint(configurator, url_prefix='/configurator')
        new_app.register_blueprint(harmony, url_prefix='/harmony')
        
        # Register routes
        @new_app.route('/')
        def index():
            return redirect('/harmony', code=303)
            
        @new_app.route('/bootstrap.min.css', methods=['GET'])
        def getBSCSS():
            with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
                bscss = f.read()
            return Response(bscss, mimetype="text/css")
        
        @new_app.route('/bootstrap.min.js', methods=['GET'])
        def getBSJS():
            with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
                bsjs = f.read()
            return Response(bsjs, mimetype="application/javascript")
        
        @new_app.route('/htmx.min.js', methods=['GET'])
        def getHTMX():
            with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
                htmx = f.read()
            return Response(htmx, mimetype="application/javascript")
            
        registerCaptureService(new_app)
        APPS.append(new_app)
        return new_app

    # Create Apps
    admin_app = make_app("Harmony.html")
    user_app = make_app("HarmonyUser.html")
    
    app = admin_app # For legacy external usage if any
    
    setConfiguratorApp(admin_app) 
    setObserverApp(admin_app)
    
    # Launch User Server in Thread
    def run_user():
        print("Launching Harmony User Server on 7001")
        user_app.run(host="0.0.0.0", port=7001, use_reloader=False)

    t = threading.Thread(target=run_user)
    t.daemon = True
    t.start()
    
    # Launch Admin Server
    print("Launching Harmony Server on 7000")
    admin_app.run(host="0.0.0.0", port=7000, use_reloader=False)

if __name__ == "__main__":
    start_servers()

