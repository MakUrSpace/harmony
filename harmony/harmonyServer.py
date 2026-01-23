

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
import os
import shutil
import pickle
import random
from collections import defaultdict

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
    from ipynb.fs.full.Observer import hStackImages, clipImage
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

def get_scale_factor(cam_name):
    try:
        if cam_name == "VirtualMap":
            return 1.0, 1.0
        cam = current_app.cc.cameras.get(cam_name)
        if cam is None or cam.mostRecentFrame is None:
            return 1.0, 1.0
        
        h, w = cam.mostRecentFrame.shape[:2]
        if w == 0 or h == 0:
            return 1.0, 1.0
            
        # perspective_res is global [1920, 1080]
        return perspective_res[0] / w, perspective_res[1] / h
    except Exception:
        return 1.0, 1.0

def scale_point(pt, scale):
    return (pt[0] * scale[0], pt[1] * scale[1])

def axial_to_ui_object(q, r):
    return {
        "VirtualMap": [
            safe_point(pt)
            for pt in current_app.cm.cc.hex_at_axial(q, r)],
        **{camName: [
            scale_point(safe_point(pt), get_scale_factor(camName))
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
                camName: [scale_point((pt[0], pt[1]), get_scale_factor(camName)) for pt in obj.changeSet[camName].changePoints]
                for camName in current_app.cc.cameras.keys()}
        }
    
    session_config = SESSIONS.get(viewId, SessionConfig())

    data = {
        "objects": objects,
        **asdict(session_config),
        "selection": {
            "firstCell": axial_to_ui_object(*session_config.selection.firstCell) if session_config.selection.firstCell is not None else None,
            "additionalCells": [axial_to_ui_object(*cell) for cell in session_config.selection.additionalCells]
        },
        "selectable": [obj.oid for obj in app.cm.memory]
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


@harmony.route('/save', methods=['POST'])
def saveHarmonyPost():
    gameName = request.form.get("game_name")
    if not gameName:
        return "Game name required", 400
    
    # Reuse existing save logic or call it directly
    return saveHarmony(gameName)


@harmony.route('/load', methods=['POST'])
def loadHarmonyPost():
    gameName = request.form.get("game_name")
    if not gameName:
        return "Game name required", 400
        
    return loadHarmony(gameName)


@harmony.route('/save_game/<gameName>')
def saveHarmony(gameName):
    global SESSIONS
    try:
        current_app.cm.saveGame(gameName)
        # Pickle memory and current sessions
        save_data = {
            'memory': current_app.cm.memory,
            'sessions': SESSIONS
        }
        with open(f"{gameName}.pickle", "wb") as f:
            pickle.dump(save_data, f)
            
        return f"Game saved as {gameName}"
    except Exception as e:
        print(f"Error saving game: {e}")
        return f"Error saving game: {e}"

@harmony.route('/load_game/<gameName>')
def loadHarmony(gameName):
    global SESSIONS
    try:
        # current_app.cm.loadGame(gameName) # Use manual pickle logic mainly
        
        pickle_path = f"{gameName}.pickle"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                load_data = pickle.load(f)
                
                # Restore Memory
                new_memory = load_data.get('memory', [])
                current_app.cm.memory = new_memory
                print(f"Loaded {len(new_memory)} objects into memory.")
                
                # Restore Sessions (Merge)
                loaded_sessions = load_data.get('sessions', {})
                # We do NOT clear, we update. 
                # This ensures the current user's session remains valid, 
                # or if there's a conflict, the saved one takes precedence (which is usually desired for load)
                # But to be safe for a NEW session after restart, we want to keep the new one active.
                # Actually, simply updating adds the old sessions back. The current session viewId is distinct.
                SESSIONS.update(loaded_sessions)
                print(f"Merged {len(loaded_sessions)} sessions from save. Total sessions: {len(SESSIONS)}")
                
        else:
            return f"Save file {gameName}.pickle not found"
        
        return f"Game {gameName} loaded. Objects: {len(current_app.cm.memory)}" 
    except Exception as e:
        print(f"Error loading game: {e}")
        return f"Error loading game: {e}"
        CONSOLE_OUTPUT = "Saved game state."
    return 'success'



ADJECTIVES = ["Cool", "Happy", "Fast", "Shiny", "Blue", "Red", "Green", "Bright", "Dark", "Loud", "Quiet", "Brave", "Calm", "Eager", "Fair", "Gentle", "Jolly", "Kind", "Lively", "Nice", "Proud", "Silly", "Witty", "Zealous"]
NOUNS = ["Tiger", "Eagle", "Shark", "Bear", "Lion", "Wolf", "Fox", "Hawk", "Owl", "Frog", "Toad", "Fish", "Crab", "Star", "Moon", "Sun", "Cloud", "Rain", "Snow", "Wind", "Storm", "River", "Lake", "Sea", "Ocean"]

def simple_id_generator():
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"

@harmony.route('/update_session_id', methods=['POST'])
def update_session_id():
    old_id = request.form.get('viewId')
    new_id = request.form.get('newViewId')
    
    if not old_id or not new_id:
        return "Invalid Request", 400
        
    if new_id in SESSIONS:
        # Reclaiming session: redirect to new_id
        return f"""<script>window.location.href = "{url_for('.buildHarmony')}?viewId={new_id}";</script>"""

    with DATA_LOCK:
        if old_id in SESSIONS:
            SESSIONS[new_id] = SESSIONS.pop(old_id)
            return f"""<script>window.location.href = "{url_for('.buildHarmony')}?viewId={new_id}";</script>"""
        else:
            return "Session not found", 404

@harmony.route('/')
def buildHarmony():
    if type(current_app.cm) is not HarmonyMachine:
        resetHarmony()
    template_name = current_app.config.get('HARMONY_TEMPLATE', 'Harmony.html')
    with open(f"{os.path.dirname(__file__)}/harmony_templates/{template_name}", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-info" value="Camera {camName}" onclick="gameWorldClick('{camName}')">''' for camName in current_app.cc.cameras.keys()])
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="gameWorldClick('VirtualMap')">{cameraButtons}"""
    defaultCam = [camName for camName, cam in current_app.cc.cameras.items()]
    if len(defaultCam) == 0:
        defaultCam = "None"
    else:
        defaultCam = defaultCam[0]
        
    # Check for existing session
    view_id = request.args.get('viewId')
    
    # Priority: 1. Query Param, 2. Cookie
    if not view_id:
        view_id = request.cookies.get('session_view_id')

    if view_id and view_id in SESSIONS:
        # Resume session
        pass
    elif view_id:
        # Register requested session (Deep link / param override calling for new session)
        SESSIONS[view_id] = SessionConfig()
    else:
        # Register new session
        # Ensure uniqueness
        while True:
            view_id = simple_id_generator()
            if view_id not in SESSIONS:
                break
        SESSIONS[view_id] = SessionConfig()
    
    rendered = template.replace(
        "{viewId}", view_id).replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{harmonyURL}", url_for('.buildHarmony')).replace(
        "{configuratorURL}", '/configurator')
        
    resp = make_response(rendered)
    cookie_val = request.cookies.get('session_view_id')
    if cookie_val != view_id:
        resp.set_cookie('session_view_id', view_id)
        
    return resp




# Colors in BGR format to match Harmony.html RGB definitions
GROUP_COLORS = {
    "moveable": (255, 80, 170),   # RGB(170, 80, 255) -> BGR
    "allies": (120, 210, 0),      # RGB(0, 210, 120) -> BGR
    "enemies": (70, 60, 230),     # RGB(230, 60, 70) -> BGR
    "targetable": (255, 200, 0),  # RGB(0, 200, 255) -> BGR
    "terrain": (135, 125, 120),   # RGB(120, 125, 135) -> BGR
    "selectable": (205, 190, 180) # RGB(180, 190, 205) -> BGR
}

def custom_object_visual(cm, changeSet, color, margin=0):
    cameras = cm.cc.cameras
    if changeSet.empty:
        return np.zeros([10, 10], dtype="float32")

    images = {cam: change.after for cam, change in changeSet.changeSet.items() if change.changeType not in ["delete", None]}
    
    maxHeight =  max([im.shape[0] + margin * 2 for im in images.values()])
    filler = np.zeros((maxHeight, 50, 3), np.uint8)

    margins = [-margin, -margin, margin * 2, margin * 2]

    for camName, change in changeSet.changeSet.items():
        if change.changeType == "delete":
            images[camName] = filler
        else:
            # Always with contours for this visual, but using custom color
            images[camName] = clipImage(cv2.addWeighted(
                    cameras[camName].mostRecentFrame.copy(),
                    0.6,
                    cv2.drawContours(
                        cameras[camName].mostRecentFrame.copy(),
                        change.changeContours, 
                        -1,
                        color, # Custom color here
                        -1
                    ),
                    0.4,
                    0
                ),
                [dim + m for dim, m in zip(change.clipBox, margins)]
            )

    return hStackImages(images.values())


# Load the template once globally or pass it around if it's static
with open(f"{os.path.dirname(__file__)}/harmony_templates/TrackedObjectRow.html") as f:
    _TRACKED_OBJECT_ROW_TEMPLATE = f.read()

def captureToChangeRow(capture, color=None):
    moveDistance = current_app.cm.cc.trackedObjectLastDistance(capture)
    moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f}"
    
    if color is not None:
        visual_image = custom_object_visual(current_app.cm, capture, color)
    else:
        visual_image = current_app.cm.object_visual(capture)

    changeRow = _TRACKED_OBJECT_ROW_TEMPLATE.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in current_app.cm.cc.changeSetToAxialCoord(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{observerURL}", url_for(".buildHarmony")).replace(
        "{encodedBA}", imageToBase64(visual_image))
    return changeRow
    

def buildObjectTable(viewId=None):
    changeRows = []
    print(f"Structure object table for viewId: {viewId}")
    
    seen_oids = set()
    
    if viewId and viewId in SESSIONS:
        session = SESSIONS[viewId]
        # Priority order: moveable, allies, enemies, targetable, terrain, selectable
        groups = [
            ("moveable", session.moveable),
            ("allies", session.allies),
            ("enemies", session.enemies),
            ("targetable", session.targetable),
            ("terrain", session.terrain)
        ]
        
        for group_name, oids in groups:
            group_rows = []
            color = GROUP_COLORS.get(group_name)
            for oid in oids:
                if oid not in seen_oids:
                    # Find the object in memory
                    for capture in current_app.cm.memory:
                        if capture.oid == oid:
                            group_rows.append(captureToChangeRow(capture, color))
                            seen_oids.add(oid)
                            break
            if group_rows:
                changeRows.append(f"<h4>{group_name.capitalize()}</h4>" + " ".join(group_rows))
                            
    # 'selectable' (all remaining objects)
    selectable_rows = []
    selectable_color = GROUP_COLORS.get("selectable")
    for capture in current_app.cm.memory:
        if capture.oid not in seen_oids:
            selectable_rows.append(captureToChangeRow(capture, selectable_color))
            seen_oids.add(capture.oid)
            
    if selectable_rows:
        changeRows.append(f"<h4>Selectable</h4>" + " ".join(selectable_rows))
            
    return " ".join(changeRows)


@harmony.route('/objects', methods=['GET'])
def getObjectTable():
    viewId = request.args.get('viewId')
    return buildObjectTable(viewId)


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
            elif appendPixel and axial_coord not in SESSIONS[viewId].selection.additionalCells:
                SESSIONS[viewId].selection.additionalCells.insert(0, axial_coord)
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
        # Check additional cells for objects
        additional_info = ""
        if selected.additionalCells:
            count = len(selected.additionalCells)
            additional_info += f"<h3>Additional cells: {count - 1}</h3>"
            
            # Iterate through additional cells to find objects
            for cell in selected.additionalCells:
                cell_objects = []
                for mem in current_app.cm.memory:
                    mem_axial = current_app.cm.cc.changeSetToAxialCoord(mem)
                    if mem_axial == cell:
                        cell_objects.append(mem.oid)
                
                if cell_objects:
                    additional_info += f"<div>Cell {cell}: Object(s) {', '.join(cell_objects)}</div>"

        if selected.secondCell:
            distance = current_app.cm.cc.axial_distance(selected.firstCell, selected.secondCell)
            distance = f"{distance} {'cell' if abs(distance) == 1 else 'cells'}"
            return interactor_template.format(info=f"""
                    <h2>Selected cell: {selected.firstCell}</h2>
                    {additional_info}
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

        @new_app.route('/HarmonyTemplate.css', methods=['GET'])
        def getHarmonyCSS():
            with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyTemplate.css", "r") as f:
                css = f.read()
            return Response(css, mimetype="text/css")
            
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

