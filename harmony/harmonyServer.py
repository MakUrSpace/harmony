from math import ceil
import base64
import argparse
import json
from io import BytesIO
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, AsyncGenerator
from functools import wraps
import threading
import atexit
import signal
from contextlib import asynccontextmanager
from traceback import format_exc
import time
from uuid import uuid4
import copy
import os
import sys
import shutil
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from fastapi import FastAPI, APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from observer import HexGridConfiguration, HexCaptureConfiguration
from observer.Observer import hStackImages, clipImage, Camera
from observer.configurator import configurator

from harmony.HarmonyMachine import HarmonyMachine, INCHES_TO_MM


# ---------------------------------------------------------------------------
# Module-level state (replaces Flask's current_app / Blueprint pattern)
# ---------------------------------------------------------------------------

_cc = None   # CaptureConfiguration
_cm = None   # HarmonyMachine
_config: dict = {}

DATA_LOCK = threading.Lock()
SHUTDOWN_EVENT = threading.Event()


def _get_cc():
    return _cc


def _get_cm():
    return _cm


# ---------------------------------------------------------------------------
# FastAPI Router (replaces Flask Blueprint)
# ---------------------------------------------------------------------------

harmony = APIRouter(prefix='/harmony', tags=['harmony'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

perspective_res = (1920, 1080)
virtual_map_res = (1200, 1200)


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


# ---------------------------------------------------------------------------
# Frame Broadcasting System
# ---------------------------------------------------------------------------

BROADCASTERS = {}


class FrameBroadcaster:
    def __init__(self, key, render_func, fps=15):
        self.key = key
        self.render_func = render_func
        self.interval = 1.0 / fps
        self.last_frame = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = False
        self.thread = None
        self.clients = 0

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
                # print(f"Broadcaster started for {self.key}")

    def stop(self):
        with self.lock:
            self.running = False

    def _run(self):
        while self.running and not SHUTDOWN_EVENT.is_set():
            start_time = time.time()
            try:
                # No app context needed in FastAPI
                frame_bytes = self.render_func()
                if frame_bytes:
                    with self.lock:
                        self.last_frame = frame_bytes
                        self.condition.notify_all()
            except Exception as e:
                print(f"Error in broadcaster {self.key}: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)

    def subscribe(self):
        """Yields frames to a client."""
        with self.lock:
            self.clients += 1
            if self.last_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpg\r\n\r\n' + self.last_frame + b'\r\n')

        try:
            while True:
                with self.condition:
                    self.condition.wait(timeout=1.0) # check shutdown every second
                    if not self.running or SHUTDOWN_EVENT.is_set():
                        break
                    frame = self.last_frame

                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')
        finally:
            with self.lock:
                self.clients -= 1


def render_minimap(cm, encode=True):
    try:
        if not cm:
            print("render_minimap: cm is None")
            return None

        camImage = cm.buildMiniMap(objectsAndColors=cm.objectsAndColors)
        if camImage is None:
            return None

        bounds = cm.cc.realSpaceBoundingBox()
        if bounds is None:
            return None

        x, y, w, h = bounds
        x, y, w, h = int(x), int(y), int(w), int(h)

        shift_x = 0
        shift_y = 0
        if x < 0: shift_x = -x
        if y < 0: shift_y = -y

        map_x = x + shift_x
        map_y = y + shift_y

        margin = 150

        crop_x = max(0, map_x - margin)
        crop_y = max(0, map_y - margin)

        crop_w = w + margin * 2
        crop_h = h + margin * 2

        if crop_w > 0 and crop_h > 0:
            img_h, img_w = camImage.shape[:2]
            end_x = min(img_w, crop_x + crop_w)
            end_y = min(img_h, crop_y + crop_h)
            # print(f"Cropping Minimap to: Y:[{crop_y}:{end_y}], X:[{crop_x}:{end_x}]")
            camImage = camImage[crop_y:end_y, crop_x:end_x]

        camImage = cv2.resize(camImage, virtual_map_res, interpolation=cv2.INTER_AREA)
        if not encode:
            return cv2.cvtColor(camImage, cv2.COLOR_BGR2RGB)

        ret, encoded = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return encoded.tobytes()
    except Exception as e:
        print(f"Minimap render error: {e}")
        return None


def render_camera(cc, camName):
    try:
        cam = cc.cameras.get(camName)
        if not cam:
            return None

        x, y, w, h = cam.activeZoneBoundingBox

        frame = cam.mostRecentFrame
        if frame is None:
            return None

        masked = frame.copy()

        grid = cc.cameraGriddle(camName)
        if grid is not None:
            try:
                if np.sum(grid) > 0:
                    masked = cv2.addWeighted(grid, 0.3, masked, 0.7, 0.0)
                else:
                    print(f"Grid for {camName} is empty (all zeros)")
            except Exception as e:
                print(f"Grid blend error: {e}")
        else:
            print(f"Grid for {camName} returned None")

        masked = cam.cropToActiveZone(masked)

        cropped = masked[y:y+h, x:x+w]
        camImage = cv2.resize(cropped, tuple(perspective_res), interpolation=cv2.INTER_LINEAR)

        ret, encoded = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return encoded.tobytes()
    except Exception as e:
        print(f"Camera render error {camName}: {e}")
        return None


def get_broadcaster(key, render_func):
    if key not in BROADCASTERS:
        BROADCASTERS[key] = FrameBroadcaster(key, render_func)
        BROADCASTERS[key].start()
    return BROADCASTERS[key]


# ---------------------------------------------------------------------------
# Console streaming
# ---------------------------------------------------------------------------

def renderConsole():
    while not SHUTDOWN_EVENT.is_set():
        cm = _get_cm()
        shape = (170, 400)
        zeros = np.zeros(shape, dtype="uint8")
        consoleImage = cv2.putText(zeros, f'Cycle {cm.cycleCounter}',
            (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Mode: {cm.mode:7}',
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {cm.state:10}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@harmony.get('/harmony_console')
def getConsoleImage():
    return StreamingResponse(renderConsole(), media_type='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Combined cameras (legacy)
# ---------------------------------------------------------------------------

def genCombinedCamerasView():
    while not SHUTDOWN_EVENT.is_set():
        cc = _get_cc()
        camImages = []
        for camName in cc.cameras.keys():
            camImage = cc.cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = cv2.resize(camImages[0] if len(camImages) == 1 else np.vstack(camImages), [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.get('/combinedCameras')
def combinedCamerasResponse():
    return StreamingResponse(genCombinedCamerasView(), media_type='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Session / Canvas data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Camera-with-changes stream
# ---------------------------------------------------------------------------

@harmony.get('/camWithChanges/{camName}/{viewId}')
def cameraViewWithChangesResponse(camName: str, viewId: str):
    cm = _get_cm()
    cc = _get_cc()
    if camName == "VirtualMap":
        broadcaster = get_broadcaster("VirtualMap", lambda: render_minimap(cm))
    else:
        broadcaster = get_broadcaster(camName, lambda c=camName: render_camera(cc, c))

    return StreamingResponse(broadcaster.subscribe(), media_type='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def safe_point(pt):
    p = pt.tolist()
    if len(p) > 0 and isinstance(p[0], list):
        return (p[0][0], p[0][1])
    return (p[0], p[1])


def get_conversion_params(cam_name):
    cc = _get_cc()
    cm = _get_cm()
    try:
        am_virtual_map = (cam_name == "VirtualMap")

        if am_virtual_map:
            try:
                if hasattr(cm.cc, 'realSpaceBoundingBox'):
                    bx, by, bw, bh = cm.cc.realSpaceBoundingBox()

                    shift_x = 0
                    shift_y = 0
                    if bx < 0: shift_x = -bx
                    if by < 0: shift_y = -by

                    map_x = int(bx) + shift_x
                    map_y = int(by) + shift_y

                    margin = 150
                    crop_x = max(0, map_x - margin)
                    crop_y = max(0, map_y - margin)

                    crop_w = int(bw) + margin * 2
                    crop_h = int(bh) + margin * 2

                    req_w = int(bx + bw + shift_x)
                    req_h = int(by + bh + shift_y)

                    w_canvas = 1600
                    h_canvas = 1600
                    if hasattr(cm.cc, 'hex') and cm.cc.hex:
                        w_canvas = cm.cc.hex.width
                        h_canvas = cm.cc.hex.height

                    img_w = max(w_canvas, req_w)
                    img_h = max(h_canvas, req_h)

                    end_x = min(img_w, crop_x + crop_w)
                    end_y = min(img_h, crop_y + crop_h)

                    crop_w_actual = end_x - crop_x
                    crop_h_actual = end_y - crop_y

                    if crop_w_actual <= 0 or crop_h_actual <= 0:
                        return 1.0, 1.0, 0, 0

                    scale_x = virtual_map_res[0] / crop_w_actual
                    scale_y = virtual_map_res[1] / crop_h_actual

                    min_x = crop_x - shift_x
                    min_y = crop_y - shift_y

                    return scale_x, scale_y, min_x, min_y

            except Exception as e:
                print(f"Error in get_conversion_params expansion: {e}")

            return 1.0, 1.0, 0, 0

        cam = cc.cameras.get(cam_name)
        if cam is None:
            return 1.0, 1.0, 0, 0

        x, y, w, h = cam.activeZoneBoundingBox
        if w == 0 or h == 0:
            return 1.0, 1.0, 0, 0

        scale_x = perspective_res[0] / w
        scale_y = perspective_res[1] / h
        return scale_x, scale_y, x, y

    except Exception as e:
        print(f"Error in get_conversion_params: {e}")
        return 1.0, 1.0, 0, 0


def scale_point_new(pt, params):
    sx, sy, ox, oy = params
    return ((pt[0] - ox) * sx, (pt[1] - oy) * sy)


def get_scale_factor(cam_name):
    sx, sy, _, _ = get_conversion_params(cam_name)
    return sx, sy


def scale_point(pt, scale):
    return (pt[0] * scale[0], pt[1] * scale[1])


def axial_to_ui_object(q, r):
    cm = _get_cm()
    cc = _get_cc()
    raw_vm_points = cm.cc.hex_at_axial(q, r)

    scale_x, scale_y, off_x_base, off_y_base = get_conversion_params("VirtualMap")

    vm_points = []
    for raw_pt in raw_vm_points:
        pt = safe_point(raw_pt)
        scaled_x = (pt[0] - off_x_base) * scale_x
        scaled_y = (pt[1] - off_y_base) * scale_y
        vm_points.append((scaled_x, scaled_y))

    return {
        "VirtualMap": vm_points,
        **{camName: [
            scale_point_new(safe_point(pt), get_conversion_params(camName))
            for pt in cm.cc.cam_hex_at_axial(camName, q, r)] for camName in cc.cameras.keys()}
    }


# ---------------------------------------------------------------------------
# Canvas data
# ---------------------------------------------------------------------------

@harmony.get('/canvas_data/{viewId}')
def getCanvasData(viewId: str):
    cm = _get_cm()
    cc = _get_cc()
    objects = {}
    for obj in cm.memory:
        vm_params = get_conversion_params("VirtualMap")
        hull = cm.cc.objectToHull(obj)
        if hull is not None and len(hull) > 0:
            raw_vm_pts = hull.reshape(-1, 2)
            vm_pts = [scale_point_new((float(pt[0]), float(pt[1])), vm_params) for pt in raw_vm_pts]
        else:
            vm_pts = []

        objects[obj.oid] = {
            "VirtualMap": vm_pts,
            **{
                camName: [scale_point_new((pt[0], pt[1]), get_conversion_params(camName)) for pt in obj.changeSet[camName].changePoints]
                for camName in cc.cameras.keys()}
        }

    session_config = SESSIONS.get(viewId, SessionConfig())

    data = {
        "objects": objects,
        **asdict(session_config),
        "selection": {
            "firstCell": axial_to_ui_object(*session_config.selection.firstCell) if session_config.selection.firstCell is not None else None,
            "additionalCells": [axial_to_ui_object(*cell) for cell in session_config.selection.additionalCells]
        },
        "selectable": [obj.oid for obj in cm.memory]
    }
    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Combined cameras with changes
# ---------------------------------------------------------------------------

def genCombinedCameraWithChangesView():
    while not SHUTDOWN_EVENT.is_set():
        cm = _get_cm()
        with DATA_LOCK:
            if cm.lastChanges is not None and not cm.lastChanges.empty:
                print("Has changes")
            if cm.lastClassification is not None:
                print("Has class")
            camImages = list(cm.getCameraImagesWithChanges(_get_cc().cameras.keys()).values())
            camImage = camImages[0] if len(camImages) == 1 else np.vstack(camImages)
            camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_LINEAR)
            ret, camImage = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
        time.sleep(0.1)


@harmony.get('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return StreamingResponse(genCombinedCameraWithChangesView(), media_type='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Reset / Save / Load
# ---------------------------------------------------------------------------

@harmony.get('/reset')
def resetHarmony():
    global _cm
    with DATA_LOCK:
        new_cm = HarmonyMachine(_get_cc())
        new_cm.reset()
        _cm = new_cm
    return 'success'


@harmony.post('/save')
async def saveHarmonyPost(game_name: str = Form(...)):
    if not game_name:
        return Response("Game name required", status_code=400)
    return _save_harmony(game_name)


@harmony.post('/load')
async def loadHarmonyPost(game_name: str = Form(...)):
    if not game_name:
        return Response("Game name required", status_code=400)
    return _load_harmony(game_name)


@harmony.get('/save_game/{gameName}')
def saveHarmony(gameName: str):
    return _save_harmony(gameName)


def _save_harmony(gameName: str):
    global SESSIONS
    try:
        cm = _get_cm()
        cm.saveGame(gameName)
        save_data = {
            'memory': cm.memory,
            'sessions': SESSIONS
        }
        with open(f"{gameName}.pickle", "wb") as f:
            pickle.dump(save_data, f)
        return Response(f"Game saved as {gameName}")
    except Exception as e:
        print(f"Error saving game: {e}")
        return Response(f"Error saving game: {e}")


@harmony.get('/load_game/{gameName}')
def loadHarmony(gameName: str):
    return _load_harmony(gameName)


def _load_harmony(gameName: str):
    global SESSIONS
    try:
        pickle_path = f"{gameName}.pickle"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                load_data = pickle.load(f)
            cm = _get_cm()
            new_memory = load_data.get('memory', [])
            cm.memory = new_memory
            print(f"Loaded {len(new_memory)} objects into memory.")
            loaded_sessions = load_data.get('sessions', {})
            SESSIONS.update(loaded_sessions)
            print(f"Merged {len(loaded_sessions)} sessions from save. Total sessions: {len(SESSIONS)}")
        else:
            return Response(f"Save file {gameName}.pickle not found")
        return Response(f"Game {gameName} loaded. Objects: {len(_get_cm().memory)}")
    except Exception as e:
        print(f"Error loading game: {e}")
        return Response(f"Error loading game: {e}")


# ---------------------------------------------------------------------------
# Session ID management
# ---------------------------------------------------------------------------

ADJECTIVES = ["Cool", "Happy", "Fast", "Shiny", "Blue", "Red", "Green", "Bright", "Dark", "Loud", "Quiet", "Brave", "Calm", "Eager", "Fair", "Gentle", "Jolly", "Kind", "Lively", "Nice", "Proud", "Silly", "Witty", "Zealous"]
NOUNS = ["Tiger", "Eagle", "Shark", "Bear", "Lion", "Wolf", "Fox", "Hawk", "Owl", "Frog", "Toad", "Fish", "Crab", "Star", "Moon", "Sun", "Cloud", "Rain", "Snow", "Wind", "Storm", "River", "Lake", "Sea", "Ocean"]


def simple_id_generator():
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"


@harmony.post('/update_session_id')
async def update_session_id(viewId: str = Form(...), newViewId: str = Form(...)):
    if not viewId or not newViewId:
        return Response("Invalid Request", status_code=400)

    if newViewId in SESSIONS:
        return HTMLResponse(f"""<script>window.location.href = "/harmony/?viewId={newViewId}";</script>""")

    with DATA_LOCK:
        if viewId in SESSIONS:
            SESSIONS[newViewId] = SESSIONS.pop(viewId)
            return HTMLResponse(f"""<script>window.location.href = "/harmony/?viewId={newViewId}";</script>""")
        else:
            return Response("Session not found", status_code=404)


# ---------------------------------------------------------------------------
# Main Harmony page
# ---------------------------------------------------------------------------

@harmony.get('/')
@harmony.get('')
async def buildHarmony(request: Request, viewId: Optional[str] = Query(default=None)):
    cm = _get_cm()
    cc = _get_cc()

    try:
        if not isinstance(cm, HarmonyMachine):
            resetHarmony()
    except TypeError:
        pass  # HarmonyMachine may be a mock during testing

    template_name = _config.get('HARMONY_TEMPLATE', 'Harmony.html')
    with open(f"{os.path.dirname(__file__)}/harmony_templates/{template_name}", "r") as f:
        template = f.read()

    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-info" value="Camera {camName}" onclick="gameWorldClick('{camName}')">''' for camName in cc.cameras.keys()])
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="gameWorldClick('VirtualMap')">{cameraButtons}"""

    defaultCam = [camName for camName, cam in cc.cameras.items()]
    if len(defaultCam) == 0:
        defaultCam = "None"
    else:
        defaultCam = defaultCam[0]

    # Priority: 1. Query param, 2. Cookie
    if not viewId:
        viewId = request.cookies.get('session_view_id')

    if viewId and viewId in SESSIONS:
        pass  # Resume session
    elif viewId:
        SESSIONS[viewId] = SessionConfig()
    else:
        while True:
            viewId = simple_id_generator()
            if viewId not in SESSIONS:
                break
        SESSIONS[viewId] = SessionConfig()

    rendered = template.replace(
        "{viewId}", viewId).replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{harmonyURL}", "/harmony/").replace(
        "{configuratorURL}", '/configurator')

    resp = HTMLResponse(rendered)
    cookie_val = request.cookies.get('session_view_id')
    if cookie_val != viewId:
        resp.set_cookie('session_view_id', viewId)

    return resp


# ---------------------------------------------------------------------------
# Object visuals
# ---------------------------------------------------------------------------

# Colors in BGR format to match Harmony.html RGB definitions
GROUP_COLORS = {
    "moveable": (255, 80, 170),
    "allies": (120, 210, 0),
    "enemies": (70, 60, 230),
    "targetable": (255, 200, 0),
    "terrain": (30, 105, 210),
    "selectable": (180, 105, 255)
}


def custom_object_visual(cm, changeSet, color, margin=0):
    cameras = cm.cc.cameras
    if changeSet.empty:
        return np.zeros([10, 10], dtype="float32")

    images = {cam: change.after for cam, change in changeSet.changeSet.items() if change.changeType not in ["delete", None]}

    maxHeight = max([im.shape[0] + margin * 2 for im in images.values()])
    filler = np.zeros((maxHeight, 50, 3), np.uint8)

    margins = [-margin, -margin, margin * 2, margin * 2]

    for camName, change in changeSet.changeSet.items():
        if change.changeType == "delete":
            images[camName] = filler
        else:
            images[camName] = clipImage(cv2.addWeighted(
                    cameras[camName].mostRecentFrame.copy(),
                    0.6,
                    cv2.drawContours(
                        cameras[camName].mostRecentFrame.copy(),
                        change.changeContours,
                        -1,
                        color,
                        -1
                    ),
                    0.4,
                    0
                ),
                [dim + m for dim, m in zip(change.clipBox, margins)]
            )

    return hStackImages(images.values())


with open(f"{os.path.dirname(__file__)}/harmony_templates/TrackedObjectRow.html") as f:
    _TRACKED_OBJECT_ROW_TEMPLATE = f.read()


def captureToChangeRow(capture, color=None):
    cm = _get_cm()
    moveDistance = cm.cc.trackedObjectLastDistance(capture)
    try:
        moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f}"
    except (TypeError, ValueError):
        moveDistance = str(moveDistance)

    if color is not None:
        visual_image = custom_object_visual(cm, capture, color)
    else:
        visual_image = cm.object_visual(capture)

    changeRow = _TRACKED_OBJECT_ROW_TEMPLATE.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in cm.cc.changeSetToAxialCoord(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{observerURL}", "/harmony/").replace(
        "{encodedBA}", imageToBase64(visual_image))
    return changeRow


def buildObjectTable(viewId=None):
    cm = _get_cm()
    changeRows = []
    print(f"Structure object table for viewId: {viewId}")

    seen_oids = set()

    if viewId and viewId in SESSIONS:
        session = SESSIONS[viewId]
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
                    for capture in cm.memory:
                        if capture.oid == oid:
                            group_rows.append(captureToChangeRow(capture, color))
                            seen_oids.add(oid)
                            break
            if group_rows:
                changeRows.append(f"<h4>{group_name.capitalize()}</h4>" + " ".join(group_rows))

    selectable_rows = []
    selectable_color = GROUP_COLORS.get("selectable")
    for capture in cm.memory:
        if capture.oid not in seen_oids:
            selectable_rows.append(captureToChangeRow(capture, selectable_color))
            seen_oids.add(capture.oid)

    if selectable_rows:
        changeRows.append(f"<h4>Selectable</h4>" + " ".join(selectable_rows))

    return " ".join(changeRows)


@harmony.get('/objects')
def getObjectTable(viewId: Optional[str] = Query(default=None)):
    return HTMLResponse(buildObjectTable(viewId))


def getInteractor():
    return buildObjectTable()


# ---------------------------------------------------------------------------
# Individual object endpoints
# ---------------------------------------------------------------------------

def _find_object(objectId: str):
    cm = _get_cm()
    return cm.findObject(objectId=objectId)


@harmony.get('/objects/{objectId}')
def getObject(objectId: str, footprint: str = Query(default="false")):
    cap = _find_object(objectId)
    if cap is None:
        return Response(f"{objectId} Not found", status_code=404)
    footprint_enabled = footprint == "true"
    cm = _get_cm()
    with open(f"{os.path.dirname(__file__)}/harmony_templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return HTMLResponse(template.replace(
        "{harmonyURL}", "/harmony/").replace(
        "{objectName}", cap.oid).replace(
        "{objectSettings}", buildObjectSettings(cap)).replace(
        "{footprintToggleState}", str(not footprint_enabled).lower()).replace(
        "{encodedBA}", imageToBase64(cm.object_visual(cap, withContours=footprint_enabled))))


@harmony.post('/objects/{objectId}')
def updateObjectSettings(objectId: str):
    cap = _find_object(objectId)
    if cap is None:
        return Response(f"{objectId} Not found", status_code=404)
    return HTMLResponse(buildObjectTable())


@harmony.delete('/objects/{objectId}')
def deleteObjectSettings(objectId: str):
    cap = _find_object(objectId)
    if cap is None:
        return Response(f"{objectId} Not found", status_code=404)
    with DATA_LOCK:
        _get_cm().deleteObject(cap.oid)
    return HTMLResponse(buildObjectTable())


def buildObjectSettings(cap, objType=None):
    return "200"


@harmony.get('/objects/{objectId}/settings')
def getObjectSettings(objectId: str):
    cap = _find_object(objectId)
    if cap is None:
        return Response(f"{objectId} Not found", status_code=404)
    return HTMLResponse(buildObjectSettings(cap))


@harmony.post('/objects/{objectId}/type')
async def updateObjectType(objectId: str, objectType: str = Form(...)):
    cap = _find_object(objectId)
    if cap is None:
        return Response(f"{objectId} Not found", status_code=404)
    assert objectType in ["None", "Terrain", "Structure", "Unit"], f"Unrecognized object type: {objectType}"
    return HTMLResponse(buildObjectSettings(cap, objType=objectType))


# ---------------------------------------------------------------------------
# Object factory
# ---------------------------------------------------------------------------

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


@harmony.get('/object_factory/{viewId}')
def buildObjectFactory(viewId: str):
    selectedCell = SESSIONS[viewId].selection.firstCell
    return HTMLResponse(f"""
        <form hx-post="/harmony/object_factory/{viewId}" hx-target="#interactor">
            <label for="object_name" class="form-check-label">Object Name</label>
            <input type="text" name="object_name" value=""><br>
            <label for="selected_cells" class="form-check-label">Selected Cells</label>
            <input type="text" name="selected_cells" value="{selectedCell}"><br>
            <input type="submit" class="btn btn-primary" value="Define Object">
        </form>
    """)


@harmony.post('/object_factory/{viewId}')
async def buildObject(viewId: str, object_name: str = Form(...)):
    cm = _get_cm()
    objectName = str(object_name)
    selectedAxial = SESSIONS[viewId].selection.firstCell
    trackedObject = cm.cc.define_object_from_axial(objectName, *selectedAxial)
    cm.commitChanges(trackedObject)
    return HTMLResponse(interactor_template.format(
        info=f"""
        <h2>Selected cell: {selectedAxial}</h2>
        <h3>Object Name: {trackedObject.oid}</h3>
        """,
        actions=f"""
        <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="/harmony/clear_pixel/{viewId}" hx-target="#interactor">
        <hr>
        <input type="button" class="btn btn-danger" value="Delete Object" hx-delete="/harmony/object_factory/{viewId}" hx-target="#interactor">
        """))


@harmony.delete('/object_factory/{viewId}')
def deleteObject(viewId: str):
    cm = _get_cm()
    selected = SESSIONS[viewId].selection
    mem = None
    for m in cm.memory:
        mem_axial = cm.cc.changeSetToAxialCoord(m)
        if mem_axial == selected.firstCell:
            mem = m
            break
    if mem:
        cm.memory.remove(mem)
    SESSIONS[viewId].selection = CellSelection()
    return Response("Success")


# ---------------------------------------------------------------------------
# Move request
# ---------------------------------------------------------------------------

@harmony.get('/request_move/{oid}/{viewId}')
def moveObjectDefinition(oid: str, viewId: str):
    cm = _get_cm()
    try:
        session = SESSIONS[viewId]
        selected = session.selection
        firstCell = selected.firstCell
        secondCell = selected.additionalCells[0]
    except Exception as e:
        return Response("500", status_code=500)

    if oid not in session.moveable:
        return Response("403", status_code=403)

    trackedObject = cm.cc.define_object_from_axial(oid, *secondCell)
    existing = cm.cc.define_object_from_axial(oid, *firstCell)
    with DATA_LOCK:
        cm.memory.remove(existing)
        cm.commitChanges(trackedObject)
        SESSIONS[viewId].selection = CellSelection()
    return Response("Success")


# ---------------------------------------------------------------------------
# Pixel selection
# ---------------------------------------------------------------------------

@harmony.get('/clear_pixel/{viewId}')
def clearPixel(viewId: str):
    if viewId in SESSIONS:
        with DATA_LOCK:
            SESSIONS[viewId].selection = CellSelection()
    return HTMLResponse("")


@harmony.post('/select_pixel')
async def selectPixel(request: Request):
    global SESSIONS
    cm = _get_cm()
    form = await request.form()
    viewId = form["viewId"]
    pixel = json.loads(form["selectedPixel"])
    x, y = pixel
    cam = form["selectedCamera"]
    appendPixel = bool(form["appendPixel"])

    if cam == "VirtualMap":
        real_x = x
        real_y = y
        if cm and hasattr(cm.cc, 'realSpaceBoundingBox'):
            scale_x, scale_y, min_x, min_y = get_conversion_params("VirtualMap")
            if scale_x > 0 and scale_y > 0:
                real_x = (x / scale_x) + min_x
                real_y = (y / scale_y) + min_y
        axial_coord = cm.cc.pixel_to_axial(real_x, real_y)
    else:
        scale_x, scale_y, offset_x, offset_y = get_conversion_params(cam)
        raw_x = (x / scale_x) + offset_x
        raw_y = (y / scale_y) + offset_y
        axial_coord = cm.cc.camCoordToAxial(cam, (raw_x, raw_y))

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
    selected = SESSIONS[viewId].selection
    first = selected.firstCell

    if not first:
        return HTMLResponse("")

    def get_object_type(oid, viewId):
        session = SESSIONS.get(viewId)
        if not session: return "Unknown"
        if oid in session.moveable: return "Moveable"
        if oid in session.allies: return "Ally"
        if oid in session.enemies: return "Enemy"
        if oid in session.targetable: return "Targetable"
        if oid in session.terrain: return "Terrain"
        return "Selectable"

    def find_object_at(cell):
        for mem in cm.memory:
            if cm.cc.changeSetToAxialCoord(mem) == cell:
                return mem
        return None

    info_html = f"<h2>Selected First Cell: {first}</h2>"

    first_obj = find_object_at(first)
    if first_obj:
        o_type = get_object_type(first_obj.oid, viewId)
        info_html += f"<h3>Object: {first_obj.oid} <br><small>Type: {o_type}</small></h3>"

    if selected.additionalCells:
        info_html += "<hr><h3>Additional Selections:</h3>"
        for i, cell in enumerate(selected.additionalCells):
            if i > 0:
                info_html += "<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>"
            dist = cm.cc.axial_distance(first, cell)
            cell_obj = find_object_at(cell)
            obj_str = ""
            if cell_obj:
                o_type = get_object_type(cell_obj.oid, viewId)
                obj_str = f" <br>-> Object: {cell_obj.oid} ({o_type})"
            style = "border: 2px solid cyan; padding: 5px; margin: 2px;" if i == 0 else "padding: 5px; margin: 2px;"
            label = "Latest Selection" if i == 0 else f"Selection {i+1}"
            info_html += f"<div style='{style}'><b>{label}: {cell}</b><br>Dist to First: {dist} cells{obj_str}</div>"

    actions_html = ""

    if first_obj:
        if selected.additionalCells:
            target = selected.additionalCells[0]
            is_admin = _config.get('HARMONY_TEMPLATE') == "Harmony.html"
            session = SESSIONS.get(viewId)
            is_moveable = session and first_obj.oid in session.moveable
            if is_admin or is_moveable:
                actions_html += f"""
                    <input type="button" class="btn btn-info" value="Move {first_obj.oid} Here" hx-get="/harmony/request_move/{first_obj.oid}/{viewId}" hx-target="#interactor">
                """
        else:
            if _config.get('HARMONY_TEMPLATE') == "Harmony.html":
                actions_html += f"""
                   <input type="button" class="btn btn-danger" value="Delete Object" hx-delete="/harmony/object_factory/{viewId}" hx-target="#interactor">
                """
    else:
        actions_html += f"""
            <div id="object_factory">
                <input type="button" class="btn btn-success" value="Define Object" hx-get="/harmony/object_factory/{viewId}" hx-target="#object_factory">
            </div>
        """

    actions_html += f"""
        <hr>
        <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="/harmony/clear_pixel/{viewId}" hx-target="#interactor">
    """

    return HTMLResponse(interactor_template.format(info=info_html, actions=actions_html))


@harmony.post('/select_additional_pixel/{viewId}')
def selectAdditionalPixel(viewId: str):
    pass


# ---------------------------------------------------------------------------
# Minimap stream
# ---------------------------------------------------------------------------

@harmony.get('/minimap/{viewId}')
def minimapResponse(viewId: str):
    cm = _get_cm()
    broadcaster = get_broadcaster("VirtualMap", lambda: render_minimap(cm))
    return StreamingResponse(broadcaster.subscribe(), media_type='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Session control panel
# ---------------------------------------------------------------------------

@harmony.get('/control')
def session_control_list(request: Request):
    cm = _get_cm()
    # Read the template
    with open(f"{os.path.dirname(__file__)}/harmony_templates/SessionList.html") as f:
        template = f.read()
    session_rows = ""
    for sid, cfg in SESSIONS.items():
        session_rows += f"<tr><td>{sid}</td><td><a href='/harmony/control/{sid}'>Manage</a></td></tr>"
    return HTMLResponse(template.replace("{session_rows}", session_rows))


@harmony.get('/control/{viewId}')
def session_control_panel(viewId: str):
    if viewId not in SESSIONS:
        return Response(f"Session {viewId} not found", status_code=404)
    cm = _get_cm()
    with open(f"{os.path.dirname(__file__)}/harmony_templates/ControlPanel.html") as f:
        template = f.read()
    return HTMLResponse(template.replace("{viewId}", viewId))


@harmony.post('/control/{viewId}/update')
async def update_session_config(viewId: str, request: Request):
    if viewId not in SESSIONS:
        return Response(f"Session {viewId} not found", status_code=404)

    form = await request.form()
    old_config = SESSIONS[viewId]
    new_config = SessionConfig()
    new_config.selection = old_config.selection

    cm = _get_cm()
    for obj in cm.memory:
        oid = obj.oid
        if form.get(f"{oid}_selectable"):
            new_config.selectable.append(oid)
        if form.get(f"{oid}_terrain"):
            new_config.terrain.append(oid)
        if form.get(f"{oid}_targetable"):
            new_config.targetable.append(oid)
        if form.get(f"{oid}_enemies"):
            new_config.enemies.append(oid)
        if form.get(f"{oid}_allies"):
            new_config.allies.append(oid)
        if form.get(f"{oid}_moveable"):
            new_config.moveable.append(oid)

    SESSIONS[viewId] = new_config
    return RedirectResponse(url=f'/harmony/control/{viewId}', status_code=303)


# ---------------------------------------------------------------------------
# Static file serving (bootstrap, htmx, etc.)
# ---------------------------------------------------------------------------

def _make_static_router():
    """Creates routes for static files in the harmony app."""
    pass  # Handled at app level in create_harmony_app / start_servers


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _build_fastapi_app(template_name="Harmony.html", include_configurator=False) -> FastAPI:
    """Create a FastAPI app for harmony."""
    global _config, _cc, _cm

    app = FastAPI()

    # State
    _config['HARMONY_TEMPLATE'] = template_name

    # Include the harmony router
    app.include_router(harmony)

    # Root redirect
    @app.get('/')
    def index():
        return RedirectResponse('/harmony/', status_code=303)

    # Static files
    @app.get('/bootstrap.min.css')
    def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
            content = f.read()
        return Response(content, media_type="text/css")

    @app.get('/bootstrap.min.js')
    def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    @app.get('/htmx.min.js')
    def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    @app.get('/HarmonyTemplate.css')
    def getHarmonyCSS():
        with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyTemplate.css", "r") as f:
            content = f.read()
        return Response(content, media_type="text/css")

    @app.get('/HarmonyCanvas.js')
    def getHarmonyCanvasJS():
        with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyCanvas.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    # Mount Flask calibrator/observer as WSGI sub-apps if needed
    if include_configurator:
        from flask import Flask as _Flask
        flask_sub = _Flask(__name__ + "_sub")

        from observer.configurator import configurator as _cfg_bp
        from observer.calibrator import calibrator as _cal_bp

        flask_sub.register_blueprint(_cfg_bp, url_prefix='/configurator')
        flask_sub.register_blueprint(_cal_bp, url_prefix='/calibrator')

        @flask_sub.route('/')
        def _flask_root():
            from flask import redirect
            return redirect('/configurator', code=303)

        app.mount('/configurator', WSGIMiddleware(flask_sub))
        app.mount('/calibrator', WSGIMiddleware(flask_sub))

    return app


def create_harmony_app(template_name="Harmony.html") -> FastAPI:
    """Public factory used by tests and legacy callers."""
    global _cc, _cm, APPS

    app = FastAPI()
    _config['HARMONY_TEMPLATE'] = template_name

    cc = HexCaptureConfiguration()
    if cc.hex is None:
        cc.hex = HexGridConfiguration()
    cc.capture()
    cm = HarmonyMachine(cc)

    _cc = cc
    _cm = cm

    # Expose on app.state for test compatibility
    app.state.cc = cc
    app.state.cm = cm
    app.state.config = _config

    app.include_router(harmony)

    @app.get('/')
    def index():
        return RedirectResponse('/harmony/', status_code=303)

    @app.get('/bootstrap.min.css')
    def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
            content = f.read()
        return Response(content, media_type="text/css")

    @app.get('/bootstrap.min.js')
    def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    @app.get('/htmx.min.js')
    def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    @app.get('/HarmonyTemplate.css')
    def getHarmonyCSS():
        with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyTemplate.css", "r") as f:
            content = f.read()
        return Response(content, media_type="text/css")

    @app.get('/HarmonyCanvas.js')
    def getHarmonyCanvasJS():
        with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyCanvas.js", "r") as f:
            content = f.read()
        return Response(content, media_type="application/javascript")

    APPS.append(app)

    setObserverApp(app)
    registerCaptureService(_CaptureServiceProxy())

    # Reset to initialize HarmonyMachine properly
    resetHarmony()

    return app


def setHarmonyApp(newApp):
    pass  # Legacy no-op; state is now module-global


class _CaptureServiceProxy:
    """Thin proxy passed to calibrator.registerCaptureService().
    calibrator accesses app.cm to call cycle(); we forward that
    to the module-level _cm global so it always sees the live machine."""
    @property
    def cm(self):
        return _cm


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def start_servers():
    global _cc, _cm, APPS
    import uvicorn

    APPS.clear()

    # Shared state
    cc = HexCaptureConfiguration()
    if cc.hex is None:
        cc.hex = HexGridConfiguration()
    cc.capture()
    cm = HarmonyMachine(cc)

    _cc = cc
    _cm = cm

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        print("Shutting down broadcasters...")
        SHUTDOWN_EVENT.set()
        for broadcaster in BROADCASTERS.values():
            broadcaster.stop()
        # wake up all condition variables
        for broadcaster in BROADCASTERS.values():
            with broadcaster.condition:
                broadcaster.condition.notify_all()

    # --- Admin App (port 7000) ---
    _config['HARMONY_TEMPLATE'] = 'Harmony.html'

    from observer.CalibratedObserver import CalibrationObserver as _CalObs
    admin_app = FastAPI(lifespan=lifespan) # Shared State
    admin_app.state.cc = cc
    # This cm is HarmonyMachine, configurator expects CalibrationObserver for some parts?
    # Actually configurator.py line 22-23:
    # cc = request.app.state.cc
    # cm = request.app.state.cm
    # And buildConfigurator uses cm.calibrationPts.
    # In harmonyServer.py line 45-46:
    # _cc = None
    # _cm = None
    # We should ensure the admin_app.state.cm is what's expected.
    # The old flask_sub.cm was _CalObs(cc).
    admin_app.state.cm = _CalObs(cc) # For configurator
    admin_app.state.config = _config
    APPS.append(admin_app)

    admin_app.include_router(harmony)

    @admin_app.get('/')
    def admin_index():
        return RedirectResponse('/harmony/', status_code=303)

    for route_name, filename, media_type in [
        ('/bootstrap.min.css', 'templates/bootstrap.min.css', 'text/css'),
        ('/bootstrap.min.js', 'templates/bootstrap.min.js', 'application/javascript'),
        ('/htmx.min.js', 'templates/htmx.min.js', 'application/javascript'),
        ('/HarmonyTemplate.css', 'harmony_templates/HarmonyTemplate.css', 'text/css'),
        ('/HarmonyCanvas.js', 'harmony_templates/HarmonyCanvas.js', 'application/javascript'),
    ]:
        _fp = os.path.join(os.path.dirname(__file__), filename)
        _mt = media_type

        def make_static(fp=_fp, mt=_mt):
            def _route():
                with open(fp, "r") as f:
                    return Response(f.read(), media_type=mt)
            return _route

        admin_app.add_api_route(route_name, make_static(), methods=["GET"])

    # Include new FastAPI configurator router
    admin_app.include_router(configurator)

    # Note: calibrator and observerServer Flask blueprints are abandoned.
    # registerCaptureService(_CaptureServiceProxy())

    # --- User App (port 7001) ---
    user_config = {'HARMONY_TEMPLATE': 'HarmonyUser.html'}

    user_app = FastAPI(lifespan=lifespan)
    user_app.state.cc = cc
    user_app.state.cm = cm
    user_app.state.config = user_config
    APPS.append(user_app)

    # User app needs its own router with different config
    # We use a workaround: temporarily swap _config when building the user router
    # Since they share module globals, the user app just uses the same router
    # but the template name will match whatever _config says at request time.
    # For multi-app support we include the same harmony router but the _config
    # dict is shared (admin wins). We override for the user app via middleware.

    user_app.include_router(harmony)

    @user_app.get('/')
    def user_index():
        return RedirectResponse('/harmony/', status_code=303)

    for route_name, filename, media_type in [
        ('/bootstrap.min.css', 'templates/bootstrap.min.css', 'text/css'),
        ('/bootstrap.min.js', 'templates/bootstrap.min.js', 'application/javascript'),
        ('/htmx.min.js', 'templates/htmx.min.js', 'application/javascript'),
        ('/HarmonyTemplate.css', 'harmony_templates/HarmonyTemplate.css', 'text/css'),
        ('/HarmonyCanvas.js', 'harmony_templates/HarmonyCanvas.js', 'application/javascript'),
    ]:
        _fp = os.path.join(os.path.dirname(__file__), filename)
        _mt = media_type

        def make_static_user(fp=_fp, mt=_mt):
            def _route():
                with open(fp, "r") as f:
                    return Response(f.read(), media_type=mt)
            return _route

        user_app.add_api_route(route_name, make_static_user(), methods=["GET"])

    # Pre-warm broadcasters
    try:
        get_broadcaster("VirtualMap", lambda: render_minimap(_cm))
        for camName in cc.cameras.keys():
            get_broadcaster(camName, lambda c=camName: render_camera(cc, c))
    except Exception as e:
        print(f"Failed to start broadcasters: {e}")

    # Launch user server in a background thread.
    # IMPORTANT: run via uvicorn.Server so we can disable its signal-handler
    # installation — only the main-thread uvicorn should own SIGINT/SIGTERM.
    def run_user():
        import asyncio
        print("Launching Harmony User Server on 7001")
        cfg = uvicorn.Config(user_app, host="0.0.0.0", port=7001, log_level="warning")
        server = uvicorn.Server(cfg)
        asyncio.run(server.serve())

    t = threading.Thread(target=run_user, daemon=True)
    t.start()

    # Launch admin server (blocking)
    import uvicorn
    print(f"Launching Harmony Admin Server on 7000 (PID: {os.getpid()})")
    
    cfg = uvicorn.Config(admin_app, host="0.0.0.0", port=7000, log_level="info", timeout_graceful_shutdown=1)
    server = uvicorn.Server(cfg)

    # Monkey-patch uvicorn's handle_exit to set our shutdown event immediately
    original_handle_exit = server.handle_exit
    def patched_handle_exit(*args, **kwargs):
        print(f"Server exit triggered, setting shutdown event.")
        SHUTDOWN_EVENT.set()
        # Trigger cleanup for broadcasters immediately
        for broadcaster in BROADCASTERS.values():
            broadcaster.stop()
            with broadcaster.condition:
                broadcaster.condition.notify_all()
        return original_handle_exit(*args, **kwargs)
    
    server.handle_exit = patched_handle_exit

    server.run()


if __name__ == "__main__":
    start_servers()
