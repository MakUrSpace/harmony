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
from observer.observerServer import observer, registerCaptureService, setObserverApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT


import sys
import os

oldPath = os.getcwd()
try:
    observerDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, observerDirectory)
    from ipynb.fs.full.HarmonyMachine import HarmonyMachine, INCHES_TO_MM
    from ipynb.fs.full.Observer import hStackImages, clipImage, Camera
finally:
    os.chdir(oldPath)

# --- Monkey Patch for Dynamic Grid (Fixes Clipping) ---
# HexCaptureConfiguration is already imported from observer above
# from observer.dynamic_grid import draw_dynamic_grid_overlay, patched_cameraGriddle # Removed due to build issue
import math

def draw_dynamic_grid_overlay(self, cam):
    """
    Draws the hex grid dynamically based on the camera's FOV in real space.
    Prevents clipping issues caused by fixed-size minimaps.
    Monkey-patched into HexCaptureConfiguration.
    """
    if self.rsc is None:
         print(f"DynamicGrid: RSC is None for {cam}")
         return None

    # Get camera frame size
    if cam not in self.cameras:
        print(f"DynamicGrid: Camera {cam} not found")
        return None

        
    height, width = self.cameras[cam].mostRecentFrame.shape[:2]
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get converter for this camera
    if str(cam) not in self.rsc.converters:
        return None
        
    converter = self.rsc.converters[str(cam)][0]
    
    # Define 4 corners of the camera frame
    corners_cam = [(0, 0), (width, 0), (width, height), (0, height)]
    
    # Project to Real Space
    corners_real = []
    for p in corners_cam:
        try:
            rp = converter.convertCameraToRealSpace(p)
            corners_real.append(rp)
        except Exception:
            continue
            
    if not corners_real:
        return None

    # Convert to Axial to find range
    qs = []
    rs = []
    for cr in corners_real:
        q, r = self.pixel_to_axial(cr[0], cr[1])
        qs.append(q)
        rs.append(r)
        
    min_q, max_q = min(qs), max(qs)
    min_r, max_r = min(rs), max(rs)
    
    # Add padding to ensure smooth edges
    pad = 2
    grid_color = (255, 255, 255)
    
    # Iterate Q, R within bounds
    poly_count = 0
    for q in range(min_q - pad, max_q + pad + 1):
        for r in range(min_r - pad, max_r + pad + 1):
            try:
                # Use class method to get camera polygon for this hex
                # cam_hex_at_axial(self, cam, q, r)
                # This method maps real space hex (q,r) back to camera space using convertRealToCameraSpace
                poly_cam = self.cam_hex_at_axial(cam, q, r)
                
                # Draw
                cv2.polylines(overlay, [poly_cam], True, grid_color, 3, cv2.LINE_AA)
                poly_count += 1
            except Exception:
                continue
                
    if poly_count == 0:
        print(f"DynamicGrid: No polygons drawn for {cam}")
    else:
        print(f"DynamicGrid: Drawn {poly_count} polygons for {cam}")

    return overlay

def patched_cameraGriddle(self, cam, objectsAndColors=[]):
    """
    Monkey-patched replacement for cameraGriddle.
    Uses dynamic grid generation when no objects are present.
    """
    height, width = self.cameras[cam].mostRecentFrame.shape[:2]

    if self.rsc is None:
         return np.zeros((height, width, 3), dtype="uint8")

    # If no objects, use the dynamic grid for better quality/coverage
    if not objectsAndColors:
        # Call the injected method
        if hasattr(self, 'draw_dynamic_grid_overlay'):
             dynamic_grid = self.draw_dynamic_grid_overlay(cam)
        else:
             # Fallback if somehow not injected properly
             dynamic_grid = draw_dynamic_grid_overlay(self, cam)
             
        if dynamic_grid is not None:
            return dynamic_grid
        # Fallback to legacy if dynamic failed
    
    # Legacy/Object path (clipped to 1200x1200mm usually)
    # This path is still used if objects need to be drawn
    try:
        M = self.rsc.converters[cam][0].M
        Minv = np.linalg.inv(M)

        warped = cv2.warpPerspective(
            self.buildMiniMap(objectsAndColors=objectsAndColors),
            Minv,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped[:height, :width]
    except Exception as e:
        print(f"Error in legacy cameraGriddle: {e}")
        return np.zeros((height, width, 3), dtype="uint8")

def patched_objectToHull(self, obj):
    """
    Monkey-patched version of objectToHull to ensure consistent coordinate transformation
    between Real Space (object positions) and Map Space (grid visualization).
    """
    all_points_real = []
    
    try:
        # Iterate over camera views for this object
        for cam_name, change in obj.changeSet.items():
            if self.rsc is None:
                continue
                
            # Skip cameras not in RSC
            if cam_name not in self.rsc.converters:
                continue
                
            converter = self.rsc.converters[cam_name][0]
            
            for contour in change.changeContours:
                # contour is (N, 1, 2) or (N, 2)
                for pt in contour:
                    # Flatten info [x, y]
                    if len(pt.shape) > 1:
                        x, y = pt[0]
                    else:
                        x, y = pt
                        
                    # Convert Camera -> Real Space
                    try:
                        real_pt = converter.convertCameraToRealSpace((float(x), float(y)))
                        all_points_real.append(real_pt)
                    except Exception:
                        continue
                    
        if not all_points_real:
            return np.array([], dtype=np.int32)
            
        # Transform Real Space -> Map Pixel Space
        pts_real = np.array(all_points_real, dtype=np.float32)
        
        if hasattr(self, 'apply_affine_pts'):
            pts_map = self.apply_affine_pts(pts_real)
        else:
            print("Warning: apply_affine_pts missing in patched_objectToHull")
            pts_map = pts_real 
            
        # Compute Convex Hull of mapped points
        pts_map_i = np.round(pts_map).astype(np.int32)
        hull = cv2.convexHull(pts_map_i)
        
        return hull
        
    except Exception as e:
        print(f"Error in patched_objectToHull: {e}")
        return np.array([], dtype=np.int32)

def patched_buildMiniMap(self, objectsAndColors=[], hex_cfg=None):
    """
    Monkey-patched replacement for buildMiniMap.
    Uses dynamic canvas size from HexGridConfiguration instead of hardcoded 1200.
    """
    width = 1600
    height = 1600
    if hasattr(self, 'hex') and self.hex:
        width = self.hex.width
        height = self.hex.height

    # Transform and Draw RealSpaceContours (Active Zones)
    # Ensure they align with the affine-transformed grid
    # FIX: We now fetch Active Zones directly from self.cc.cameras to avoid warpPerspective clipping
    shift_x = 0
    shift_y = 0
    
    # Initialize with default or current dimensions to ensure 'image' exists
    image = self.draw_hex_grid_overlay(
        np.zeros([height, width, 3], dtype="uint8"), 0, 0, width, height
    )
    
    try:
        true_real_contours = []
        cameras = None
        
        # Determine where cameras are stored
        if hasattr(self, 'cameras') and self.cameras:
            cameras = self.cameras
        elif hasattr(self, 'cc') and hasattr(self.cc, 'cameras'):
            cameras = self.cc.cameras
            
        if cameras:
            for cam_name, cam in cameras.items():
                # Correct Property is 'activeZone' not 'activeZonePolygon'
                # Check both for compatibility/safety
                az = getattr(cam, 'activeZone', None)
                if az is None:
                    az = getattr(cam, 'activeZonePolygon', None)
                
                if az is not None and len(az) > 0:
                    # These are in Camera Coordinates. Transform to Real Space.
                    if hasattr(self, 'rsc') and cam_name in self.rsc.converters:
                        converter = self.rsc.converters[cam_name][0]
                        
                        # Convert polygon points
                        real_pts = []
                        for pt in az:
                            # Robustly handle (N, 2) and (N, 1, 2) shapes
                            x, y = 0, 0
                            try:
                                is_nested = False
                                if hasattr(pt, 'shape'):
                                     if len(pt.shape) > 1:
                                         is_nested = True
                                elif hasattr(pt, '__len__') and len(pt) == 1 and hasattr(pt[0], '__len__'):
                                     is_nested = True
                                
                                if is_nested:
                                    x = float(pt[0][0])
                                    y = float(pt[0][1])
                                else:
                                    x = float(pt[0])
                                    y = float(pt[1])
                            except Exception as e:
                                print(f"MiniMap: Error processing point {pt}: {e}")
                                continue

                            r_pt = converter.convertCameraToRealSpace((x, y))
                            real_pts.append(r_pt)
                        
                        true_real_contours.append(np.array(real_pts, dtype=np.float32))

        # Calculate Global Bounding Box from these TRUE contours
        if true_real_contours:
            all_pts = np.vstack(true_real_contours)
            min_x = np.min(all_pts[:, 0])
            max_x = np.max(all_pts[:, 0])
            min_y = np.min(all_pts[:, 1])
            max_y = np.max(all_pts[:, 1])
            
            # Recalculate Shift & Dimensions based on TRUE bounds
            shift_x = 0
            shift_y = 0
            if min_x < 0: shift_x = -min_x
            if min_y < 0: shift_y = -min_y
            
            req_w = int(max_x - min_x) if min_x < 0 else int(max_x)
            req_h = int(max_y - min_y) if min_y < 0 else int(max_y)

            # Expand canvas if needed
            width = max(width, req_w)
            height = max(height, req_h)
            
            # Update hex config (Critical for grid alignment)
            if hasattr(self, 'hex') and self.hex:
                self.hex.width = width
                self.hex.height = height
                self.hex.offset_xy = (shift_x, shift_y)

            # Re-initialize image with new dimensions if they changed
            image = self.draw_hex_grid_overlay(
                np.zeros([height, width, 3], dtype="uint8"), 0, 0, width, height
            )

            # Draw the Unclipped Contours
            transformed_contours = []
            for pts_real in true_real_contours:
                if hasattr(self, 'apply_affine_pts'):
                    pts_map = self.apply_affine_pts(pts_real)
                else:
                    pts_map = pts_real
                    print("ERROR: apply_affine_pts not found!")

                # Apply the shift
                pts_map[:, 0] += shift_x
                pts_map[:, 1] += shift_y
                    
                cnt_map = np.round(pts_map).astype(np.int32).reshape(-1, 1, 2)
                transformed_contours.append(cnt_map)
            
            cv2.drawContours(image, transformed_contours, -1, (125, 125, 125), 6)

            # Draw Unified Boundary
            if len(transformed_contours) > 0:
                try:
                    # Create a mask for the union of all active zones
                    union_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # Fill all contours on the mask
                    cv2.drawContours(union_mask, transformed_contours, -1, 255, -1)
                    
                    # Find the outer contour of the union
                    # OpenCV 4.x returns (contours, hierarchy)
                    # OpenCV 3.x returns (image, contours, hierarchy)
                    cnts_res = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(cnts_res) == 2:
                        union_cnts = cnts_res[0]
                    else:
                        union_cnts = cnts_res[1]
                    
                    if len(union_cnts) > 0:
                        # Draw the unified boundary (White, Thicker)
                        cv2.drawContours(image, union_cnts, -1, (255, 255, 255), 4)

                except Exception as e:
                    print(f"Error drawing unified boundary: {e}")
            
    except Exception as e:
        print(f"Error drawing true active zones: {e}")

    drawnObjs = []
    print(f"MiniMap: Rendering {len(objectsAndColors)} objects")
    
    for objAndColor in objectsAndColors[::-1]:
        obj = objAndColor.object
        color = objAndColor.color
        if obj in drawnObjs:
            continue
        drawnObjs.append(obj)
        
        try:
            hull = self.objectToHull(obj)
            if hull is not None and len(hull) > 0:
                # Hull is already in map coordinates (from patched_objectToHull)
                # We need to apply the shift to it as well!
                # patched_objectToHull uses apply_affine_pts but doesn't know about our local shift_x/y
                
                # Shift hull points
                hull[:, 0, 0] += int(shift_x)
                hull[:, 0, 1] += int(shift_y)
                
                image = cv2.drawContours(image, [hull], -1, color, -1)
            else:
                print(f"MiniMap: Zero-area/Empty hull for object {obj}")
        except Exception as e:
            print(f"MiniMap: Error drawing object hull: {e}")

    return image

HexCaptureConfiguration.draw_dynamic_grid_overlay = draw_dynamic_grid_overlay
HexCaptureConfiguration.cameraGriddle = patched_cameraGriddle
HexCaptureConfiguration.buildMiniMap = patched_buildMiniMap
HexCaptureConfiguration.objectToHull = patched_objectToHull
print("Monkey Patched HexCaptureConfiguration methods for dynamic grid/minimap generation")
# ------------------------------------------------------


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')


perspective_res = (1920, 1080)
virtual_map_res = (1200, 1200)


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


@harmony.route('/harmony_console', methods=['GET'])
def getConsoleImage():
    return Response(stream_with_context(renderConsole()), mimetype='multipart/x-mixed-replace; boundary=frame')



# ----------------------------------------------------------------------------------
# Frame Broadcasting System
# ----------------------------------------------------------------------------------

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
                print(f"Broadcaster started for {self.key}")

    def stop(self):
        with self.lock:
            self.running = False

    def _run(self):
        while self.running:
            start_time = time.time()
            try:
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
        # Send last frame immediately if available (Zero TTFF)
        with self.lock:
            self.clients += 1
            if self.last_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpg\r\n\r\n' + self.last_frame + b'\r\n')
        
        try:
            while True:
                with self.condition:
                    self.condition.wait()
                    if not self.running:
                        break
                    frame = self.last_frame
                
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')
        finally:
            with self.lock:
                self.clients -= 1


def render_minimap(cm):
    try:
        if not cm:
            print("render_minimap: cm is None")
            return None
            
        camImage = cm.buildMiniMap()
        
        x, y, w, h = cm.cc.realSpaceBoundingBox()
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
            camImage = camImage[crop_y:end_y, crop_x:end_x]
        
        camImage = cv2.resize(camImage, virtual_map_res, interpolation=cv2.INTER_AREA)
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
                 # Check if grid has content
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


@harmony.route('/camWithChanges/<camName>/<viewId>')
def cameraViewWithChangesResponse(camName, viewId):
    if camName == "VirtualMap":
        # Pass current_app.cm to lambda to creation, but cached one uses its own cm
        broadcaster = get_broadcaster("VirtualMap", lambda: render_minimap(current_app.cm))
    else:
        broadcaster = get_broadcaster(camName, lambda: render_camera(current_app.cc, camName))
        
    return Response(stream_with_context(broadcaster.subscribe()), mimetype='multipart/x-mixed-replace; boundary=frame')


def safe_point(pt):
    p = pt.tolist()
    # Handle contour format [[x,y]] vs point format [x,y]
    if len(p) > 0 and isinstance(p[0], list):
        return (p[0][0], p[0][1])
    return (p[0], p[1])

def get_conversion_params(cam_name):
    # Returns (scale_x, scale_y, offset_x, offset_y)
    try:
        am_virtual_map = (cam_name == "VirtualMap")
        
        if am_virtual_map:
            try:
                if hasattr(current_app.cm.cc, 'realSpaceBoundingBox'):
                    bx, by, bw, bh = current_app.cm.cc.realSpaceBoundingBox()
                    
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
                    if hasattr(current_app.cm.cc, 'hex') and current_app.cm.cc.hex:
                        w_canvas = current_app.cm.cc.hex.width
                        h_canvas = current_app.cm.cc.hex.height
                        
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
            
            # Fallback
            return 1.0, 1.0, 0, 0

        cam = current_app.cc.cameras.get(cam_name)
        if cam is None:
            return 1.0, 1.0, 0, 0
            
        x, y, w, h = cam.activeZoneBoundingBox
        if w == 0 or h == 0:
            return 1.0, 1.0, 0, 0
            
        # Target resolution is 1920x1080 for physical cameras
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
    # Legacy wrapper if needed, but we should use get_conversion_params
    sx, sy, _, _ = get_conversion_params(cam_name)
    return sx, sy

def scale_point(pt, scale):
    return (pt[0] * scale[0], pt[1] * scale[1])

def axial_to_ui_object(q, r):
    # For VirtualMap, we need to handle its offset.
    # The buildMiniMap function (in Observer.ipynb) uses self.cc.realSpaceBoundingBox to crop.
    # realSpaceBoundingBox is [min_x, min_y, max_x, max_y] (or w, h?)
    # Let's check Observer.ipynb if we can... we can't easily see it right now.
    # But usually bounding box is x, y, w, h or min/max.
    # Assuming realSpaceBoundingBox is available on cc.
    
    raw_vm_points = current_app.cm.cc.hex_at_axial(q, r)
    
    # Get scale and offset from conversion params (now handles VirtualMap correctly)
    scale_x, scale_y, off_x_base, off_y_base = get_conversion_params("VirtualMap") 
    
    vm_points = []
    for raw_pt in raw_vm_points:
        pt = safe_point(raw_pt)
        # Apply transformation: (pt - off) * scale
        # get_conversion_params returns offset/scale such that:
        # ui_x = (x - off_x) * scale_x
        
        scaled_x = (pt[0] - off_x_base) * scale_x
        scaled_y = (pt[1] - off_y_base) * scale_y
        vm_points.append((scaled_x, scaled_y))

    return {
        "VirtualMap": vm_points,
        **{camName: [
            scale_point_new(safe_point(pt), get_conversion_params(camName))
            for pt in current_app.cm.cc.cam_hex_at_axial(camName, q, r)] for camName in current_app.cc.cameras.keys()}
    }


@harmony.route('/canvas_data/<viewId>')
def getCanvasData(viewId):
    objects = {}
    for obj in current_app.cm.memory:
        vm_pts = []
        # Prepare VirtualMap points

        
        # Use centralized conversion logic
        vm_params = get_conversion_params("VirtualMap")
        
        # FIX: Use objectToHull to get the full shape instead of single hex center
        # This matches what is drawn on the server side map
        hull = current_app.cm.cc.objectToHull(obj)
        if hull is not None and len(hull) > 0:
             # hull is (N, 1, 2) or (N, 2)
             # We need to flatten it to list of points
             raw_vm_pts = hull.reshape(-1, 2)
             vm_pts = [scale_point_new((float(pt[0]), float(pt[1])), vm_params) for pt in raw_vm_pts]
        else:
             vm_pts = []

        objects[obj.oid] = {
            "VirtualMap": vm_pts,
            **{
                camName: [scale_point_new((pt[0], pt[1]), get_conversion_params(camName)) for pt in obj.changeSet[camName].changePoints]
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
            camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_LINEAR)
            ret, camImage = cv2.imencode('.jpg', camImage, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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
    "terrain": (30, 105, 210),    # RGB(210, 105, 30) -> BGR
    "selectable": (180, 105, 255) # RGB(255, 105, 180) -> BGR
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
        # Virtual Map Input Logic
        # UI coordinates are offset by min_x, min_y.
        # We need to map UI (0,0) back to Real (min_x, min_y).
        # And apply reverse scaling.
        real_x = x
        real_y = y
        if current_app.cm and hasattr(current_app.cm.cc, 'realSpaceBoundingBox'):
            scale_x, scale_y, min_x, min_y = get_conversion_params("VirtualMap")
            
            # UI = (Raw - Offset) * Scale
            # (Raw - Offset) = UI / Scale
            # Raw = (UI / Scale) + Offset
            if scale_x > 0 and scale_y > 0:
                real_x = (x / scale_x) + min_x
                real_y = (y / scale_y) + min_y
            
        axial_coord = current_app.cm.cc.pixel_to_axial(real_x, real_y)
    else:
        # Camera Input Logic
        # Client sends coordinates in 1920x1080 of the Active Zone.
        # We need to map back to full camera frame coordinates.
        scale_x, scale_y, offset_x, offset_y = get_conversion_params(cam)
        
        # UI = (Raw - Offset) * Scale
        # Raw - Offset = UI / Scale
        # Raw = (UI / Scale) + Offset
        
        raw_x = (x / scale_x) + offset_x
        raw_y = (y / scale_y) + offset_y
        
        axial_coord = current_app.cm.cc.camCoordToAxial(cam, (raw_x, raw_y))
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
        return ""

    # Helper to resolve type from SESSIONS config
    def get_object_type(oid, viewId):
        session = SESSIONS.get(viewId)
        if not session: return "Unknown"
        
        # Priority Order
        if oid in session.moveable: return "Moveable"
        if oid in session.allies: return "Ally"
        if oid in session.enemies: return "Enemy"
        if oid in session.targetable: return "Targetable"
        if oid in session.terrain: return "Terrain"
        
        # Default fallback
        return "Selectable"

    # Helper to find object at cell
    def find_object_at(cell):
        for mem in current_app.cm.memory:
            if current_app.cm.cc.changeSetToAxialCoord(mem) == cell:
                return mem
        return None

    # Build info string
    info_html = f"<h2>Selected First Cell: {first}</h2>"

    first_obj = find_object_at(first)
    if first_obj:
        o_type = get_object_type(first_obj.oid, viewId)
        info_html += f"<h3>Object: {first_obj.oid} <br><small>Type: {o_type}</small></h3>"
        
    # Additional cells
    if selected.additionalCells:
        info_html += "<hr><h3>Additional Selections:</h3>"
        for i, cell in enumerate(selected.additionalCells):
            if i > 0:
                 info_html += "<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>"

            dist = current_app.cm.cc.axial_distance(first, cell)
            
            cell_obj = find_object_at(cell)
            obj_str = ""
            if cell_obj:
                o_type = get_object_type(cell_obj.oid, viewId)
                obj_str = f" <br>-> Object: {cell_obj.oid} ({o_type})"
            
            # Highlight most recent (index 0)
            style = "border: 2px solid cyan; padding: 5px; margin: 2px;" if i == 0 else "padding: 5px; margin: 2px;"
            label = "Latest Selection" if i == 0 else f"Selection {i+1}"
            
            info_html += f"<div style='{style}'><b>{label}: {cell}</b><br>Dist to First: {dist} cells{obj_str}</div>"

    # Actions
    actions_html = ""
    
    if first_obj:
        if selected.additionalCells:
            # Move to latest selection
            target = selected.additionalCells[0] 
            
            # Check permissions: Admin can always move, User only if Moveable
            is_admin = current_app.config.get('HARMONY_TEMPLATE') == "Harmony.html"
            session = SESSIONS.get(viewId)
            is_moveable = session and first_obj.oid in session.moveable
            
            if is_admin or is_moveable:
                actions_html += f"""
                    <input type="button" class="btn btn-info" value="Move {first_obj.oid} Here" hx-get="{url_for(".buildHarmony")}request_move/{first_obj.oid}/{viewId}" hx-target="#interactor">
                """
        else:
             # Only allow Admin to delete (Harmony.html)
             if current_app.config.get('HARMONY_TEMPLATE') == "Harmony.html":
                 actions_html += f"""
                    <input type="button" class="btn btn-danger" value="Delete Object" hx-delete="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#interactor">
                 """

    else:
        # If no object at first cell, allow defining one
        actions_html += f"""
            <div id="object_factory">
                <input type="button" class="btn btn-success" value="Define Object" hx-get="{url_for(".buildHarmony")}object_factory/{viewId}" hx-target="#object_factory">
            </div>
        """
        
    actions_html += f"""
        <hr>
        <input type="button" class="btn btn-danger" value="Clear Selection" hx-get="{url_for(".buildHarmony")}clear_pixel/{viewId}" hx-target="#interactor">
    """

    return interactor_template.format(info=info_html, actions=actions_html)


@harmony.route('/select_additional_pixel/<viewId>', methods=['POST'])
def selectAdditionalPixel(viewId):
    pass


@harmony.route('/minimap/<viewId>')
def minimapResponse(viewId):
    broadcaster = get_broadcaster("VirtualMap", lambda: render_minimap(current_app.cm))
    return Response(stream_with_context(broadcaster.subscribe()), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    app.cm = HarmonyMachine(app.cc)
    # app.register_blueprint(configurator, url_prefix='/configurator') # Separated
    app.register_blueprint(harmony, url_prefix='/harmony')
    with app.app_context():
        resetHarmony()
    # setConfiguratorApp(app) # Separated
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
    def make_app(template_name, register_capture=False):
        new_app = Flask(__name__)
        new_app.secret_key = str(uuid4())
        new_app.config['HARMONY_TEMPLATE'] = template_name
        
        if APPS:
            new_app.cm = APPS[0].cm
            new_app.cc = APPS[0].cc # Ensure cc is also shared if cm is
        else:
            # Use the shared state initialized in start_servers
            new_app.cc = cc
            new_app.cm = cm
            
        # Register blueprints
        # Configurator blueprint should only be registered for the admin app
        if register_capture: # Using register_capture as a proxy for admin app
            new_app.register_blueprint(configurator, url_prefix='/configurator')
        new_app.register_blueprint(harmony, url_prefix='/harmony')
        
        # Register other blueprints if needed, but observer/calibrator logic might need app instance
        # They are registered below in start_servers for the admin app mostly?
        # No, make_app seems to be general.
        
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

        @new_app.route('/HarmonyCanvas.js', methods=['GET'])
        def getHarmonyCanvasJS():
            with open(f"{os.path.dirname(__file__)}/harmony_templates/HarmonyCanvas.js", "r") as f:
                js = f.read()
            return Response(js, mimetype="application/javascript")
            
        if register_capture:
            registerCaptureService(new_app)
        APPS.append(new_app)
        return new_app

    # Create Apps
    admin_app = make_app("Harmony.html", register_capture=True)
    user_app = make_app("HarmonyUser.html", register_capture=False)
    
    app = admin_app # For legacy external usage if any
    
    # Monkey Patch for RTSPCamera to ensure threads are started
    try:
        from ipynb.fs.full.Observer import RTSPCamera
        
        original_collectImage = RTSPCamera.collectImage
        
        def patched_collectImage(self, timeout_s=5):
            # Ensure thread is started
            if self._cap_thread is None or not self._cap_thread.is_alive():
                print(f"Auto-starting capture thread for {self.camName}")
                self.start_capture()
                # Give it a moment to start
                time.sleep(0.1)
                
            return original_collectImage(self, timeout_s)
            
        RTSPCamera.collectImage = patched_collectImage
        print("Monkey Patched RTSPCamera.collectImage to auto-start threads")
    except ImportError:
        print("Could not import RTSPCamera for patching (might not be in use or import failed)")
        pass

    setConfiguratorApp(admin_app)
    setObserverApp(admin_app)
    
    # --- Monkey Patch for Calibrator App Sync ---
    # Separated to configuratorServer.py, likely not needed here anymore or logic needs change if they ran together.
    # But since they don't, we remove it.
    # --------------------------------------------
    
    # Launch User Server in Thread
    def run_user():
        print("Launching Harmony User Server on 7001")
        user_app.run(host="0.0.0.0", port=7001, use_reloader=False)

    t = threading.Thread(target=run_user)
    t.daemon = True
    t.start()
    
    # Launch Admin Server
    print("Launching Harmony Server on 7000")
    
    # Pre-warm Broadcasters
    # We do this after creating the app and getting CC/CM
    try:
        # VirtualMap
        get_broadcaster("VirtualMap", lambda: render_minimap(admin_app.cm))
        
        # Cameras
        for camName in admin_app.cc.cameras.keys():
            # Lambda capture issue: must bind camName
            # Also bind cc to local variable in loop or just use admin_app.cc
            get_broadcaster(camName, lambda c=camName: render_camera(admin_app.cc, c))
            
    except Exception as e:
        print(f"Failed to start broadcasters: {e}")

    admin_app.run(host="0.0.0.0", port=7000, use_reloader=False)

if __name__ == "__main__":
    start_servers()

