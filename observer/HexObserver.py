#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import cv2
from dataclasses import dataclass, asdict


# In[2]:


from observer.CalibratedObserver import CalibratedCaptureConfiguration, CalibratedObserver, MiniMapObject, TrackedObject, CameraChange


# In[3]:


@dataclass
class HexGridConfiguration:
    size: float = 38
    rotation_deg: float = 0
    offset_xy: tuple = (0, 0)
    anchor_xy: tuple = (-12, -19)
    width: int = 1600
    height: int = 1600

    def __post_init__(self):
        self.anchor_xy = (-int((self.size / 2.86) + 0.25), -int(self.size / 2))


class HexCaptureConfiguration(CalibratedCaptureConfiguration):
    def loadConfiguration(self, path="observerConfiguration.json"):
        super().loadConfiguration()
        try:
            config = self.readConfigFile()
        except Exception:
            config = {}
        self.hex = config.get("hex", None)
        if self.hex is not None:
            self.hex = HexGridConfiguration(**self.hex)
        self.grid_overlays = None

    def buildConfiguration(self):
        config = super().buildConfiguration()
        if self.hex is not None:
            config['hex'] = asdict(self.hex)
        return config

    def realSpaceBoundingBox(self):
        true_real_contours = []
        try:
            for cam_name, cam in self.cameras.items():
                az = getattr(cam, 'activeZone', None)
                if az is None:
                    az = getattr(cam, 'activeZonePolygon', None)
                if az is not None and len(az) > 0:
                    if hasattr(self, 'rsc') and self.rsc is not None and cam_name in self.rsc.converters:
                        real_pts = []
                        rsc = self.rsc
                        for pt in az:
                            x, y = 0, 0
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
                            converter = rsc.closestConverterToCamCoord(cam_name, (x, y))
                            r_pt = converter.convertCameraToRealSpace((x, y))
                            real_pts.append(r_pt)
                        true_real_contours.append(np.array(real_pts, dtype=np.float32))

            if true_real_contours:
                all_pts = np.vstack(true_real_contours)
                if hasattr(self, 'apply_affine_pts'):
                    all_pts = self.apply_affine_pts(all_pts)
                min_x = np.min(all_pts[:, 0])
                max_x = np.max(all_pts[:, 0])
                min_y = np.min(all_pts[:, 1])
                max_y = np.max(all_pts[:, 1])
                return min_x, min_y, max_x - min_x, max_y - min_y
        except Exception as e:
            print(f"Error in realSpaceBoundingBox: {e}")

        return 0, 0, 1600, 1600

    def axial_to_pixel(self, q: float, r: float) -> np.ndarray:
        """
        Axial (q, r) → grid-space pixel (x, y), pointy-top hexes
        """
        x = self.hex.size * math.sqrt(3) * (q + r / 2.0)
        y = self.hex.size * 1.5 * r
        center = np.array([[x, y]], dtype=np.float32)
        center_p = self.apply_affine_pts(center)[0]
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def axial_round(q: float, r: float) -> tuple[int, int]:
        """
        Round fractional axial coords to nearest hex
        """
        x = q
        z = r
        y = -x - z

        rx = round(x)
        ry = round(y)
        rz = round(z)

        dx = abs(rx - x)
        dy = abs(ry - y)
        dz = abs(rz - z)

        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dy > dz:
            ry = -rx - rz
        else:
            rz = -rx - ry

        return int(rx), int(rz)

    @staticmethod
    def pixel_to_axial_frac(x: float, y: float, size: float) -> tuple[float, float]:
        """
        Grid-space pixel → fractional axial (q, r)
        """
        r = (2.0 / 3.0) * (y / size)
        q = (x / (math.sqrt(3) * size)) - (r / 2.0)
        return q, r

    def pixel_to_axial(
        self,
        px: float, py: float
    ) -> tuple[int, int]:
        """
        Image pixel → axial hex (q, r)
        """
        # Invert affine (2x3 → 3x3)
        M = self.make_affine_2x3()
        M3 = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M3)

        pt = np.array([[px, py, 1.0]], dtype=np.float32).T
        gx, gy, _ = (M_inv @ pt).ravel()

        qf, rf = self.pixel_to_axial_frac(gx, gy, self.hex.size)
        return self.axial_round(qf, rf)

    def camCoordToAxial(self, cam, cam_coord: tuple[float]):
        real_coord = self.rsc.camCoordToRealSpace(cam, cam_coord)
        axial = self.pixel_to_axial(*real_coord)
        return axial

    def axialToCamCoord(self, cam, axial_coord: tuple[float]):
        real_coord = self.axial_to_pixel(*axial_coord)
        return self.rsc.realSpaceToCamCoord(real_coord, cam)

    def changeSetToAxialCoord(self, changeSet):
        real_coord = self.rsc.changeSetToRealCenter(changeSet)
        return self.pixel_to_axial(*real_coord)

    @staticmethod
    def axial_distance(a: tuple[int, int], b: tuple[int, int] = (0, 0)) -> int:
        q1, r1 = a
        q2, r2 = b
        return (abs(q1 - q2)
              + abs(q1 + r1 - q2 - r2)
              + abs(r1 - r2)) // 2

    def trackedObjectLastDistance(self, trackedObject):
        if trackedObject.isNewObject:
            return 0
        previousChangeSet = trackedObject.previousVersion()
        current_axial = self.changeSetToAxialCoord(trackedObject)
        previous_axial = self.changeSetToAxialCoord(previousChangeSet)
        return self.axial_distance(current_axial, previous_axial)

    def make_affine_2x3(self) -> np.ndarray:
        ax, ay = self.hex.anchor_xy
        tx, ty = self.hex.offset_xy
        theta = math.radians(self.hex.rotation_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        a11 = c
        a12 = -s
        a21 = s
        a22 = c

        # p' = A(p - anchor) + anchor + offset = A p + (anchor + offset - A anchor)
        b1 = ax + tx - (a11 * ax + a12 * ay)
        b2 = ay + ty - (a21 * ax + a22 * ay)

        return np.array([[a11, a12, b1],
                         [a21, a22, b2]], dtype=np.float32)

    def apply_affine_pts(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        pts_xy: (N,2) float array
        returns: (N,2) float array
        """
        M = self.make_affine_2x3()
        x = pts_xy[:, 0]
        y = pts_xy[:, 1]
        xp = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        yp = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return np.stack([xp, yp], axis=1)

    def hex_offsets_pointy(self) -> np.ndarray:
        """
        Corner offsets for a pointy-top hex centered at origin in *grid space*.
        Returns (6,2) float offsets.
        """
        offs = []
        for k in range(6):
            ang = math.radians(60 * k - 30)  # pointy-top
            offs.append([self.hex.size * math.cos(ang), self.hex.size * math.sin(ang)])
        return np.array(offs, dtype=np.float32),

    def draw_hex_grid_overlay(
        self,
        base_img: np.ndarray,
        x0: int, y0: int, w: int, h: int,
        *,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Draw a pointy-top hex grid over bbox [x0:x0+w, y0:y0+h] with
        scale/rotation/offset applied consistently to BOTH centers and corners.
        """
        overlay = self.draw_grid(color=color, thickness=thickness)

        # Blend only inside bbox
        out = base_img.copy()
        roi = out[y0:y0+h, x0:x0+w]
        roi_overlay = overlay[y0:y0+h, x0:x0+w]
        cv2.addWeighted(roi_overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
        out[y0:y0+h, x0:x0+w] = roi
        return out

    def draw_grid(
        self,
        color: tuple[int, int, int] = (80, 80, 80),
        thickness: int = 3
    ):
        params = (self.hex.width, self.hex.height, self.hex.size, self.hex.offset_xy, self.hex.anchor_xy, self.hex.rotation_deg, color, thickness)
        if hasattr(self, '_cached_grid_params') and self._cached_grid_params == params:
            return self._cached_grid_overlay.copy()

        self._cached_grid_params = params

        # FIX: OpenCV expects (height, width, 3)
        overlay = np.zeros((self.hex.height, self.hex.width, 3), dtype=np.uint8)

        # If you need it computed for internal state, keep it; otherwise can remove.
        M = self.make_affine_2x3()

        dx = math.sqrt(3) * self.hex.size
        dy = 1.5 * self.hex.size

        corner_offs = self.hex_offsets_pointy()[0]  # (6,2) float offsets

        pad = 4.0 * self.hex.size
        xmin_g = self.hex.anchor_xy[0] - pad
        xmax_g = self.hex.anchor_xy[0] + self.hex.width + pad
        ymin_g = self.hex.anchor_xy[1] - pad
        ymax_g = self.hex.anchor_xy[1] + self.hex.height + pad

        row = 0
        cy_g = ymin_g
        while cy_g <= ymax_g:
            x_off = 0.0 if (row % 2 == 0) else (dx / 2.0)
            cx_g = xmin_g + x_off
            while cx_g <= xmax_g:
                center_g = np.array([[cx_g, cy_g]], dtype=np.float32)

                # Transform center for quick reject
                center_p = self.apply_affine_pts(center_g)[0]

                if (-200 <= center_p[0] <= self.hex.width + 200) and (-200 <= center_p[1] <= self.hex.height + 200):
                    # Build hex in grid space then transform corners
                    poly_g = corner_offs + center_g          # (6,2)
                    poly_p = self.apply_affine_pts(poly_g)   # (6,2) float in pixel space

                    # Optional: reject if polygon bbox is fully off-screen
                    minx, miny = poly_p.min(axis=0)
                    maxx, maxy = poly_p.max(axis=0)
                    if maxx < -5 or maxy < -5 or minx > self.hex.width + 5 or miny > self.hex.height + 5:
                        cx_g += dx
                        continue

                    poly_i = np.round(poly_p).astype(np.int32).reshape((-1, 1, 2))

                    cv2.polylines(
                        overlay,
                        [poly_i],
                        isClosed=True,
                        color=color,
                        thickness=thickness,
                        lineType=cv2.LINE_8
                    )

                cx_g += dx
            cy_g += dy
            row += 1

        self._cached_grid_overlay = overlay.copy()
        return overlay

    def hex_at_axial(
        self,
        q: int,
        r: int
    ):
        """
        Fill a single pointy-top hex at axial coordinate (q, r) and return polygon points in grid coordinates.

        img               : OpenCV image to draw into (H, W, 3)
        q, r              : axial hex coordinates
        hex_size          : distance from hex center to corner (in grid space)
        color             : BGR color
        apply_affine_pts  : optional function (N,2)->(N,2) to map grid→pixel space
                             (use the same one as your grid renderer)
        """

        adjust = int(self.hex.size / 2)
        center = self.axial_to_pixel(q, r)
        cx, cy = center

        # --- hex corner offsets (pointy-top) ---
        corners = []
        for k in range(6):
            ang = math.radians(60 * k - 30)
            corners.append([
                cx + self.hex.size * math.cos(ang),
                cy + self.hex.size * math.sin(ang),
            ])

        poly = np.array(corners, dtype=np.float32)
        poly = self.apply_affine_pts(poly)

        poly_i = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        return poly_i

    def cam_hex_at_axial(self, cam, q: int, r: int):
        w, h = self.cameras[cam].mostRecentFrame.shape[:2]
        center_real = self.axial_to_pixel(q, r)
        cam_space_converter = self.rsc.closestConverterToRealCoord(cam, center_real)
        M = cam_space_converter.M
        Minv = np.linalg.inv(M)
        grid_poly = self.hex_at_axial(q, r)
        cam_poly = np.array([[int(d) for d in cam_space_converter.convertRealToCameraSpace(p[0])] for p in grid_poly], dtype="int32")
        return cam_poly

    def objectToHull(self, obj):
        """
        Monkey-patched version of objectToHull to ensure consistent coordinate transformation
        between Real Space (object positions) and Map Space (grid visualization).
        """
        all_points_real = []
        
        rsc = self.rsc
        try:
            # Iterate over camera views for this object
            for cam_name, change in obj.changeSet.items():
                if rsc is None:
                    continue
                    
                # Skip cameras not in RSC
                if cam_name not in rsc.converters:
                    continue
                    
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
                            converter = rsc.closestConverterToCamCoord(cam_name, (float(x), float(y)))
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

    def buildMiniMap(self, objectsAndColors: list["MiniMapObject"] = None, hex_cfg = None):
        """
        Monkey-patched replacement for buildMiniMap.
        Uses dynamic canvas size from HexGridConfiguration instead of hardcoded 1200.
        """
        if objectsAndColors is None:
            objectsAndColors = []
            
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
                        if hasattr(self, 'rsc') and self.rsc is not None and cam_name in self.rsc.converters:
                            # Convert polygon points
                            real_pts = []
                            rsc = self.rsc
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

                                converter = rsc.closestConverterToCamCoord(cam_name, (x, y))
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

    def changeSetToAxial(self, changeSet):
        return self.pixel_to_axial(*self.rsc.changeSetToRealCenter(changeSet))

    def build_camera_grid_overlay(self, cam):
        """
        Draws the hex grid dynamically based on the camera's FOV in real space.
        Prevents clipping issues caused by fixed-size minimaps.
        Monkey-patched into HexCaptureConfiguration.
        """
        rsc = self.rsc
        if rsc is None:
             print(f"DynamicGrid: RSC is None for {cam}")
             return None

        # Get camera frame size
        if cam not in self.cameras:
            print(f"DynamicGrid: Camera {cam} not found")
            return None

            
        height, width = self.cameras[cam].mostRecentFrame.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get converter for this camera
        if str(cam) not in rsc.converters:
            return None
            
        # Define 4 corners of the camera frame
        corners_cam = [(0, 0), (width, 0), (width, height), (0, height)]
        
        # Project to Real Space
        corners_real = []
        for p in corners_cam:
            try:
                converter = rsc.closestConverterToCamCoord(str(cam), p)
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

    @property
    def dynamicGrid(self):
        overlays = getattr(self, "grid_overlays", {})
        if overlays:
            return overlays
        self.grid_overlays = {cam: self.build_camera_grid_overlay(cam) for cam in self.cameras.keys()}
        return self.grid_overlays

    def cameraGriddle(self, cam, objectsAndColors=None):
        """
        Monkey-patched replacement for cameraGriddle.
        Uses dynamic grid generation when no objects are present.
        """
        if objectsAndColors is None:
            objectsAndColors = []

        if self.rsc is None:
            height, width = self.cameras[cam].mostRecentFrame.shape[:2]
            return np.zeros((height, width, 3), dtype="uint8")

        # If no objects, use the dynamic grid for better quality/coverage
        if not objectsAndColors:
            return self.dynamicGrid[cam]
        
        # Legacy/Object path (clipped to 1200x1200mm usually)
        # This path is still used if objects need to be drawn
        rsc = self.rsc
        try:
            warped = np.zeros((height, width, 3), dtype="uint8")
            minimap = self.buildMiniMap(objectsAndColors=objectsAndColors)
            if rsc is not None and cam in rsc.converters:
                for converter in rsc.converters[cam]:
                    M = converter.M
                    Minv = np.linalg.inv(M)
    
                    w_part = cv2.warpPerspective(
                        minimap,
                        Minv,
                        (width, height),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    warped = np.maximum(warped, w_part)
            return warped[:height, :width]
        except Exception as e:
            print(f"Error in legacy cameraGriddle: {e}")
            return np.zeros((height, width, 3), dtype="uint8")

    def griddleCameras(self, objectsAndColors=[], alpha=0.3):
        blended = {}
        img = self.buildMiniMap(objectsAndColors=objectsAndColors)

        if self.rsc is None:
            return {}

        for cam in self.cameras.keys():
            cameraChanges = self.cameras[cam].cropToActiveZone(self.cameras[cam].mostRecentFrame.copy())
            w, h = cameraChanges.shape[:2]
            warped = np.zeros((h, w, 3), dtype="uint8")
            for converter in self.rsc.converters[cam]:
                M = converter.M
                Minv = np.linalg.inv(M)

                # Use nearest for crisp overlays; use linear for smoother images
                w_part = cv2.warpPerspective(
                    img,
                    Minv,
                    (h, w),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                warped = np.maximum(warped, w_part)
            blended[cam] = self.cameras[cam].cropToActiveZone(cv2.addWeighted(warped[:w, :h], alpha, cameraChanges, 1.0 - alpha, 0.0))
        return blended

    def axialDistanceBetweenObjects(self, changeSetA, changeSetB):
        real_coord_a = self.rsc.changeSetToRealCenter(changeSetA)
        real_coord_b = self.rsc.changeSetToRealCenter(changeSetB)
        axial_coord_a = self.pixel_to_axial(*real_coord_a)
        axial_coord_b = self.pixel_to_axial(*real_coord_b)
        return self.axial_distance(axial_coord_a, axial_coord_b)

    def define_object_from_axial(self, oid, q, r):
        if self.rsc is None:
            raise Exception("Cannot define an object without a RealSpace Configuration")

        rsc = self.rsc
        realSpacePoly = self.hex_at_axial(q, r)

        changeSet = {}
        for cam in self.cameras.keys():
            center_real = self.axial_to_pixel(q, r)
            converter = rsc.closestConverterToRealCoord(cam, center_real)
            cam_poly = np.array([[int(d) for d in converter.convertRealToCameraSpace(p[0])] for p in realSpacePoly], dtype="int32")
            changeSet[cam] = CameraChange(
                camName=cam,
                changeContours=[cam_poly.reshape((-1, 1, 2))],
                before=self.cameras[cam].mostRecentFrame,
                after=self.cameras[cam].mostRecentFrame,
                changeType = "add",
                lastChange = None,
            )

        return TrackedObject(oid=oid, changeSet=changeSet)

    def define_object_from_axials(self, oid: str, axials: list[tuple[int]]):
        if self.rsc is None:
            raise Exception("Cannot define an object without a RealSpace Configuration")
        if not axials:
            raise ValueError("axials must be non-empty")

        rsc = self.rsc
        changeSet = {}

        for cam in self.cameras.keys():
            # Use current frame size for the mask
            frame = self.cameras[cam].mostRecentFrame
            if frame is None:
                raise Exception(f"No mostRecentFrame for camera {cam}")
            H, W = frame.shape[:2]

            mask = np.zeros((H, W), dtype=np.uint8)

            # 1) Project each axial hex to camera space and fill it into the mask
            for (q, r) in axials:
                real_hex = self.hex_at_axial(q, r)  # iterable of points, often (6,1,2) or (6,2)
                
                center_real = self.axial_to_pixel(q, r)
                conv = rsc.closestConverterToRealCoord(cam, center_real)

                cam_pts = []
                for p in real_hex:
                    p = p[0] if (hasattr(p, "__len__") and len(p) == 1 and len(p[0]) == 2) else p
                    px, py = conv.convertRealToCameraSpace((p[0], p[1]))
                    cam_pts.append([int(round(px)), int(round(py))])

                poly = np.array(cam_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 255)

            # 2) Extract contours with hierarchy so holes are preserved
            # RETR_CCOMP gives a two-level hierarchy: outer contours + hole contours.
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                # No pixels filled (maybe all projected off-frame)
                changeContours = []
            else:
                # Optional: simplify contours a bit (reduce jaggies)
                changeContours = []
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    # epsilon in pixels; tweak if needed
                    eps = 1.0
                    approx = cv2.approxPolyDP(cnt, eps, closed=True)
                    if len(approx) >= 3:
                        changeContours.append(approx.astype(np.int32))

            changeSet[cam] = CameraChange(
                camName=cam,
                changeContours=changeContours,
                before=frame,
                after=frame,
                changeType="add",
                lastChange=None,
            )

        return TrackedObject(oid=oid, changeSet=changeSet)




if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    hc = HexCaptureConfiguration()
    hc.hex = HexGridConfiguration()
    co = CalibratedObserver(hc)
    co.cycle()
    plt.imshow(co.buildMiniMap(objectsAndColors=[MiniMapObject(mem, (255, 0, 0)) for mem in co.memory]))
    plt.show()
    # for i in range(2):
    #     co.cycleForChange()
    #     plt.imshow(co.buildMiniMap(objectsAndColors=[MiniMapObject(mem, (255, 0, 0)) for mem in co.memory]))
    #     plt.show()
    # a = hc.changeSetToAxial(co.memory[0])
    # b = hc.changeSetToAxial(co.memory[1])
    # print(f"Axial Distance between objects: {hc.axial_distance(a, b)}")
    img = hc.griddleCameras(objectsAndColors=[MiniMapObject(mem, (255, 0, 0)) for mem in co.memory])['0']
    plt.imshow(img)

    try:
        foo = hc.define_object_from_axial("foo", -1, 9)
    except Exception as e:
        print(f"Failed: {e}")


# In[ ]:




