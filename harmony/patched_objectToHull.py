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
            # Note: rsc.converters is {cam_name: [converter, ...]}
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
                    # rsc.camCoordToRealSpace wraps converter but acts on single point
                    # We can use the converter directly for speed/robustness
                    # Convert floats
                    real_pt = converter.convertCameraToRealSpace((float(x), float(y)))
                    all_points_real.append(real_pt)
                    
        if not all_points_real:
            # Fallback or empty
            return np.array([], dtype=np.int32)
            
        # Transform Real Space -> Map Pixel Space
        # using the SAME affine transform as draw_grid
        pts_real = np.array(all_points_real, dtype=np.float32)
        
        if hasattr(self, 'apply_affine_pts'):
            pts_map = self.apply_affine_pts(pts_real)
        else:
            # Fallback if method missing (shouldn't happen)
            print("Warning: apply_affine_pts missing in patched_objectToHull")
            pts_map = pts_real 
            
        # Compute Convex Hull of mapped points
        pts_map_i = np.round(pts_map).astype(np.int32)
        hull = cv2.convexHull(pts_map_i)
        
        return hull
        
    except Exception as e:
        print(f"Error in patched_objectToHull: {e}")
        return np.array([], dtype=np.int32)
