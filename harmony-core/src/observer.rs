use opencv::core::{Mat, Point, Rect, Point2f, Vector};
use opencv::prelude::*;
use std::collections::HashMap;

pub struct Camera {
    pub name: String,
    pub cam_path: String,
    pub active_zone: Vec<Point>,
    pub rotate: bool,
    pub auth: Vec<String>,
    pub image_buffer: Vec<Mat>,
    pub reference_frame: Option<Mat>,
    pub frame_rx: Option<tokio::sync::watch::Receiver<Vec<u8>>>,
    pub frame_tx: Option<tokio::sync::watch::Sender<Vec<u8>>>,
    pub raw_frame_rx: Option<tokio::sync::watch::Receiver<Vec<u8>>>,
    pub raw_frame_tx: Option<tokio::sync::watch::Sender<Vec<u8>>>,
}



impl Camera {
    pub fn crop_to_active_zone(&self, image: &Mat) -> opencv::Result<Mat> {
        let mut pts = opencv::core::Vector::<Point>::new();
        for p in &self.active_zone {
            pts.push(*p);
        }
        let mut contours = opencv::core::Vector::<opencv::core::Vector<Point>>::new();
        contours.push(pts);

        let size = image.size()?;
        let mut mask = Mat::new_rows_cols_with_default(
            size.height,
            size.width,
            opencv::core::CV_8UC1,
            opencv::core::Scalar::all(0.0),
        )?;

        opencv::imgproc::draw_contours(
            &mut mask,
            &contours,
            -1,
            opencv::core::Scalar::all(255.0),
            -1,
            opencv::imgproc::LINE_AA,
            &opencv::core::no_array(),
            2147483647,
            Point::new(0, 0),
        )?;

        let mut dst = Mat::default();
        opencv::core::bitwise_and(image, image, &mut dst, &mask)?;

        Ok(dst)
    }

    pub fn mask_frame_to_active_zone(&self, frame: &Mat) -> opencv::Result<Mat> {
        let size = frame.size()?;
        let mut mask = Mat::new_rows_cols_with_default(
            size.height,
            size.width,
            opencv::core::CV_8UC1,
            opencv::core::Scalar::all(0.0),
        )?;

        let mut pts = opencv::core::Vector::<Point>::new();
        for p in &self.active_zone {
            pts.push(*p);
        }
        let mut contours = opencv::core::Vector::<opencv::core::Vector<Point>>::new();
        contours.push(pts);

        opencv::imgproc::fill_poly(
            &mut mask,
            &contours,
            opencv::core::Scalar::all(255.0),
            opencv::imgproc::LINE_8,
            0,
            Point::new(0, 0),
        )?;

        let mut masked = Mat::default();
        opencv::core::bitwise_and(frame, frame, &mut masked, &mask)?;
        Ok(masked)
    }

    pub fn contours_between(&self, im0: &Mat, im1: &Mat) -> opencv::Result<opencv::core::Vector<opencv::core::Vector<Point>>> {
        let mut diff = Mat::default();
        let mut im0_gray = Mat::default();
        let mut im1_gray = Mat::default();

        opencv::imgproc::cvt_color(im0, &mut im0_gray, opencv::imgproc::COLOR_BGR2GRAY, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        opencv::imgproc::cvt_color(im1, &mut im1_gray, opencv::imgproc::COLOR_BGR2GRAY, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        opencv::core::absdiff(&im0_gray, &im1_gray, &mut diff)?;

        let mut blurred = Mat::default();
        opencv::imgproc::gaussian_blur(
            &diff,
            &mut blurred,
            opencv::core::Size::new(15, 25),
            0.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut thresh = Mat::default();
        opencv::imgproc::threshold(
            &blurred,
            &mut thresh,
            20.0,
            255.0,
            opencv::imgproc::THRESH_BINARY,
        )?;

        let kernel = opencv::imgproc::get_structuring_element(
            opencv::imgproc::MORPH_RECT,
            opencv::core::Size::new(3, 3),
            Point::new(-1, -1),
        )?;
        let mut dilate = Mat::default();

        opencv::imgproc::dilate(
            &thresh,
            &mut dilate,
            &kernel,
            Point::new(-1, -1),
            2,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )?;

        let mut contours = opencv::core::Vector::<opencv::core::Vector<Point>>::new();
        opencv::imgproc::find_contours(
            &dilate,
            &mut contours,
            opencv::imgproc::RETR_EXTERNAL,
            opencv::imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        Ok(contours)
    }

    pub fn change_between(&self, change_frame: &Mat, reference_frame: &Mat) -> opencv::Result<Option<CameraChange>> {
        let masked_ref = self.mask_frame_to_active_zone(reference_frame)?;
        let masked_change = self.mask_frame_to_active_zone(change_frame)?;
        let contours = self.contours_between(&masked_ref, &masked_change)?;

        let mut filtered_contours = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let b_rect = opencv::imgproc::bounding_rect(&contour)?;
            let area = (b_rect.width * b_rect.height) as f64;
            
            if area > 1000.0 && area < 100000.0 {
                let mut c_vec = Vec::new();
                for j in 0..contour.len() {
                    c_vec.push(contour.get(j)?);
                }
                filtered_contours.push(c_vec);
            }
        }

        if !filtered_contours.is_empty() {
            let mut change = CameraChange::new(self.name.clone(), reference_frame.clone(), change_frame.clone());
            change.change_contours = filtered_contours;
            Ok(Some(change))
        } else {
            Ok(None)
        }
    }
}

pub struct CameraRealSpaceConverter {
    pub m: Mat,
    pub m_inv: Mat,
    pub cam_pts: Vec<(f64, f64)>,
    pub real_pts: Vec<(f64, f64)>,
}

impl CameraRealSpaceConverter {
    pub fn new(cam_pts: &Vec<(f64, f64)>, real_pts: &Vec<(f64, f64)>) -> opencv::Result<Self> {
        let mut src_pts = Vector::<Point2f>::new();
        for p in cam_pts {
            src_pts.push(Point2f::new(p.0 as f32, p.1 as f32));
        }
        let mut dst_pts = Vector::<Point2f>::new();
        for p in real_pts {
            dst_pts.push(Point2f::new(p.0 as f32, p.1 as f32));
        }

        let mut no_array = opencv::core::no_array();
        let m = opencv::calib3d::find_homography(
            &src_pts,
            &dst_pts,
            &mut no_array,
            0,
            3.0
        )?;
        
        let m_inv = m.inv(opencv::core::DECOMP_LU)?.to_mat()?;
        Ok(Self { 
            m, 
            m_inv,
            cam_pts: cam_pts.clone(),
            real_pts: real_pts.clone(),
        })
    }

    pub fn convert_camera_to_real_space(&self, pt: (f64, f64)) -> opencv::Result<(f64, f64)> {
        let m_ptr = self.m.data_typed::<f64>()?;
        let px = pt.0;
        let py = pt.1;
        let den = m_ptr[6] * px + m_ptr[7] * py + m_ptr[8];
        let x = (m_ptr[0] * px + m_ptr[1] * py + m_ptr[2]) / den;
        let y = (m_ptr[3] * px + m_ptr[4] * py + m_ptr[5]) / den;
        Ok((x, y))
    }

    pub fn convert_real_to_camera_space(&self, pt: (f64, f64)) -> opencv::Result<(f64, f64)> {
        let m_inv_ptr = self.m_inv.data_typed::<f64>()?;
        let px = pt.0;
        let py = pt.1;
        let den = m_inv_ptr[6] * px + m_inv_ptr[7] * py + m_inv_ptr[8];
        let x = (m_inv_ptr[0] * px + m_inv_ptr[1] * py + m_inv_ptr[2]) / den;
        let y = (m_inv_ptr[3] * px + m_inv_ptr[4] * py + m_inv_ptr[5]) / den;
        Ok((x, y))
    }

    pub fn cam_centroid(&self) -> (f64, f64) {
        let sum_x: f64 = self.cam_pts.iter().map(|p| p.0).sum();
        let sum_y: f64 = self.cam_pts.iter().map(|p| p.1).sum();
        let len = self.cam_pts.len() as f64;
        (sum_x / len, sum_y / len)
    }

    pub fn real_centroid(&self) -> (f64, f64) {
        let sum_x: f64 = self.real_pts.iter().map(|p| p.0).sum();
        let sum_y: f64 = self.real_pts.iter().map(|p| p.1).sum();
        let len = self.real_pts.len() as f64;
        (sum_x / len, sum_y / len)
    }
}

pub struct RealSpaceConverter {
    pub converters: Vec<CameraRealSpaceConverter>,
}

impl RealSpaceConverter {
    pub fn new(converters: Vec<CameraRealSpaceConverter>) -> Self {
        Self { converters }
    }

    pub fn closest_converter_to_cam_coord(&self, pt: (f64, f64)) -> Option<&CameraRealSpaceConverter> {
        self.converters.iter().min_by(|a, b| {
            let dist_a = (a.cam_centroid().0 - pt.0).powi(2) + (a.cam_centroid().1 - pt.1).powi(2);
            let dist_b = (b.cam_centroid().0 - pt.0).powi(2) + (b.cam_centroid().1 - pt.1).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn closest_converter_to_real_coord(&self, pt: (f64, f64)) -> Option<&CameraRealSpaceConverter> {
        self.converters.iter().min_by(|a, b| {
            let dist_a = (a.real_centroid().0 - pt.0).powi(2) + (a.real_centroid().1 - pt.1).powi(2);
            let dist_b = (b.real_centroid().0 - pt.0).powi(2) + (b.real_centroid().1 - pt.1).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn convert_camera_to_real_space(&self, pt: (f64, f64)) -> opencv::Result<(f64, f64)> {
        if let Some(closest) = self.closest_converter_to_cam_coord(pt) {
            closest.convert_camera_to_real_space(pt)
        } else {
            Err(opencv::Error::new(opencv::core::StsBadArg, "No converters available".to_string()))
        }
    }

    pub fn convert_real_to_camera_space(&self, pt: (f64, f64)) -> opencv::Result<(f64, f64)> {
        if let Some(closest) = self.closest_converter_to_real_coord(pt) {
            closest.convert_real_to_camera_space(pt)
        } else {
            Err(opencv::Error::new(opencv::core::StsBadArg, "No converters available".to_string()))
        }
    }
}

#[derive(Clone)]
pub struct HexGridConfiguration {
    pub size: f64,
    pub rotation_deg: f64,
    pub offset_xy: (f64, f64),
    pub anchor_xy: (i32, i32),
    pub width: i32,
    pub height: i32,
}

impl Default for HexGridConfiguration {
    fn default() -> Self {
        let size = 38.0;
        Self {
            size,
            rotation_deg: 0.0,
            offset_xy: (0.0, 0.0),
            anchor_xy: (-(size / 2.86 + 0.25) as i32, -(size / 2.0) as i32),
            width: 1600,
            height: 1600,
        }
    }
}

impl HexGridConfiguration {
    pub fn axial_to_pixel(&self, q: f64, r: f64) -> (f64, f64) {
        let x = self.size * 3.0_f64.sqrt() * (q + r / 2.0);
        let y = self.size * 1.5 * r;

        let angle_rad = self.rotation_deg.to_radians();
        let rotated_x = x * angle_rad.cos() - y * angle_rad.sin();
        let rotated_y = x * angle_rad.sin() + y * angle_rad.cos();

        (rotated_x + self.offset_xy.0, rotated_y + self.offset_xy.1)
    }

    pub fn pixel_to_axial_frac(&self, px: f64, py: f64) -> (f64, f64) {
        let x = px - self.offset_xy.0;
        let y = py - self.offset_xy.1;

        let angle_rad = (-self.rotation_deg).to_radians();
        let unrotated_x = x * angle_rad.cos() - y * angle_rad.sin();
        let unrotated_y = x * angle_rad.sin() + y * angle_rad.cos();

        let r = (2.0 / 3.0) * (unrotated_y / self.size);
        let q = (unrotated_x / (3.0_f64.sqrt() * self.size)) - (r / 2.0);
        (q, r)
    }

    pub fn pixel_to_axial(&self, px: f64, py: f64) -> (i32, i32) {
        let (q, r) = self.pixel_to_axial_frac(px, py);
        Self::axial_round(q, r)
    }

    pub fn axial_round(q: f64, r: f64) -> (i32, i32) {
        let x = q;
        let z = r;
        let y = -x - z;

        let mut rx = x.round() as i32;
        let mut ry = y.round() as i32;
        let mut rz = z.round() as i32;

        let dx = (rx as f64 - x).abs();
        let dy = (ry as f64 - y).abs();
        let dz = (rz as f64 - z).abs();

        if dx > dy && dx > dz {
            rx = -ry - rz;
        } else if dy > dz {
            ry = -rx - rz;
        } else {
            rz = -rx - ry;
        }

        (rx, rz)
    }
}

pub struct HexCaptureConfiguration {
    pub cameras: HashMap<String, Camera>,
    pub rsc: HashMap<String, RealSpaceConverter>,
    pub hex: Option<HexGridConfiguration>,
    pub show_grid: bool,
    pub show_objects: bool,
    pub grid_polys_cache: HashMap<String, serde_json::Value>,
    pub calibration_plan: serde_json::Value,
}

impl HexCaptureConfiguration {
    pub fn load_from_file(path: &str) -> opencv::Result<Self> {
        let mut cameras = HashMap::new();
        let mut rsc = HashMap::new();
        let mut hex = Some(HexGridConfiguration::default());
        let mut show_grid = true;
        let mut show_objects = true;
        let mut calibration_plan = serde_json::json!({});
        
        let file_contents = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&file_contents) {
            if let Some(hex_obj) = json.get("hex").and_then(|v| v.as_object()) {
                let size = hex_obj.get("size").and_then(|v| v.as_f64()).unwrap_or(38.0);
                let rotation_deg = hex_obj.get("rotation_deg").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let offset_xy = hex_obj.get("offset_xy").and_then(|v| v.as_array())
                    .map(|arr| (arr[0].as_f64().unwrap_or(0.0), arr[1].as_f64().unwrap_or(0.0)))
                    .unwrap_or((0.0, 0.0));
                let anchor_xy = hex_obj.get("anchor_xy").and_then(|v| v.as_array())
                    .map(|arr| (arr[0].as_i64().unwrap_or(0) as i32, arr[1].as_i64().unwrap_or(0) as i32))
                    .unwrap_or((0, 0));
                let width = hex_obj.get("width").and_then(|v| v.as_i64()).unwrap_or(1600) as i32;
                let height = hex_obj.get("height").and_then(|v| v.as_i64()).unwrap_or(1600) as i32;
                hex = Some(HexGridConfiguration { size, rotation_deg, offset_xy, anchor_xy, width, height });
            }

            if let Some(v) = json.get("show_grid").and_then(|v| v.as_bool()) {
                show_grid = v;
            }
            if let Some(v) = json.get("show_objects").and_then(|v| v.as_bool()) {
                show_objects = v;
            }

            // Parse Cameras
            if let Some(obj) = json.as_object() {
                for (k, v) in obj {
                    if let Some(cam_obj) = v.as_object() {
                        if cam_obj.contains_key("addr") {
                            let cam_path = cam_obj.get("addr").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let rotate = cam_obj.get("rot").and_then(|v| v.as_bool()).unwrap_or(false);
                            let az_str = cam_obj.get("az").and_then(|v| v.as_str()).unwrap_or("[]");
                            let mut active_zone = Vec::new();
                            if let Ok(az_arr) = serde_json::from_str::<Vec<[f64; 2]>>(az_str) {
                                for p in az_arr {
                                    active_zone.push(Point::new(p[0] as i32, p[1] as i32));
                                }
                            }
                            let mut auth = Vec::new();
                            if let Some(auth_arr) = cam_obj.get("auth").and_then(|v| v.as_array()) {
                                for item in auth_arr {
                                    if let Some(s) = item.as_str() {
                                        auth.push(s.to_string());
                                    }
                                }
                            }
                            
                            let (tx, rx) = tokio::sync::watch::channel(vec![]);
                            let (raw_tx, raw_rx) = tokio::sync::watch::channel(vec![]);
                            let cam = Camera {
                                name: k.clone(),
                                cam_path: cam_path.clone(),
                                active_zone,
                                rotate,
                                auth,
                                image_buffer: Vec::new(),
                                reference_frame: None,
                                frame_rx: Some(rx),
                                frame_tx: Some(tx),
                                raw_frame_rx: Some(raw_rx),
                                raw_frame_tx: Some(raw_tx),
                            };
                            cameras.insert(k.clone(), cam);
                        }
                    }
                }
            }

            // Parse RSC
            if let Some(rsc_array) = json.get("rsc").and_then(|v| v.as_array()) {
                let mut temp_rsc: HashMap<String, Vec<CameraRealSpaceConverter>> = HashMap::new();
                for item in rsc_array {
                    if let Some(arr) = item.as_array() {
                        if arr.len() == 2 {
                            let cam_id_str = arr[0].as_str().unwrap_or("").to_string();
                            let cam_id = cam_id_str.replace("RTSPCamera", "");
                            
                            if let Some(pts_arr) = arr[1].as_array() {
                                if pts_arr.len() >= 2 {
                                    let mut cam_pts = Vec::new();
                                    let mut real_pts = Vec::new();
                                    
                                    if let Some(c_pts) = pts_arr[0].as_array() {
                                        for cp in c_pts {
                                            if let Some(c_arr) = cp.as_array() {
                                                cam_pts.push((c_arr[0].as_f64().unwrap(), c_arr[1].as_f64().unwrap()));
                                            }
                                        }
                                    }
                                    if let Some(r_pts) = pts_arr[1].as_array() {
                                        for rp in r_pts {
                                            if let Some(r_arr) = rp.as_array() {
                                                real_pts.push((r_arr[0].as_f64().unwrap(), r_arr[1].as_f64().unwrap()));
                                            }
                                        }
                                    }
                                    
                                    if cam_pts.len() >= 4 && real_pts.len() >= 4 {
                                        if let Ok(converter) = CameraRealSpaceConverter::new(&cam_pts, &real_pts) {
                                            temp_rsc.entry(cam_id).or_insert_with(Vec::new).push(converter);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (cam_id, converters) in temp_rsc {
                    rsc.insert(cam_id, RealSpaceConverter::new(converters));
                }
            } // close rsc_array if let
            
            if let Some(cp) = json.get("calibrationPlan") {
                calibration_plan = cp.clone();
            }
        }

        let mut config = Self {
            cameras,
            rsc,
            hex,
            show_grid,
            show_objects,
            grid_polys_cache: HashMap::new(),
            calibration_plan,
        };

        config.precompute_grid_polys();

        Ok(config)
    }

    pub fn save_to_file(&self, path: &str) -> opencv::Result<()> {
        let mut map = serde_json::Map::new();

        // Save cameras
        for (name, cam) in &self.cameras {
            let mut cam_obj = serde_json::Map::new();
            cam_obj.insert("addr".to_string(), serde_json::json!(cam.cam_path));
            cam_obj.insert("rot".to_string(), serde_json::json!(cam.rotate));
            let az_pts: Vec<[i32; 2]> = cam.active_zone.iter().map(|p| [p.x, p.y]).collect();
            cam_obj.insert("az".to_string(), serde_json::json!(serde_json::to_string(&az_pts).unwrap_or("[]".to_string())));
            cam_obj.insert("auth".to_string(), serde_json::json!(cam.auth));
            map.insert(name.clone(), serde_json::Value::Object(cam_obj));
        }

        // Save rsc
        let mut rsc_array = Vec::new();
        for (name, rsc_manager) in &self.rsc {
            for converter in &rsc_manager.converters {
                let mut cam_pts = Vec::new();
                for p in &converter.cam_pts {
                    cam_pts.push(serde_json::json!([p.0, p.1]));
                }
                let mut real_pts = Vec::new();
                for p in &converter.real_pts {
                    real_pts.push(serde_json::json!([p.0, p.1]));
                }
                rsc_array.push(serde_json::json!([name, [cam_pts, real_pts]]));
            }
        }
        map.insert("rsc".to_string(), serde_json::Value::Array(rsc_array));

        // Save hex
        if let Some(hex_cfg) = &self.hex {
            let mut hex_obj = serde_json::Map::new();
            hex_obj.insert("size".to_string(), serde_json::json!(hex_cfg.size));
            hex_obj.insert("rotation_deg".to_string(), serde_json::json!(hex_cfg.rotation_deg));
            hex_obj.insert("offset_xy".to_string(), serde_json::json!([hex_cfg.offset_xy.0, hex_cfg.offset_xy.1]));
            hex_obj.insert("anchor_xy".to_string(), serde_json::json!([hex_cfg.anchor_xy.0, hex_cfg.anchor_xy.1]));
            hex_obj.insert("width".to_string(), serde_json::json!(hex_cfg.width));
            hex_obj.insert("height".to_string(), serde_json::json!(hex_cfg.height));
            hex_obj.insert("hex_nudges".to_string(), serde_json::json!({}));
            map.insert("hex".to_string(), serde_json::Value::Object(hex_obj));
        }

        map.insert("show_grid".to_string(), serde_json::json!(self.show_grid));
        map.insert("show_objects".to_string(), serde_json::json!(self.show_objects));
        map.insert("calibrationPlan".to_string(), self.calibration_plan.clone());
        map.insert("show_objects".to_string(), serde_json::json!(self.show_objects));

        // grid_overlays is not currently synced, we can stub it if needed, or omit it (Python doesn't strictly require it to be saved).
        map.insert("grid_overlays".to_string(), serde_json::json!({}));

        let json_str = serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string());
        std::fs::write(path, json_str).map_err(|e| opencv::Error::new(opencv::core::StsError, format!("Failed to write config: {}", e)))?;

        Ok(())
    }

    fn point_in_polygon(pt: (f64, f64), poly: &Vec<opencv::core::Point>) -> bool {
        let mut inside = false;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let (xi, yi) = (poly[i].x as f64, poly[i].y as f64);
            let (xj, yj) = (poly[j].x as f64, poly[j].y as f64);
            let intersect = ((yi > pt.1) != (yj > pt.1)) &&
                (pt.0 < (xj - xi) * (pt.1 - yi) / (yj - yi) + xi);
            if intersect {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    pub fn precompute_grid_polys(&mut self) {
        if let Some(hex_cfg) = &self.hex {
            for (cam_name, converter) in &self.rsc {
                let mut data_polys = Vec::new();

                let q_min = -30;
                let q_max = 30;
                let r_min = -30;
                let r_max = 30;

                for q in q_min..=q_max {
                    for r in r_min..=r_max {
                        let center = hex_cfg.axial_to_pixel(q as f64, r as f64);
                        let mut corners = Vec::new();
                        for k in 0..6 {
                            let ang = (60.0 * k as f64 - 30.0).to_radians();
                            let cx = hex_cfg.size * ang.cos();
                            let cy = hex_cfg.size * ang.sin();
                            let angle_rad = hex_cfg.rotation_deg.to_radians();
                            let rotated_cx = cx * angle_rad.cos() - cy * angle_rad.sin();
                            let rotated_cy = cx * angle_rad.sin() + cy * angle_rad.cos();
                            corners.push((
                                center.0 + rotated_cx,
                                center.1 + rotated_cy,
                            ));
                        }
                        
                        let mut cam_corners = Vec::new();
                        let mut in_bounds = false;
                        for corner in corners {
                            if let Ok(cam_pt) = converter.convert_real_to_camera_space(corner) {
                                if cam_pt.0 >= -100.0 && cam_pt.0 <= 2660.0 && cam_pt.1 >= -100.0 && cam_pt.1 <= 2020.0 {
                                    in_bounds = true;
                                }
                                cam_corners.push(serde_json::json!([cam_pt.0 as i32, cam_pt.1 as i32]));
                            }
                        }
                        
                        if in_bounds && cam_corners.len() == 6 {
                            let mut valid = true;
                            if let Some(cam) = self.cameras.get(cam_name) {
                                if !cam.active_zone.is_empty() {
                                    if let Ok(cam_center) = converter.convert_real_to_camera_space(center) {
                                        valid = Self::point_in_polygon(cam_center, &cam.active_zone);
                                    } else {
                                        valid = false;
                                    }
                                }
                            }
                            if valid {
                                data_polys.push(serde_json::json!({
                                    "q": q,
                                    "r": r,
                                    "poly": cam_corners
                                }));
                            }
                        }
                    }
                }
                
                self.grid_polys_cache.insert(cam_name.clone(), serde_json::Value::Array(data_polys));
            }
        }
    }
}

#[derive(Debug)]
pub enum ChangeType {
    Add,
    Move,
    Delete,
    Unclassified,
}

pub struct CameraChange {
    pub cam_name: String,
    pub change_contours: Vec<Vec<Point>>,
    pub before: Mat,
    pub after: Mat,
    pub change_type: ChangeType,
    pub area: f64,
    pub change_points: Vec<Point>,
    pub corner: Point,
    pub width: i32,
    pub height: i32,
    pub clip_box: Rect,
    pub center: Point,
}

impl CameraChange {
    pub fn new(cam_name: String, before: Mat, after: Mat) -> Self {
        Self {
            cam_name,
            change_contours: Vec::new(),
            before,
            after,
            change_type: ChangeType::Unclassified,
            area: 0.0,
            change_points: Vec::new(),
            corner: Point::new(0, 0),
            width: 0,
            height: 0,
            clip_box: Rect::default(),
            center: Point::new(0, 0),
        }
    }
}
