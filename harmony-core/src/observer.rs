use opencv::core::{Mat, Point, Rect};
use opencv::prelude::*;

pub struct Camera {
    pub name: String,
    pub active_zone: Vec<Point>,
    pub rotate: bool,
    pub image_buffer: Vec<Mat>,
    pub reference_frame: Option<Mat>,
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
            // ... further area / center calculations can go here
            Ok(Some(change))
        } else {
            Ok(None)
        }
    }
}

use std::collections::HashMap;

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
        (x, y)
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
    pub hex: Option<HexGridConfiguration>,
    pub show_grid: bool,
    pub show_objects: bool,
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

