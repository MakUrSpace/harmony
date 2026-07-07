async fn object_snapshot_feed(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<AppState>>,
    axum::extract::Path((oid, cam_name)): axum::extract::Path<(String, String)>,
) -> impl axum::response::IntoResponse {
    tracing::info!("Route hit: /harmony/objects/{}/snapshot/{}", oid, cam_name);
    
    let mut rx = None;
    let mut bounding_box = None;
    
    {
        let machine = state.machine.lock().await;
        if let Some(obj) = machine.memory.get(&oid) {
            if let Some(cam) = machine.cc.cameras.get(&cam_name) {
                rx = cam.raw_frame_rx.clone();
                if let Some(rsc) = machine.cc.rsc.get(&cam_name) {
                    if let Some(hex) = &machine.cc.hex {
                        let mut min_x = f64::MAX;
                        let mut min_y = f64::MAX;
                        let mut max_x = f64::MIN;
                        let mut max_y = f64::MIN;
                        
                        for (q, r) in &obj.constituent_axials {
                            let center = hex.axial_to_pixel(*q as f64, *r as f64);
                            if let Ok(cam_pt) = rsc.convert_real_to_camera_space(center) {
                                if cam_pt.0 < min_x { min_x = cam_pt.0; }
                                if cam_pt.0 > max_x { max_x = cam_pt.0; }
                                if cam_pt.1 < min_y { min_y = cam_pt.1; }
                                if cam_pt.1 > max_y { max_y = cam_pt.1; }
                            }
                        }
                        
                        if min_x != f64::MAX {
                            bounding_box = Some((min_x, min_y, max_x, max_y));
                        }
                    }
                }
            }
        }
    }
    
    let bounding_box = match bounding_box {
        Some(bb) => bb,
        None => return (axum::http::StatusCode::NOT_FOUND, "Could not map object to camera").into_response(),
    };
    
    if let Some(rx) = rx {
        let frame_bytes = rx.borrow().clone();
        if frame_bytes.is_empty() {
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "No frame").into_response();
        }
        
        let mat = match opencv::imgcodecs::imdecode(&opencv::core::Vector::from_slice(&frame_bytes), opencv::imgcodecs::IMREAD_COLOR) {
            Ok(m) => m,
            Err(_) => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to decode frame").into_response(),
        };
        
        let margin = 20.0;
        let mut x = (bounding_box.0 - margin).round() as i32;
        let mut y = (bounding_box.1 - margin).round() as i32;
        let mut w = (bounding_box.2 - bounding_box.0 + 2.0 * margin).round() as i32;
        let mut h = (bounding_box.3 - bounding_box.1 + 2.0 * margin).round() as i32;
        
        let frame_cols = mat.cols();
        let frame_rows = mat.rows();
        
        if x < 0 { w += x; x = 0; }
        if y < 0 { h += y; y = 0; }
        if x + w > frame_cols { w = frame_cols - x; }
        if y + h > frame_rows { h = frame_rows - y; }
        
        if w <= 0 || h <= 0 {
            return (axum::http::StatusCode::BAD_REQUEST, "Object outside frame").into_response();
        }
        
        let roi = match opencv::core::Mat::roi(&mat, opencv::core::Rect::new(x, y, w, h)) {
            Ok(r) => r,
            Err(_) => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to crop").into_response(),
        };
        
        let mut buf = opencv::core::Vector::<u8>::new();
        let params = opencv::core::Vector::<i32>::new();
        match opencv::imgcodecs::imencode(".jpg", &roi, &mut buf, &params) {
            Ok(true) => {
                let bytes = buf.to_vec();
                return (
                    axum::http::StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
                    bytes
                ).into_response();
            },
            _ => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to encode crop").into_response(),
        }
    }
    
    (axum::http::StatusCode::NOT_FOUND, "No frame source").into_response()
}
