use askama::Template;
use axum::{
    extract::{State, Query},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Router,
};
use axum_extra::extract::cookie::{Cookie, CookieJar};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use std::net::SocketAddr;
use tower_http::services::ServeDir;
use harmony_core::machine::HarmonyMachine;
use harmony_core::observer::{CameraChange, HexCaptureConfiguration};
use opencv::prelude::MatTraitConst;
use opencv::prelude::VectorToVec;

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellSelection {
    pub first_cell: Option<(i32, i32)>,
    pub additional_cells: Vec<(i32, i32)>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    pub moveable: Vec<String>,
    pub selectable: Vec<String>,
    pub terrain: Vec<String>,
    pub allies: Vec<String>,
    pub enemies: Vec<String>,
    pub targetable: Vec<String>,
    pub selection: CellSelection,
    pub selected_oid: Option<String>,
    pub show_grid: bool,
    pub show_objects: bool,
    pub can_publish_selection: bool,
    pub published_selection: Option<Vec<(i32, i32)>>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            moveable: vec![],
            selectable: vec![],
            terrain: vec![],
            allies: vec![],
            enemies: vec![],
            targetable: vec![],
            selection: CellSelection { first_cell: None, additional_cells: vec![] },
            selected_oid: None,
            show_grid: false,
            show_objects: true,
            can_publish_selection: true,
            published_selection: None,
        }
    }
}

struct AppState {
    machine: Arc<Mutex<HarmonyMachine>>,
    frame_tx: broadcast::Sender<(String, bytes::Bytes)>,
    sessions: Mutex<std::collections::HashMap<String, SessionConfig>>,
}

fn spawn_camera_stream(
    cam_name: String,
    cam_path: String,
    machine: Arc<tokio::sync::Mutex<HarmonyMachine>>,
    tx: tokio::sync::watch::Sender<Vec<u8>>,
    raw_tx: tokio::sync::watch::Sender<Vec<u8>>
) {
    tokio::task::spawn_blocking(move || {
        use opencv::videoio::{VideoCapture, CAP_ANY};
        use opencv::core::{Mat, Vector};
        use opencv::imgcodecs::imencode;
        use opencv::prelude::*;
        
        let mut final_cam_path = cam_path.clone();
        if !final_cam_path.starts_with("http://") && !final_cam_path.starts_with("https://") && !final_cam_path.starts_with("rtsp://") && !final_cam_path.starts_with("/") && !final_cam_path.is_empty() {
            final_cam_path = format!("rtsp://{}", final_cam_path);
        }

        let mut cam = match VideoCapture::from_file(&final_cam_path, CAP_ANY) {
            Ok(c) => c,
            Err(e) => {
                println!("Failed to open camera: {}", e);
                return;
            }
        };
        
        let mut frame = Mat::default();
        let mut last_active_zone: Vec<opencv::core::Point> = vec![];
        
        loop {
            match cam.read(&mut frame) {
                Ok(true) => {
                    // Update active zone if lock is available
                    if let Ok(m) = machine.try_lock() {
                        if let Some(c) = m.cc.cameras.get(&cam_name) {
                            last_active_zone = c.active_zone.clone();
                        }
                    }

                    let mut processed_frame = frame.clone();

                    if !last_active_zone.is_empty() {
                        let mut pts = opencv::core::Vector::<opencv::core::Point>::new();
                        for p in &last_active_zone {
                            pts.push(*p);
                        }
                        let mut contours = opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();
                        contours.push(pts);

                        if let Ok(size) = frame.size() {
                            if let Ok(mut mask) = Mat::new_rows_cols_with_default(
                                size.height,
                                size.width,
                                opencv::core::CV_8UC1,
                                opencv::core::Scalar::all(0.0),
                            ) {
                                let _ = opencv::imgproc::draw_contours(
                                    &mut mask,
                                    &contours,
                                    -1,
                                    opencv::core::Scalar::all(255.0),
                                    -1,
                                    opencv::imgproc::LINE_AA,
                                    &opencv::core::no_array(),
                                    2147483647,
                                    opencv::core::Point::new(0, 0),
                                );
                                let mut dst = Mat::default();
                                if opencv::core::bitwise_and(&frame, &frame, &mut dst, &mask).is_ok() {
                                    processed_frame = dst;
                                }
                            }
                        }
                    }

                    let mut buf = Vector::<u8>::new();
                    let params = Vector::<i32>::new();
                    
                    // Encode and send cropped/processed frame
                    if let Ok(true) = imencode(".jpg", &processed_frame, &mut buf, &params) {
                        let bytes: Vec<u8> = buf.to_vec();
                        let _ = tx.send(bytes); // Ignore error if no receivers
                    }

                    // Encode and send raw uncropped frame
                    let mut raw_buf = Vector::<u8>::new();
                    if last_active_zone.is_empty() {
                        // If no active zone, raw is same as processed, just send the same buf
                        let _ = raw_tx.send(buf.to_vec());
                    } else {
                        if let Ok(true) = imencode(".jpg", &frame, &mut raw_buf, &params) {
                            let _ = raw_tx.send(raw_buf.to_vec());
                        }
                    }
                }
                _ => {
                    // Suppress spam. In a robust setup we'd try to reopen the VideoCapture if it's dead.
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
    });
}

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
#[tokio::main]
async fn main() {
    std::env::set_var("OPENCV_FFMPEG_LOGLEVEL", "-8");
    std::env::set_var("OPENCV_VIDEOIO_DEBUG", "0");
    
    tracing_subscriber::fmt::init();

    let mut cc = HexCaptureConfiguration::load_from_file("observerConfiguration.json")
        .unwrap_or_else(|e| {
            eprintln!("Failed to load configuration: {:?}", e);
            HexCaptureConfiguration {
                cameras: std::collections::HashMap::new(),
                rsc: std::collections::HashMap::new(),
                hex: Some(harmony_core::observer::HexGridConfiguration::default()),
                show_grid: true,
                show_objects: true,
                grid_polys_cache: std::collections::HashMap::new(),
                calibration_plan: serde_json::json!({}),
            }
        });
        
    let machine = Arc::new(Mutex::new(HarmonyMachine::new(cc)));
    let (frame_tx, _) = broadcast::channel(16);

    {
        let mut m = machine.lock().await;
        for (cam_name, cam) in m.cc.cameras.iter_mut() {
            let tx = cam.frame_tx.take();
            let raw_tx = cam.raw_frame_tx.take();
            if let (Some(tx), Some(raw_tx)) = (tx, raw_tx) {
                spawn_camera_stream(cam_name.clone(), cam.cam_path.clone(), machine.clone(), tx, raw_tx);
            }
        }
    }

    let state = Arc::new(AppState {
        machine: machine.clone(),
        frame_tx: frame_tx.clone(),
        sessions: Mutex::new(std::collections::HashMap::new()),
    });

    let tx = frame_tx.clone();
    tokio::spawn(async move {
        loop {
            {
                let mut m = machine.lock().await;
                if let Err(e) = m.cycle() {
                    tracing::error!("Machine cycle error: {}", e);
                }
                for (cam_name, cam) in m.cc.cameras.iter() {
                    if let Some(frame) = &cam.reference_frame {
                        let mut buf = opencv::core::Vector::<u8>::new();
                        let params = opencv::core::Vector::<i32>::new();
                        if let Ok(success) = opencv::imgcodecs::imencode(".jpg", frame, &mut buf, &params) {
                            if success {
                                let bytes = bytes::Bytes::copy_from_slice(buf.as_slice());
                                let _ = tx.send((cam_name.clone(), bytes));
                            }
                        }
                    }
                }
            }
            // In python, the calibrator pool time was 1.0s and observer was 0.01s.
            // We can just sleep for a short duration.
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
    });

    let static_dir = if std::path::Path::new("harmony-web/static").exists() {
        "harmony-web/static"
    } else {
        "static"
    };

    let admin_app = Router::new()
        .route("/", get(|| async { axum::response::Redirect::to("/harmony") }))
        .route("/harmony", get(harmony))
        .route("/harmony/canvas_data/:view_id", get(canvas_data))
        .route("/harmony/grid_polys/:cam_name", get(grid_polys))
        .route("/harmony/select_pixel", post(select_pixel))
        .route("/harmony/clear_selection", post(clear_selection))
        .route("/harmony/camWithChanges/:cam_name/:view_id", get(cam_with_changes_feed))
        .route("/video_feed/:cam_name", get(raw_video_feed))
        .route("/configurator/camera/:cam_name", get(raw_video_feed))
        .route("/harmony/objects", get(get_objects).post(define_object))
        .route("/harmony/objects/:oid/snapshot/:cam_name", get(object_snapshot_feed))
        .route("/harmony/objects/:oid", axum::routing::delete(delete_object))
        .route("/harmony/objects/:oid/move", post(move_object))
        .route("/harmony/objects/:oid/rotate", post(rotate_object))
        .route("/harmony/objects/:oid/type", post(update_object_type))
        .route("/harmony/save", post(save_game))
        .route("/harmony/load", post(load_game))
        .route("/harmony/reset", get(reset_game))
        .route("/harmony/set_overlays", post(set_overlays))
        .route("/harmony/control", get(session_list))
        .route("/harmony/control/:view_id", get(session_control_panel))
        .route("/harmony/control/:view_id/update", post(update_session_config))
        .route("/harmony/publish_selection", post(publish_selection))
        .route("/harmony/clear_published_selection", post(clear_published_selection))
        .route("/harmony/clear_all_published_selections", post(clear_all_published_selections))
        .route("/observer", get(observer))
        .route("/configurator", get(configurator).post(configurator_save))
        .route("/configurator/delete_cam/:name", post(configurator_delete_cam))
        .route("/configurator/activezone/:name", post(configurator_activezone))
        .route("/configurator/manual_calibration", post(configurator_manual_calibration))
        .route("/configurator/grid_configuration", post(configurator_grid))
        .route("/configurator/new_camera", get(new_camera).post(new_camera_submit))
        .route("/configurator/calibrator", get(calibrator))
        .route("/calibrator/reset", get(calibrator_reset))
        .route("/calibrator/camWithChanges/:name", get(raw_video_feed))
        .route("/calibrator/get_mode_controller", get(calibrator_mode_controller))
        .route("/calibrator/commit_calibration", get(calibrator_commit))
        .route("/calibrator/objects", get(calibrator_objects))
        .route("/calibrator/observer_console", get(calibrator_console))
        .nest_service("/static", ServeDir::new(static_dir.clone()))
        .fallback_service(ServeDir::new(static_dir.clone()))
        .route("/configurator/", get(configurator).post(configurator_save))
        .route("/configurator/clear_calibration/:name", post(configurator_clear_calibration))
        .with_state(state.clone());

    let user_app = Router::new()
        .route("/", get(harmony_user))
        .route("/harmony_user", get(harmony_user))
        .route("/harmony/canvas_data/:view_id", get(canvas_data))
        .route("/harmony/grid_polys/:cam_name", get(grid_polys))
        .route("/harmony/select_pixel", post(select_pixel))
        .route("/harmony/clear_selection", post(clear_selection))
        .route("/harmony/objects", get(get_objects))
        .route("/harmony/objects/:oid/snapshot/:cam_name", get(object_snapshot_feed))
        .route("/harmony/objects/:oid/move", post(move_object))
        .route("/harmony/objects/:oid/rotate", post(rotate_object))
        .route("/harmony/set_overlays", post(set_overlays))
        .route("/harmony/publish_selection", post(publish_selection))
        .route("/harmony/clear_published_selection", post(clear_published_selection))
        .route("/harmony/camWithChanges/:cam_name/:view_id", get(cam_with_changes_feed)) // Alias to clean video feed for client overlays
        .nest_service("/static", ServeDir::new(static_dir.clone()))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state);

    let admin_addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let user_addr = SocketAddr::from(([0, 0, 0, 0], 8081));
    
    tracing::info!("Admin server listening on {}", admin_addr);
    tracing::info!("User server listening on {}", user_addr);

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(&admin_addr).await.unwrap();
        axum::serve(listener, admin_app).await.unwrap();
    });

    let listener = tokio::net::TcpListener::bind(&user_addr).await.unwrap();
    axum::serve(listener, user_app).await.unwrap();
}



#[derive(Template)]
#[template(path = "observer.html")]
struct ObserverTemplate {
    default_camera: String,
    cameras: Vec<String>,
}

async fn observer(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /observer");
    let machine = state.machine.lock().await;
    let mut cameras: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    cameras.sort();
    let default_camera = cameras.first().cloned().unwrap_or_default();

    let template = ObserverTemplate {
        default_camera,
        cameras,
    };
    match askama::Template::render(&template) {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

struct CameraConfig {
    name: String,
    active_zone: String,
}

#[derive(Template)]
#[template(path = "configurator.html")]
struct ConfiguratorTemplate {
    cameras: Vec<CameraConfig>,
    camera_names_json: String,
    calibration_pts_json: String,
    grid_polys_json: String,
    show_grid: bool,
    show_objects: bool,
    hex: harmony_core::observer::HexGridConfiguration,
}

async fn configurator(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /configurator");
    let machine = state.machine.lock().await;
    let mut cameras: Vec<CameraConfig> = machine.cc.cameras.iter().map(|(k, v)| {
        let pts: Vec<String> = v.active_zone.iter().map(|p| format!("[{},{}]", p.x, p.y)).collect();
        CameraConfig {
            name: k.clone(),
            active_zone: format!("[{}]", pts.join(", ")),
        }
    }).collect();
    cameras.sort_by(|a, b| a.name.cmp(&b.name));

    let camera_names: Vec<String> = cameras.iter().map(|c| c.name.clone()).collect();
    let camera_names_json = serde_json::to_string(&camera_names).unwrap_or_else(|_| "[]".to_string());
    
    let calibration_pts_json = serde_json::to_string(&machine.cc.calibration_plan).unwrap_or_else(|_| "{}".to_string());

    let grid_polys_json = serde_json::to_string(&machine.cc.grid_polys_cache).unwrap_or_else(|_| "{}".to_string());

    let hex = machine.cc.hex.clone().unwrap_or_default();
    let show_grid = machine.cc.show_grid;
    let show_objects = machine.cc.show_objects;

    let template = ConfiguratorTemplate { 
        cameras, 
        camera_names_json, 
        calibration_pts_json,
        grid_polys_json,
        show_grid, 
        show_objects, 
        hex 
    };
    match askama::Template::render(&template) {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

#[derive(Template)]
#[template(path = "calibrator.html")]
struct CalibratorTemplate {
    default_camera: String,
    cameras: Vec<String>,
}

async fn calibrator(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator");
    let machine = state.machine.lock().await;
    let mut cameras: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    cameras.sort();
    let default_camera = cameras.first().cloned().unwrap_or_default();

    let template = CalibratorTemplate {
        default_camera,
        cameras,
    };
    match askama::Template::render(&template) {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

#[derive(Template)]
#[template(path = "new_camera.html")]
struct NewCameraTemplate {}

async fn new_camera(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /configurator/new_camera");
    let template = NewCameraTemplate {};
    match askama::Template::render(&template) {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

use axum::extract::Form;

async fn configurator_save(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let machine = state.machine.lock().await;
    tracing::info!("Saving configurator state to observerConfiguration.json");
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration: {}", e);
    } else {
        tracing::info!("Successfully saved configurator state.");
    }
    axum::response::Redirect::to("/configurator")
}

async fn configurator_delete_cam(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>
) -> impl IntoResponse {
    tracing::info!("Route hit: /configurator/delete_cam/{}", name);
    let mut machine = state.machine.lock().await;
    machine.cc.cameras.remove(&name);
    machine.cc.rsc.remove(&name);
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration after deleting camera {}: {}", name, e);
    } else {
        tracing::info!("Successfully deleted camera: {}", name);
    }
    axum::response::Redirect::to("/configurator")
}

#[derive(serde::Deserialize)]
struct ActiveZoneForm {
    az: String,
}

async fn configurator_activezone(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Form(form): Form<ActiveZoneForm>
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    tracing::info!("Updating active zone for camera {}", name);
    if let Some(cam) = machine.cc.cameras.get_mut(&name) {
        if let Ok(pts) = serde_json::from_str::<Vec<[i32; 2]>>(&form.az) {
            cam.active_zone = pts.into_iter().map(|p| opencv::core::Point::new(p[0], p[1])).collect();
            tracing::info!("Updated active zone with {} points", cam.active_zone.len());
        } else {
            tracing::error!("Failed to parse active zone JSON: {}", form.az);
        }
    } else {
        tracing::error!("Camera {} not found for active zone update", name);
    }
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration: {}", e);
    } else {
        tracing::info!("Saved active zone to configuration.");
    }
    axum::response::Redirect::to("/configurator")
}


#[derive(serde::Deserialize)]
struct NewCameraForm {
    #[serde(rename = "camName")]
    cam_name: String,
    #[serde(rename = "camRot", default)]
    cam_rot: Option<String>,
    #[serde(rename = "camAddr")]
    cam_addr: String,
    #[serde(rename = "camAuth", default)]
    cam_auth: Option<String>,
}

async fn new_camera_submit(
    State(state): State<Arc<AppState>>,
    Form(form): Form<NewCameraForm>
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    tracing::info!("Adding new camera: {} at {}", form.cam_name, form.cam_addr);
    
    let (tx, rx) = tokio::sync::watch::channel(vec![]);
    let (raw_tx, raw_rx) = tokio::sync::watch::channel(vec![]);
    
    let mut cam_path = form.cam_addr.clone();
    if !cam_path.starts_with("http://") && !cam_path.starts_with("https://") && !cam_path.starts_with("rtsp://") && !cam_path.starts_with("/") {
        cam_path = format!("rtsp://{}", cam_path);
    }
    
    spawn_camera_stream(form.cam_name.clone(), cam_path.clone(), state.machine.clone(), tx, raw_tx);

    let new_cam = harmony_core::observer::Camera {
        name: form.cam_name.clone(),
        cam_path: form.cam_addr.clone(),

        active_zone: vec![],
        rotate: false,
        auth: vec![],
        image_buffer: vec![],
        reference_frame: None,
        frame_rx: Some(rx),
        frame_tx: None,
        raw_frame_rx: Some(raw_rx),
        raw_frame_tx: None,
    };
    machine.cc.cameras.insert(form.cam_name.clone(), new_cam);
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration after adding camera: {}", e);
    } else {
        tracing::info!("Successfully added and saved camera: {}", form.cam_name);
    }
    
    let mut headers = header::HeaderMap::new();
    headers.insert("HX-Redirect", header::HeaderValue::from_static("/configurator"));
    (headers, axum::http::StatusCode::OK).into_response()
}


async fn calibrator_reset() -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/reset");
    Html("Reset calibrator")
}

use tokio_stream::wrappers::BroadcastStream;
use futures_util::stream::StreamExt;
use axum::http::header;


async fn calibrator_cam_with_changes(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>
) -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/camera/{} with changes", name);
    let rx = state.frame_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |res| {
        let name_clone = name.clone();
        async move {
            match res {
                Ok((cam_name, bytes)) if cam_name == name_clone => {
                    let header = format!("--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                    let footer = "\r\n";
                    let mut resp_bytes = bytes::BytesMut::new();
                    resp_bytes.extend_from_slice(header.as_bytes());
                    resp_bytes.extend_from_slice(&bytes);
                    resp_bytes.extend_from_slice(footer.as_bytes());
                    Some(Ok::<_, axum::Error>(resp_bytes.freeze()))
                }
                _ => None,
            }
        }
    });

    let body = axum::body::Body::from_stream(stream);

    Response::builder()
        .header(header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame")
        .body(body)
        .unwrap()
}

async fn calibrator_mode_controller() -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/mode_controller");
    Html("Mode Controller UI")
}

async fn calibrator_commit(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/commit");
    let machine = state.machine.lock().await;
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration: {}", e);
    } else {
        tracing::info!("Calibration committed to file.");
    }
    Html("Calibration committed")
}

async fn calibrator_objects() -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/objects");
    Html("Calibration objects table")
}

async fn calibrator_console() -> impl IntoResponse {
    tracing::info!("Route hit: /calibrator/console");
    Html("Observer console feed")
}

#[derive(serde::Deserialize)]
pub struct HarmonyQuery {
    #[serde(rename = "viewId")]
    view_id: Option<String>,
}

#[derive(Template)]
#[template(path = "harmony.html")]
struct HarmonyTemplate {
    view_id: String,
    default_camera: String,
    camera_buttons: String,
    harmony_url: String,
    configurator_url: String,
    show_grid_checked: String,
    show_objects_checked: String,
}

#[derive(Template)]
#[template(path = "harmony_user.html")]
struct HarmonyUserTemplate {
    view_id: String,
    default_camera: String,
    camera_buttons: String,
    harmony_url: String,
    show_grid_checked: String,
    show_objects_checked: String,
}

#[derive(Template)]
#[template(path = "SessionList.html")]
struct SessionListTemplate {
    sessions: Vec<String>,
}

#[derive(Template)]
#[template(path = "ControlPanel.html")]
#[allow(non_snake_case)]
struct ControlPanelTemplate {
    viewId: String,
    config: SessionConfig,
    objects: Vec<harmony_core::machine::TrackedObject>,
}

const ADJECTIVES: &[&str] = &[
    "Cool", "Happy", "Fast", "Shiny", "Blue", "Red", "Green", "Bright", "Dark",
    "Loud", "Quiet", "Brave", "Calm", "Eager", "Fair", "Gentle", "Jolly", "Kind",
    "Lively", "Nice", "Proud", "Silly", "Witty", "Zealous"
];

const NOUNS: &[&str] = &[
    "Tiger", "Eagle", "Shark", "Bear", "Lion", "Wolf", "Fox", "Hawk", "Owl",
    "Frog", "Toad", "Fish", "Crab", "Star", "Moon", "Sun", "Cloud", "Rain",
    "Snow", "Wind", "Storm", "River", "Lake", "Sea", "Ocean"
];

async fn build_harmony_context(
    state: &AppState,
    query: &HarmonyQuery,
    jar: &CookieJar,
) -> (String, String, String, String, String) {
    let mut view_id = query.view_id.clone().or_else(|| {
        jar.get("session_view_id").map(|c| c.value().to_string())
    });

    if let Some(vid) = &view_id {
        let mut sessions = state.sessions.lock().await;
        if !sessions.contains_key(vid) {
            sessions.insert(vid.clone(), SessionConfig::default());
        }
    } else {
        let mut sessions = state.sessions.lock().await;
        let mut vid = String::new();
        let mut counter = 0;
        loop {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() + counter;
            let adj_idx = (t % (ADJECTIVES.len() as u128)) as usize;
            let noun_idx = ((t / (ADJECTIVES.len() as u128)) % (NOUNS.len() as u128)) as usize;
            vid = format!("{}-{}", ADJECTIVES[adj_idx], NOUNS[noun_idx]);
            if !sessions.contains_key(&vid) {
                break;
            }
            counter += 1;
        }
        view_id = Some(vid.clone());
        sessions.insert(vid.clone(), SessionConfig::default());
    }
    let view_id = view_id.unwrap();

    let (show_grid_checked, show_objects_checked) = {
        let sessions = state.sessions.lock().await;
        if let Some(config) = sessions.get(&view_id) {
            (
                if config.show_grid { "checked".to_string() } else { "".to_string() },
                if config.show_objects { "checked".to_string() } else { "".to_string() }
            )
        } else {
            ("".to_string(), "checked".to_string())
        }
    };

    let machine = state.machine.lock().await;
    let mut cams: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    cams.sort();

    let mut camera_buttons = String::new();
    camera_buttons.push_str(&format!(
        r#"<input type="button" class="btn btn-info" value="Virtual Map" onclick="gameWorldClick('VirtualMap')"> "#
    ));
    for cam in &cams {
        camera_buttons.push_str(&format!(
            r#"<input type="button" class="btn btn-info" value="Camera {}" onclick="gameWorldClick('{}')"> "#,
            cam, cam
        ));
    }

    let default_camera = cams.first().cloned().unwrap_or_else(|| "None".to_string());

    (view_id, default_camera, camera_buttons, show_grid_checked, show_objects_checked)
}

async fn harmony(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HarmonyQuery>,
    jar: CookieJar,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony");
    let (view_id, default_camera, camera_buttons, show_grid_checked, show_objects_checked) =
        build_harmony_context(&state, &query, &jar).await;

    let template = HarmonyTemplate {
        view_id: view_id.clone(),
        default_camera,
        camera_buttons,
        harmony_url: "/harmony/".to_string(),
        configurator_url: "/configurator".to_string(),
        show_grid_checked,
        show_objects_checked,
    };

    let updated_jar = jar.add(Cookie::new("session_view_id", view_id.clone()));

    match askama::Template::render(&template) {
        Ok(html) => (updated_jar, Html(html)).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

async fn harmony_user(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HarmonyQuery>,
    jar: CookieJar,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony_user");
    let (view_id, default_camera, camera_buttons, show_grid_checked, show_objects_checked) =
        build_harmony_context(&state, &query, &jar).await;

    let template = HarmonyUserTemplate {
        view_id: view_id.clone(),
        default_camera,
        camera_buttons,
        harmony_url: "/harmony/".to_string(),
        show_grid_checked,
        show_objects_checked,
    };

    let updated_jar = jar.add(Cookie::new("session_view_id", view_id.clone()));

    match askama::Template::render(&template) {
        Ok(html) => (updated_jar, Html(html)).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

#[derive(serde::Serialize)]
struct CanvasData {
    grid_cache_key: String,
    objects: std::collections::HashMap<String, serde_json::Value>,
    cameras: Vec<String>,
    selectable: Vec<String>,
    moveable: Vec<String>,
    terrain: Vec<String>,
    targetable: Vec<String>,
    enemies: Vec<String>,
    allies: Vec<String>,
    selection: serde_json::Value,
    constituent_axials: std::collections::HashMap<String, Vec<(i32, i32)>>,
    published_selections: Vec<Vec<(i32, i32)>>,
}

async fn canvas_data(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/canvas_data/{}", view_id);
    let machine = state.machine.lock().await;
    let sessions = state.sessions.lock().await;
    let session = sessions.get(&view_id).cloned().unwrap_or_default();

    let mut comp_cams: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    comp_cams.sort();
    comp_cams.push("VirtualMap".to_string());
    while comp_cams.len() < 4 {
        comp_cams.push("No Camera".to_string());
    }
    comp_cams.truncate(4);

    let hex = machine.cc.hex.clone().unwrap_or_default();
    
    let mut selection_json = serde_json::json!({
        "firstCell": null,
        "additionalCells": []
    });
    
    let mut gen_cell_map = |q: i32, r: i32| -> serde_json::Value {
        let center = hex.axial_to_pixel(q as f64, r as f64);
        let mut corners = Vec::new();
        for k in 0..6 {
            let ang = (60.0 * k as f64 - 30.0).to_radians();
            let cx = hex.size * ang.cos();
            let cy = hex.size * ang.sin();
            let angle_rad = hex.rotation_deg.to_radians();
            let rotated_cx = cx * angle_rad.cos() - cy * angle_rad.sin();
            let rotated_cy = cx * angle_rad.sin() + cy * angle_rad.cos();
            corners.push(vec![center.0 + rotated_cx, center.1 + rotated_cy]);
        }
        let mut map = serde_json::Map::new();
        map.insert("_q".to_string(), serde_json::json!(q));
        map.insert("_r".to_string(), serde_json::json!(r));
        map.insert("VirtualMap".to_string(), serde_json::json!(corners));
        for (cam_name, polys_val) in &machine.cc.grid_polys_cache {
            if let Some(polys) = polys_val.as_array() {
                if let Some(poly) = polys.iter().find(|p| p["q"].as_i64() == Some(q as i64) && p["r"].as_i64() == Some(r as i64)) {
                    if let Some(pts_arr) = poly["poly"].as_array() {
                        let pts: Vec<Vec<f64>> = pts_arr.iter().map(|pt| {
                            let pt_arr = pt.as_array().unwrap();
                            vec![pt_arr[0].as_f64().unwrap(), pt_arr[1].as_f64().unwrap()]
                        }).collect();
                        map.insert(cam_name.clone(), serde_json::json!(pts));
                    }
                }
            }
        }
        serde_json::Value::Object(map)
    };

    if let Some(cell) = session.selection.first_cell {
        selection_json["firstCell"] = gen_cell_map(cell.0, cell.1);
    }
    let mut add_cells = Vec::new();
    for cell in &session.selection.additional_cells {
        add_cells.push(gen_cell_map(cell.0, cell.1));
    }
    selection_json["additionalCells"] = serde_json::Value::Array(add_cells);

    let mut objects_json = std::collections::HashMap::new();
    let mut constituent_axials = std::collections::HashMap::new();
    let hex = machine.cc.hex.clone().unwrap_or_default();
    
    for (oid, obj) in &machine.memory {
        let mut vm_polys = Vec::new();
        for (q, r) in &obj.constituent_axials {
            let center = hex.axial_to_pixel(*q as f64, *r as f64);
            let mut corners = Vec::new();
            for k in 0..6 {
                let ang = (60.0 * k as f64 - 30.0).to_radians();
                let cx = hex.size * ang.cos();
                let cy = hex.size * ang.sin();
                let angle_rad = hex.rotation_deg.to_radians();
                let rotated_cx = cx * angle_rad.cos() - cy * angle_rad.sin();
                let rotated_cy = cx * angle_rad.sin() + cy * angle_rad.cos();
                corners.push(vec![center.0 + rotated_cx, center.1 + rotated_cy]);
            }
            vm_polys.push(corners);
        }
        let mut obj_map = serde_json::Map::new();
        obj_map.insert("VirtualMap".to_string(), serde_json::json!(vm_polys));

        for (cam_name, polys_val) in &machine.cc.grid_polys_cache {
            let mut cam_pts = Vec::new();
            if let Some(polys) = polys_val.as_array() {
                for (q, r) in &obj.constituent_axials {
                    if let Some(poly) = polys.iter().find(|p| p["q"].as_i64() == Some(*q as i64) && p["r"].as_i64() == Some(*r as i64)) {
                        if let Some(pts_arr) = poly["poly"].as_array() {
                            let pts: Vec<Vec<f64>> = pts_arr.iter().map(|pt| {
                                let pt_arr = pt.as_array().unwrap();
                                vec![pt_arr[0].as_f64().unwrap(), pt_arr[1].as_f64().unwrap()]
                            }).collect();
                            cam_pts.push(pts);
                        }
                    }
                }
            }
            if !cam_pts.is_empty() {
                obj_map.insert(cam_name.clone(), serde_json::json!(cam_pts));
            }
        }

        objects_json.insert(oid.clone(), serde_json::Value::Object(obj_map));
        constituent_axials.insert(oid.clone(), obj.constituent_axials.clone());
    }

    let mut terrain = session.terrain.clone();
    let mut moveable = session.moveable.clone();
    let mut targetable = session.targetable.clone();
    let mut selectable = session.selectable.clone();
    
    for (oid, obj) in &machine.memory {
        if obj.object_type == "Terrain" && !terrain.contains(oid) {
            terrain.push(oid.clone());
        } else if obj.object_type == "Structure" && !targetable.contains(oid) {
            targetable.push(oid.clone());
            selectable.push(oid.clone());
        } else if obj.object_type == "Unit" && !moveable.contains(oid) {
            moveable.push(oid.clone());
            selectable.push(oid.clone());
        } else if obj.object_type == "Selectable" && !selectable.contains(oid) {
            selectable.push(oid.clone());
        }
    }

    let mut published_selections = Vec::new();
    for session in sessions.values() {
        if let Some(sel) = &session.published_selection {
            published_selections.push(sel.clone());
        }
    }

    let data = CanvasData {
        grid_cache_key: "default".to_string(), // Rust backend doesn't implement caching yet
        objects: objects_json,
        cameras: comp_cams,
        selectable,
        moveable,
        terrain,
        targetable,
        enemies: session.enemies.clone(),
        allies: session.allies.clone(),
        selection: selection_json,
        constituent_axials,
        published_selections,
    };

    axum::response::Json(data)
}

async fn grid_polys(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(cam_name): axum::extract::Path<String>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/grid_polys/{}", cam_name);
    let machine = state.machine.lock().await;
    
    if cam_name == "VirtualMap" {
        if let Some(hex) = &machine.cc.hex {
            let mut v_polys = Vec::new();
            let radius = 30;
            for q in -radius..=radius {
                for r in -radius..=radius {
                    if q + r >= -radius && q + r <= radius {
                        let center = hex.axial_to_pixel(q as f64, r as f64);
                        let mut corners = Vec::new();
                        for k in 0..6 {
                            let ang = (60.0 * k as f64 - 30.0).to_radians();
                            let cx = hex.size * ang.cos();
                            let cy = hex.size * ang.sin();
                            let angle_rad = hex.rotation_deg.to_radians();
                            let rotated_cx = cx * angle_rad.cos() - cy * angle_rad.sin();
                            let rotated_cy = cx * angle_rad.sin() + cy * angle_rad.cos();
                            corners.push(vec![center.0 + rotated_cx, center.1 + rotated_cy]);
                        }
                        v_polys.push(serde_json::json!({
                            "q": q,
                            "r": r,
                            "poly": corners
                        }));
                    }
                }
            }
            return axum::response::Json(serde_json::Value::Array(v_polys));
        }
    }
    
    if let Some(polys) = machine.cc.grid_polys_cache.get(&cam_name) {
        axum::response::Json(polys.clone())
    } else {
        axum::response::Json(serde_json::json!([]))
    }
}

#[derive(serde::Deserialize)]
struct SelectPixelPayload {
    viewId: String,
    selectedPixel: String,
    selectedCamera: String,
    appendPixel: String,
    #[serde(rename = "isAdmin")]
    is_admin: Option<String>,
}

async fn select_pixel(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(payload): axum::extract::Form<SelectPixelPayload>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/select_pixel by view_id {}", payload.viewId);
    let pixel: Result<(f64, f64), _> = serde_json::from_str(&payload.selectedPixel);
    let (x, y) = pixel.unwrap_or((0.0, 0.0));
    
    // Scale undo for non-virtual map (default scale in frontend)
    // Actually the python version did:
    // raw_x = (x / scale_x) + offset_x
    // scale_x = 1920/w if w>0
    
    let mut axial_coord = (0, 0);
    
    {
        let machine = state.machine.lock().await;
        let cam_name = payload.selectedCamera.replace("RTSPCamera", "");
        
        if cam_name != "VirtualMap" {
            if let Some(rsc) = machine.cc.rsc.get(&cam_name) {
                if let Ok(real_pt) = rsc.convert_camera_to_real_space((x, y)) {
                    if let Some(hex_cfg) = &machine.cc.hex {
                        axial_coord = hex_cfg.pixel_to_axial(real_pt.0, real_pt.1);
                    }
                }
            }
        }
        
        let mut objects_at_cell = Vec::new();
        for (oid, obj) in &machine.memory {
            if obj.constituent_axials.contains(&axial_coord) {
                objects_at_cell.push(oid.clone());
            }
        }
        objects_at_cell.sort();

        let mut sessions = state.sessions.lock().await;
        let session = sessions.entry(payload.viewId.clone()).or_insert_with(SessionConfig::default);
        
        let append = payload.appendPixel.to_lowercase() == "true";
        if append && session.selection.first_cell.is_some() {
            if !session.selection.additional_cells.contains(&axial_coord) {
                session.selection.additional_cells.insert(0, axial_coord);
            }
        } else {
            session.selection.first_cell = Some(axial_coord);
            session.selection.additional_cells.clear();
        }
        if !append {
            if objects_at_cell.is_empty() {
                session.selected_oid = None;
            } else {
                let current_idx = session.selected_oid.as_ref()
                    .and_then(|oid| objects_at_cell.iter().position(|id| id == oid))
                    .unwrap_or(objects_at_cell.len() - 1);
                    
                let next_idx = (current_idx + 1) % objects_at_cell.len();
                session.selected_oid = Some(objects_at_cell[next_idx].clone());
            }
        }
        
        let is_admin = payload.is_admin.as_deref() == Some("true");
        let html = render_interactor(&session, &machine, &payload.viewId, is_admin);
        axum::response::Html(html)
    }
}

pub fn render_interactor(session: &SessionConfig, machine: &harmony_core::machine::HarmonyMachine, view_id: &str, is_admin: bool) -> String {
    let first_cell_q = session.selection.first_cell.map(|c| c.0).unwrap_or(0);
    let first_cell_r = session.selection.first_cell.map(|c| c.1).unwrap_or(0);
    
    let mut interactor_html = String::new();
    if session.selection.first_cell.is_some() {
        interactor_html.push_str(&format!("<div id='interactor'><h5>Selected Hexes:</h5><ul class='list-unstyled mb-2'><li>({}, {}) - <span class='text-muted'>Primary</span></li>", first_cell_q, first_cell_r));
    } else {
        interactor_html.push_str("<div id='interactor'><h5>Selected Hexes:</h5><ul class='list-unstyled mb-2'>");
    }
        
    if let Some(first_cell) = session.selection.first_cell {
        for (idx, cell) in session.selection.additional_cells.iter().enumerate() {
            let dist = (first_cell.0 - cell.0).abs()
                .max((first_cell.0 + first_cell.1 - cell.0 - cell.1).abs())
                .max((first_cell.1 - cell.1).abs());
            interactor_html.push_str(&format!("<li>({}, {}) - <span class='text-muted'>Additional {}</span> <span style='color: #6f42c1; font-weight: 500;'>(Distance: {})</span></li>", cell.0, cell.1, idx + 1, dist));
        }
    }
    
    interactor_html.push_str("</ul>");
    interactor_html.push_str(&format!("<p class='mb-1'>Object: {}</p>", 
        session.selected_oid.as_deref().unwrap_or("None")));
        
    if let Some(_first_cell) = session.selection.first_cell {
        if let Some(oid) = &session.selected_oid {
            let mut is_multi_cell = false;
            if let Some(obj) = machine.memory.get(oid) {
                is_multi_cell = obj.constituent_axials.len() > 1;
            }

            if is_multi_cell && (session.moveable.contains(oid) || is_admin) {
                let admin_input = if is_admin { "<input type='hidden' name='isAdmin' value='true'>" } else { "" };
                interactor_html.push_str(&format!(
                    "<div class='d-flex justify-content-between mt-2'>
                        <form hx-post='/harmony/objects/{}/rotate' hx-target='#interactor' class='flex-fill me-1'>
                            <input type='hidden' name='direction' value='left'>
                            <input type='hidden' name='viewId' value='{}'>
                            {}
                            <button type='submit' class='btn btn-outline-info btn-sm w-100'>Rotate ↺</button>
                        </form>
                        <form hx-post='/harmony/objects/{}/rotate' hx-target='#interactor' class='flex-fill ms-1'>
                            <input type='hidden' name='direction' value='right'>
                            <input type='hidden' name='viewId' value='{}'>
                            {}
                            <button type='submit' class='btn btn-outline-info btn-sm w-100'>Rotate ↻</button>
                        </form>
                    </div>",
                    oid, view_id, admin_input, oid, view_id, admin_input
                ));
            }

            if let Some(second_cell) = session.selection.additional_cells.first() {
                if session.moveable.contains(oid) || is_admin {
                    let admin_input = if is_admin { "<input type='hidden' name='isAdmin' value='true'>" } else { "" };
                    interactor_html.push_str(&format!(
                        "<form hx-post='/harmony/objects/{}/move' hx-target='#interactor' class='mt-2'>
                            <input type='hidden' name='q' value='{}'>
                            <input type='hidden' name='r' value='{}'>
                            <input type='hidden' name='viewId' value='{}'>
                            {}
                            <button type='submit' class='btn btn-warning btn-sm w-100'>Move to ({}, {})</button>
                        </form>",
                        oid, second_cell.0, second_cell.1, view_id, admin_input, second_cell.0, second_cell.1
                    ));
                }
            }
        }
    }
    
    if session.can_publish_selection {
        interactor_html.push_str(&format!(
            "<hr class='my-2'>
            <div class='d-flex justify-content-between'>
                <form hx-post='/harmony/publish_selection' hx-target='#publishedFeedbackContainer' hx-swap='innerHTML' class='flex-fill me-1'>
                    <input type='hidden' name='viewId' value='{}'>
                    <button type='submit' class='btn btn-primary btn-sm w-100'>Publish Selection</button>
                </form>
                <form hx-post='/harmony/clear_published_selection' hx-target='#publishedFeedbackContainer' hx-swap='innerHTML' class='flex-fill ms-1'>
                    <input type='hidden' name='viewId' value='{}'>
                    <button type='submit' class='btn btn-secondary btn-sm w-100'>Clear Broadcast</button>
                </form>
            </div>
            <div id='publishedFeedbackContainer'></div>",
            view_id, view_id
        ));
    }
    
    interactor_html.push_str("</div>");
    interactor_html
}

// Ensure select_pixel is not broken by the extraction. I will update select_pixel to use the helper.
// In the original file, the code I am replacing starts at line 1170 which is inside select_pixel.
// Wait, I am replacing lines 1170 to 1250! So I must also finish `select_pixel` inside this block.

#[derive(serde::Deserialize)]
struct ClearSelectionPayload {
    viewId: String,
}

async fn clear_selection(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(payload): axum::extract::Form<ClearSelectionPayload>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/clear_selection by view_id {}", payload.viewId);
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        session.selection.first_cell = None;
        session.selection.additional_cells.clear();
        session.selected_oid = None;
    }
    axum::response::Html("".to_string())
}


#[derive(Template)]
#[template(path = "objects.html")]
struct ObjectsTemplate {
    objects: Vec<harmony_core::machine::TrackedObject>,
    selection_json: String,
    cameras: Vec<String>,
    is_user_ui: bool,
}

#[derive(serde::Deserialize)]
struct GetObjectsQuery {
    viewId: Option<String>,
    isUserUI: Option<String>,
}

async fn get_objects(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<GetObjectsQuery>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/objects");
    let (mut objects, mut cameras): (Vec<harmony_core::machine::TrackedObject>, Vec<String>) = {
        let machine = state.machine.lock().await;
        (
            machine.memory.values().cloned().collect(),
            machine.cc.cameras.keys().cloned().collect()
        )
    };
    objects.sort_by(|a, b| a.oid.cmp(&b.oid));
    cameras.sort();
    let mut selection_json = "[]".to_string();
    if let Some(vid) = query.viewId {
        let sessions = state.sessions.lock().await;
        if let Some(session) = sessions.get(&vid) {
            let mut cells = Vec::new();
            if let Some(c) = session.selection.first_cell {
                cells.push(vec![c.0, c.1]);
            }
            for c in &session.selection.additional_cells {
                cells.push(vec![c.0, c.1]);
            }
            selection_json = serde_json::to_string(&cells).unwrap_or_else(|_| "[]".to_string());
            
            for obj in objects.iter_mut() {
                let mut highest_tag = None;
                if session.selectable.contains(&obj.oid) { highest_tag = Some("Selectable"); }
                if session.terrain.contains(&obj.oid) { highest_tag = Some("Terrain"); }
                if session.targetable.contains(&obj.oid) { highest_tag = Some("Targetable"); }
                if session.enemies.contains(&obj.oid) { highest_tag = Some("Enemies"); }
                if session.allies.contains(&obj.oid) { highest_tag = Some("Allies"); }
                if session.moveable.contains(&obj.oid) { highest_tag = Some("Moveable"); }
                
                if let Some(tag) = highest_tag {
                    obj.object_type = tag.to_string();
                }
            }
        }
    }
    
    let template = ObjectsTemplate {
        objects,
        selection_json,
        cameras,
        is_user_ui: query.isUserUI.is_some(),
    };
    
    match askama::Template::render(&template) {
        Ok(html) => (
            [(axum::http::header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")],
            Html(html)
        ).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct DefineObjectForm {
    name: String,
    cells: String,
}

async fn define_object(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(form): axum::extract::Form<DefineObjectForm>,
) -> impl IntoResponse {
    let cells: Result<Vec<Vec<i32>>, _> = serde_json::from_str(&form.cells);
    let mut axials = Vec::new();
    if let Ok(cells_arr) = cells {
        for c in cells_arr {
            if c.len() >= 2 {
                axials.push((c[0], c[1]));
            }
        }
    }
    
    let mut machine = state.machine.lock().await;
    let oid = if form.name.is_empty() {
        format!("obj_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis())
    } else {
        form.name.clone()
    };
    
    let obj = machine.memory.entry(oid.clone()).or_insert_with(|| harmony_core::machine::TrackedObject {
        oid: oid.clone(),
        object_type: "Selectable".to_string(),
        constituent_axials: vec![],
    });
    
    obj.constituent_axials = axials;
    
    axum::response::Html(format!("Object defined: {}", oid))
}

async fn delete_object(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(oid): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    machine.memory.remove(&oid);
    axum::response::Html(format!("Object deleted: {}", oid))
}

#[derive(serde::Deserialize)]
struct MoveObjectForm {
    q: i32,
    r: i32,
    viewId: String,
    #[serde(rename = "isAdmin")]
    is_admin: Option<String>,
}

async fn move_object(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(oid): axum::extract::Path<String>,
    axum::extract::Form(payload): axum::extract::Form<MoveObjectForm>,
) -> impl IntoResponse {
    let mut result_html = format!("<div id='interactor'><h5>Failed to move {}</h5></div>", oid);
    
    let is_admin = payload.is_admin.as_deref() == Some("true");
    
    {
        let mut machine = state.machine.lock().await;
        if let Some(obj) = machine.memory.get_mut(&oid) {
            if let Some(first_axial) = obj.constituent_axials.first().cloned() {
                let dq = payload.q - first_axial.0;
                let dr = payload.r - first_axial.1;
                
                let mut new_axials = Vec::new();
                for (q, r) in &obj.constituent_axials {
                    new_axials.push((*q + dq, *r + dr));
                }
                obj.constituent_axials = new_axials;
            }
        }
    }
    
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        let machine = state.machine.lock().await;
        if let Some(obj) = machine.memory.get(&oid) {
            if !obj.constituent_axials.is_empty() {
                session.selection.first_cell = Some(obj.constituent_axials[0]);
                session.selection.additional_cells = obj.constituent_axials[1..].to_vec();
            }
        }
        result_html = render_interactor(session, &machine, &payload.viewId, is_admin);
    }
    
    axum::response::Html(result_html)
}

#[derive(serde::Deserialize)]
struct RotateObjectForm {
    direction: String,
    viewId: String,
    #[serde(rename = "isAdmin")]
    is_admin: Option<String>,
}

async fn rotate_object(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(oid): axum::extract::Path<String>,
    axum::extract::Form(payload): axum::extract::Form<RotateObjectForm>,
) -> impl IntoResponse {
    let mut result_html = format!("<div id='interactor'><h5>Failed to rotate {}</h5></div>", oid);
    
    {
        let mut machine = state.machine.lock().await;
        if let Some(obj) = machine.memory.get_mut(&oid) {
            if let Some(first_axial) = obj.constituent_axials.first().cloned() {
                let center_q = first_axial.0;
                let center_r = first_axial.1;
                
                let mut new_axials = Vec::new();
                for (q, r) in &obj.constituent_axials {
                    let dq = q - center_q;
                    let dr = r - center_r;
                    
                    let (new_dq, new_dr) = if payload.direction == "left" {
                        (dq + dr, -dq)
                    } else {
                        // right
                        (-dr, dq + dr)
                    };
                    
                    new_axials.push((center_q + new_dq, center_r + new_dr));
                }
                obj.constituent_axials = new_axials;
                result_html = format!("<div id='interactor'><h5>Rotated {} ({})</h5></div>", oid, payload.direction);
            }
        }
    }
    let is_admin = payload.is_admin.as_deref() == Some("true");

    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        let machine = state.machine.lock().await;
        if let Some(obj) = machine.memory.get(&oid) {
            if !obj.constituent_axials.is_empty() {
                session.selection.first_cell = Some(obj.constituent_axials[0]);
                session.selection.additional_cells = obj.constituent_axials[1..].to_vec();
            }
        }
        result_html = render_interactor(session, &machine, &payload.viewId, is_admin);
    }
    
    axum::response::Html(result_html)
}

#[derive(serde::Deserialize)]
struct UpdateTypeForm {
    #[serde(rename = "objectType")]
    object_type: String,
}

async fn update_object_type(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(oid): axum::extract::Path<String>,
    axum::extract::Form(form): axum::extract::Form<UpdateTypeForm>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    if let Some(obj) = machine.memory.get_mut(&oid) {
        obj.object_type = form.object_type;
        
        let mut sessions = state.sessions.lock().await;
        for session in sessions.values_mut() {
            session.terrain.retain(|x| x != &oid);
            session.targetable.retain(|x| x != &oid);
            session.moveable.retain(|x| x != &oid);
            session.selectable.retain(|x| x != &oid);
        }
        
        axum::response::Html(format!("Updated type for {}", oid))
    } else {
        axum::response::Html("Object not found".to_string())
    }
}

async fn save_game() -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/save_game");
    axum::response::Html("Game saved (stub)".to_string())
}

async fn load_game() -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/load_game");
    axum::response::Html("Game loaded (stub)".to_string())
}

async fn reset_game() -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/reset_game");
    axum::response::Html("Game reset (stub)".to_string())
}

#[derive(serde::Deserialize)]
struct SetOverlaysForm {
    show_grid: bool,
    show_objects: bool,
    #[serde(rename = "viewId")]
    view_id: String,
}

async fn set_overlays(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(form): axum::extract::Form<SetOverlaysForm>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/set_overlays for view {}", form.view_id);
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&form.view_id) {
        session.show_grid = form.show_grid;
        session.show_objects = form.show_objects;
    }
    axum::response::Html("".to_string())
}


use futures_util::stream::Stream;
use std::convert::Infallible;

async fn video_feed(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(cam_name): axum::extract::Path<String>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /video_feed/{} - starting stream", cam_name);
    let rx = {
        let machine = state.machine.lock().await;
        machine.cc.cameras.get(&cam_name).and_then(|c| c.frame_rx.clone())
    };

    if let Some(mut rx) = rx {
        let stream = async_stream::stream! {
            loop {
                if rx.changed().await.is_ok() {
                    let frame = rx.borrow().clone();
                    if !frame.is_empty() {
                        let header = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                        let mut chunk = Vec::new();
                        chunk.extend_from_slice(header.as_bytes());
                        chunk.extend_from_slice(&frame);
                        chunk.extend_from_slice(b"\r\n");
                        yield Ok::<_, Infallible>(axum::body::Bytes::from(chunk));
                    }
                } else {
                    break;
                }
            }
        };

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("multipart/x-mixed-replace; boundary=frame"),
        );
        (headers, axum::body::Body::from_stream(stream)).into_response()
    } else {
        (axum::http::StatusCode::NOT_FOUND, "Camera not found").into_response()
    }
}

#[derive(serde::Deserialize)]
struct GridConfigurationForm {
    size: f64,
}

async fn configurator_grid(
    State(state): State<Arc<AppState>>,
    Form(form): Form<GridConfigurationForm>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /configurator/grid_configuration with size: {}", form.size);
    let mut machine = state.machine.lock().await;
    if let Some(ref mut hex) = machine.cc.hex {
        hex.size = form.size;
    } else {
        let mut hex = harmony_core::observer::HexGridConfiguration::default();
        hex.size = form.size;
        machine.cc.hex = Some(hex);
    }
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration: {}", e);
    } else {
        tracing::info!("Saved grid configuration.");
    }
    let html = format!(r#"
        <form hx-post="/configurator/grid_configuration" hx-swap="outerHTML">
            <div class="d-flex align-items-center gap-2 mb-3">
                <input type="number" class="form-control bg-light text-dark fw-bold" name="size" min="10" max="60" value="{}"
                    style="width: 5em;">
                <label class="form-label mb-0 fw-bold" for="size">Size (px)</label>
            </div>
            <input type="submit" class="btn btn-primary" value="Submit Grid Configuration">
        </form>"#, form.size);
    axum::response::Html(html)
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ManualCalibrationPayload {
    pixel: Vec<Vec<f64>>,
    axial: Vec<Vec<i32>>,
    width: Option<f64>,
    height: Option<f64>,
}

#[derive(Clone, Debug)]
struct CalibPt {
    pixel: (f64, f64),
    axial: (i32, i32),
    real: (f64, f64),
}

async fn configurator_manual_calibration(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<std::collections::HashMap<String, ManualCalibrationPayload>>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    let mut success_count = 0;

    for (cam_name, data) in &payload {
        if data.pixel.len() < 3 || data.axial.len() < 3 {
            continue;
        }

        let w = data.width.unwrap_or(1920.0);
        let h = data.height.unwrap_or(1080.0);

        let mut unique_pts: std::collections::HashMap<(i32, i32), CalibPt> = std::collections::HashMap::new();
        let hex = machine.cc.hex.clone().unwrap_or_default();
        
        for i in 0..data.pixel.len() {
            let px = data.pixel[i][0] * w;
            let py = data.pixel[i][1] * h;
            let ax = data.axial[i][0];
            let ay = data.axial[i][1];
            
            let real = hex.axial_to_pixel(ax as f64, ay as f64);
            unique_pts.insert((ax, ay), CalibPt {
                pixel: (px, py),
                axial: (ax, ay),
                real,
            });
        }
        
        let mut calib_pt_columns: std::collections::HashMap<i32, Vec<CalibPt>> = std::collections::HashMap::new();
        for (_, pt) in &unique_pts {
            calib_pt_columns.entry(pt.axial.1).or_insert_with(Vec::new).push(pt.clone());
        }
        
        let mut col_values: Vec<i32> = calib_pt_columns.keys().copied().collect();
        col_values.sort();
        
        for r_val in &col_values {
            calib_pt_columns.get_mut(r_val).unwrap().sort_by_key(|pt| pt.axial.0);
        }
        
        let mut valid_blocks = Vec::new();
        if col_values.len() >= 2 {
            for col_idx in 0..(col_values.len() - 1) {
                let col = col_values[col_idx];
                let next_col = col_values[col_idx + 1];
                let col_pts = calib_pt_columns.get(&col).unwrap();
                let next_col_pts = calib_pt_columns.get(&next_col).unwrap();
                
                let max_idx = std::cmp::min(col_pts.len(), next_col_pts.len()).saturating_sub(1);
                for idx in 0..max_idx {
                    let calib_pt = &col_pts[idx];
                    let next_same_r = &col_pts[idx + 1];
                    let next_q_0 = &next_col_pts[idx];
                    let next_q_1 = &next_col_pts[idx + 1];
                    
                    valid_blocks.push(vec![
                        calib_pt.clone(),
                        next_same_r.clone(),
                        next_q_1.clone(),
                        next_q_0.clone(),
                    ]);
                }
            }
        }
        
        let mut converters = Vec::new();
        for block in valid_blocks {
            let cam_pts = vec![block[0].pixel, block[1].pixel, block[2].pixel, block[3].pixel];
            let real_pts = vec![block[0].real, block[1].real, block[2].real, block[3].real];
            if let Ok(converter) = harmony_core::observer::CameraRealSpaceConverter::new(&cam_pts, &real_pts) {
                converters.push(converter);
            }
        }
        
        if !converters.is_empty() {
            tracing::info!("Updating RealSpaceConverter for camera {} with {} blocks", cam_name, converters.len());
            machine.cc.rsc.insert(cam_name.clone(), harmony_core::observer::RealSpaceConverter::new(converters));
            machine.cc.precompute_grid_polys();
            success_count += 1;
        } else {
            tracing::warn!("No valid RealSpaceConverter blocks generated for camera {}", cam_name);
        }
    }

    if let Ok(plan_json) = serde_json::to_value(&payload) {
        machine.cc.calibration_plan = plan_json;
    }

    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration after manual calibration: {}", e);
    } else {
        tracing::info!("Saved manual calibration to configuration.");
    }

    if success_count > 0 {
        axum::response::Json(serde_json::json!({ "status": "success", "message": format!("Calibrated {} cameras", success_count) })).into_response()
    } else {
        (axum::http::StatusCode::BAD_REQUEST, axum::response::Json(serde_json::json!({ "status": "error", "message": "No valid calibrations found" }))).into_response()
    }
}

async fn configurator_clear_calibration(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    tracing::info!("Clearing calibration for camera {}", name);
    machine.cc.rsc.remove(&name);
    machine.cc.grid_polys_cache.remove(&name);
    
    // Also remove from calibration_plan if it exists
    if let Some(plan) = machine.cc.calibration_plan.as_object_mut() {
        plan.remove(&name);
    }
    
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration after clearing calibration: {}", e);
    }
    axum::response::Redirect::to("/configurator")
}

async fn cam_with_changes_feed(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((cam_name, _view_id)): axum::extract::Path<(String, String)>,
) -> impl IntoResponse {
    // We ignore view_id for now as overlays are handled client-side
    tracing::info!("Route hit: /harmony/camWithChanges/{}/{} - alias to video_feed", cam_name, _view_id);
    
    if cam_name.eq_ignore_ascii_case("VirtualMap") {
        let stream = async_stream::stream! {
            let img = opencv::core::Mat::new_rows_cols_with_default(1200, 1200, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0)).unwrap();
            let mut buf = opencv::core::Vector::<u8>::new();
            let params = opencv::core::Vector::<i32>::new();
            opencv::imgcodecs::imencode(".jpg", &img, &mut buf, &params).unwrap();
            let frame = buf.to_vec();
            
            loop {
                let header = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                let mut chunk = Vec::new();
                chunk.extend_from_slice(header.as_bytes());
                chunk.extend_from_slice(&frame);
                chunk.extend_from_slice(b"\r\n");
                yield Ok::<_, std::convert::Infallible>(chunk);
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            }
        };
        return axum::response::Response::builder()
            .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            .body(axum::body::Body::from_stream(stream))
            .unwrap();
    }
    let rx = {
        let machine = state.machine.lock().await;
        machine.cc.cameras.get(&cam_name).and_then(|c| c.frame_rx.clone())
    };

    if let Some(mut rx) = rx {
        let stream = async_stream::stream! {
            loop {
                if rx.changed().await.is_ok() {
                    let frame = rx.borrow().clone();
                    if !frame.is_empty() {
                        let header = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                        let mut chunk = Vec::new();
                        chunk.extend_from_slice(header.as_bytes());
                        chunk.extend_from_slice(&frame);
                        chunk.extend_from_slice(b"\r\n");
                        yield Ok::<_, std::convert::Infallible>(chunk);
                    }
                } else {
                    break;
                }
            }
        };

        axum::response::Response::builder()
            .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            .body(axum::body::Body::from_stream(stream))
            .unwrap()
    } else {
        (axum::http::StatusCode::NOT_FOUND, "Camera not found").into_response()
    }
}

async fn raw_video_feed(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(cam_name): axum::extract::Path<String>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /raw_video_feed/{} - starting stream", cam_name);
    let rx = {
        let machine = state.machine.lock().await;
        machine.cc.cameras.get(&cam_name).and_then(|c| c.raw_frame_rx.clone())
    };

    if let Some(mut rx) = rx {
        let stream = async_stream::stream! {
            loop {
                if rx.changed().await.is_ok() {
                    let frame = rx.borrow().clone();
                    if !frame.is_empty() {
                        let header = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                        let mut chunk = Vec::new();
                        chunk.extend_from_slice(header.as_bytes());
                        chunk.extend_from_slice(&frame);
                        chunk.extend_from_slice(b"\r\n");
                        yield Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(chunk));
                    }
                } else {
                    break;
                }
            }
        };

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("multipart/x-mixed-replace; boundary=frame"),
        );
        (headers, axum::body::Body::from_stream(stream)).into_response()
    } else {
        (axum::http::StatusCode::NOT_FOUND, "Camera not found").into_response()
    }
}

async fn session_list(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let sessions = state.sessions.lock().await;
    let sids: Vec<String> = sessions.keys().cloned().collect();
    let template = SessionListTemplate { sessions: sids };
    match template.render() {
        Ok(html) => axum::response::Html(html).into_response(),
        Err(err) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {}", err)).into_response(),
    }
}

async fn session_control_panel(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let config = {
        let sessions = state.sessions.lock().await;
        sessions.get(&view_id).cloned()
    };
    
    if let Some(config) = config {
        let mut objects: Vec<harmony_core::machine::TrackedObject> = {
            let machine = state.machine.lock().await;
            machine.memory.values().cloned().collect()
        };
        objects.sort_by(|a, b| a.oid.cmp(&b.oid));
        let template = ControlPanelTemplate {
            viewId: view_id.clone(),
            config: config.clone(),
            objects,
        };
        match template.render() {
            Ok(html) => axum::response::Html(html).into_response(),
            Err(err) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {}", err)).into_response(),
        }
    } else {
        (axum::http::StatusCode::NOT_FOUND, "Session not found").into_response()
    }
}

#[derive(serde::Deserialize)]
struct PublishSelectionForm {
    viewId: String,
}

async fn publish_selection(
    State(state): State<Arc<AppState>>,
    Form(payload): Form<PublishSelectionForm>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.lock().await;
    let mut published = None;
    if let Some(session) = sessions.get(&payload.viewId) {
        if session.can_publish_selection {
            if let Some(first) = session.selection.first_cell {
                let mut cells = vec![first];
                cells.extend(&session.selection.additional_cells);
                published = Some(cells);
            }
        }
    }
    
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        session.published_selection = published;
    }
    
    axum::response::Html("<div id='publishedFeedback' class='alert alert-success mt-2 p-1'>Selection published globally.</div>")
}

async fn clear_published_selection(
    State(state): State<Arc<AppState>>,
    Form(payload): Form<PublishSelectionForm>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        session.published_selection = None;
    }
    axum::response::Html("<div id='publishedFeedback' class='alert alert-secondary mt-2 p-1'>Broadcast cleared.</div>")
}

async fn clear_all_published_selections(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.lock().await;
    for session in sessions.values_mut() {
        session.published_selection = None;
    }
    axum::response::Html("<p>All broadcasts cleared.</p>")
}

async fn update_session_config(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
    Form(form): Form<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.lock().await;
    if let Some(config) = sessions.get_mut(&view_id) {
        config.selectable.clear();
        config.terrain.clear();
        config.targetable.clear();
        config.enemies.clear();
        config.allies.clear();
        config.moveable.clear();
        config.can_publish_selection = false;

        for (k, v) in form {
            if v == "on" {
                if k == "can_publish_selection" {
                    config.can_publish_selection = true;
                } else if let Some(oid) = k.strip_suffix("_selectable") {
                    config.selectable.push(oid.to_string());
                } else if let Some(oid) = k.strip_suffix("_terrain") {
                    config.terrain.push(oid.to_string());
                } else if let Some(oid) = k.strip_suffix("_targetable") {
                    config.targetable.push(oid.to_string());
                } else if let Some(oid) = k.strip_suffix("_enemies") {
                    config.enemies.push(oid.to_string());
                } else if let Some(oid) = k.strip_suffix("_allies") {
                    config.allies.push(oid.to_string());
                } else if let Some(oid) = k.strip_suffix("_moveable") {
                    config.moveable.push(oid.to_string());
                }
            }
        }
        axum::response::Redirect::to(&format!("/harmony/control/{}", view_id)).into_response()
    } else {
        (axum::http::StatusCode::NOT_FOUND, "Session not found").into_response()
    }
}
