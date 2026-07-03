use askama::Template;
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use std::net::SocketAddr;
use tower_http::services::ServeDir;
use harmony_core::machine::HarmonyMachine;
use harmony_core::observer::HexCaptureConfiguration;

struct AppState {
    machine: Arc<Mutex<HarmonyMachine>>,
    frame_tx: broadcast::Sender<(String, bytes::Bytes)>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cc = HexCaptureConfiguration {
        cameras: std::collections::HashMap::new(),
        hex: Some(harmony_core::observer::HexGridConfiguration::default()),
        show_grid: true,
        show_objects: true,
    };
    let machine = Arc::new(Mutex::new(HarmonyMachine::new(cc)));
    let (frame_tx, _) = broadcast::channel(16);

    let state = Arc::new(AppState {
        machine: machine.clone(),
        frame_tx: frame_tx.clone(),
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

    // Set up basic router with static file serving
    let app = Router::new()
        .route("/", get(index))
        .route("/observer", get(observer))
        .route("/configurator", get(configurator).post(configurator_save))
        .route("/configurator/delete_cam/:name", post(configurator_delete_cam))
        .route("/configurator/cam:name_activezone", post(configurator_activezone))
        .route("/configurator/new_camera", get(new_camera).post(new_camera_submit))
        .route("/configurator/calibrator", get(calibrator))
        .route("/calibrator/reset", get(calibrator_reset))
        .route("/calibrator/camWithChanges/:name", get(calibrator_cam_with_changes))
        .route("/calibrator/get_mode_controller", get(calibrator_mode_controller))
        .route("/calibrator/commit_calibration", get(calibrator_commit))
        .route("/calibrator/objects", get(calibrator_objects))
        .route("/calibrator/observer_console", get(calibrator_console))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    title: &'static str,
}

async fn index() -> impl IntoResponse {
    let template = IndexTemplate {
        title: "Harmony Web Server",
    };
    match template.render() {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

#[derive(Template)]
#[template(path = "observer.html")]
struct ObserverTemplate {
    default_camera: String,
    cameras: Vec<String>,
}

async fn observer(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let machine = state.machine.lock().await;
    let mut cameras: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    cameras.sort();
    let default_camera = cameras.first().cloned().unwrap_or_default();

    let template = ObserverTemplate {
        default_camera,
        cameras,
    };
    match template.render() {
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
}

async fn configurator(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let machine = state.machine.lock().await;
    let mut cameras: Vec<CameraConfig> = machine.cc.cameras.iter().map(|(k, v)| {
        let pts: Vec<String> = v.active_zone.iter().map(|p| format!("[{},{}]", p.x, p.y)).collect();
        CameraConfig {
            name: k.clone(),
            active_zone: format!("[{}]", pts.join(", ")),
        }
    }).collect();
    cameras.sort_by(|a, b| a.name.cmp(&b.name));

    let template = ConfiguratorTemplate { cameras };
    match template.render() {
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
    let machine = state.machine.lock().await;
    let mut cameras: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    cameras.sort();
    let default_camera = cameras.first().cloned().unwrap_or_default();

    let template = CalibratorTemplate {
        default_camera,
        cameras,
    };
    match template.render() {
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
    let template = NewCameraTemplate {};
    match template.render() {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", err),
        ).into_response(),
    }
}

use axum::extract::Form;

async fn configurator_save(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // In Python this calls cc.store()
    let _machine = state.machine.lock().await;
    axum::response::Redirect::to("/configurator")
}

async fn configurator_delete_cam(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    machine.cc.cameras.remove(&name);
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
    if let Some(cam) = machine.cc.cameras.get_mut(&name) {
        // Simple parsing of JSON like [[0,0],[100,0]]
        if let Ok(pts) = serde_json::from_str::<Vec<[i32; 2]>>(&form.az) {
            cam.active_zone = pts.into_iter().map(|p| opencv::core::Point::new(p[0], p[1])).collect();
        }
    }
    axum::response::Redirect::to("/configurator")
}

#[derive(serde::Deserialize)]
struct NewCameraForm {
    #[serde(rename = "camName")]
    cam_name: String,
    #[serde(rename = "camType")]
    _cam_type: String,
    #[serde(rename = "camPath")]
    _cam_path: String,
}

async fn new_camera_submit(
    State(state): State<Arc<AppState>>,
    Form(form): Form<NewCameraForm>
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    let new_cam = harmony_core::observer::Camera {
        name: form.cam_name.clone(),
        active_zone: vec![
            opencv::core::Point::new(0, 0),
            opencv::core::Point::new(100, 0),
            opencv::core::Point::new(100, 100),
            opencv::core::Point::new(0, 100),
        ],
        rotate: false,
        image_buffer: vec![],
        reference_frame: None,
    };
    machine.cc.cameras.insert(form.cam_name, new_cam);
    axum::response::Redirect::to("/configurator")
}

async fn calibrator_reset() -> impl IntoResponse {
    Html("Reset calibrator")
}

use tokio_stream::wrappers::BroadcastStream;
use futures_util::stream::StreamExt;
use axum::http::header;
use axum::response::Response;

async fn calibrator_cam_with_changes(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>
) -> impl IntoResponse {
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
    Html("Mode Controller UI")
}

async fn calibrator_commit() -> impl IntoResponse {
    Html("Calibration committed")
}

async fn calibrator_objects() -> impl IntoResponse {
    Html("Calibration objects table")
}

async fn calibrator_console() -> impl IntoResponse {
    Html("Observer console feed")
}
