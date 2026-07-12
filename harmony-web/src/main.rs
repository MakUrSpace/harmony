use askama::Template;
use axum::{
    extract::{State, Query},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Router,
};
use axum::http::header;
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

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlobalConfig {
    pub terrain: Vec<String>,
    pub groups: std::collections::HashMap<String, Vec<String>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    pub is_gm: bool,
    pub moveable_objects: Vec<String>,
    pub moveable_groups: Vec<String>,
    pub ally_groups: Vec<String>,
    pub enemy_groups: Vec<String>,
    pub targetable_groups: Vec<String>,
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
            is_gm: false,
            moveable_objects: vec![],
            moveable_groups: vec![],
            ally_groups: vec![],
            enemy_groups: vec![],
            targetable_groups: vec![],
            selection: CellSelection { first_cell: None, additional_cells: vec![] },
            selected_oid: None,
            show_grid: false,
            show_objects: true,
            can_publish_selection: true,
            published_selection: None,
        }
    }
}

impl SessionConfig {
    pub fn effective_moveable(&self, global: &GlobalConfig) -> Vec<String> {
        let mut result = self.moveable_objects.clone();
        for group in &self.moveable_groups {
            if let Some(oids) = global.groups.get(group) {
                result.extend(oids.clone());
            }
        }
        result
    }

    pub fn effective_allies(&self, global: &GlobalConfig) -> Vec<String> {
        let mut result = vec![];
        for group in &self.ally_groups {
            if let Some(oids) = global.groups.get(group) {
                result.extend(oids.clone());
            }
        }
        result
    }

    pub fn effective_enemies(&self, global: &GlobalConfig) -> Vec<String> {
        let mut result = vec![];
        for group in &self.enemy_groups {
            if let Some(oids) = global.groups.get(group) {
                result.extend(oids.clone());
            }
        }
        result
    }

    pub fn effective_targetable(&self, global: &GlobalConfig) -> Vec<String> {
        let mut result = vec![];
        for group in &self.targetable_groups {
            if let Some(oids) = global.groups.get(group) {
                result.extend(oids.clone());
            }
        }
        result
    }
}

use axum::extract::ws::{WebSocketUpgrade, WebSocket, Message as WsMessage};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct ChatMessage {
    pub author: String,
    pub content: String,
    pub timestamp: u64,
    pub from_discord: bool,
    pub channel: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum ChatStatus {
    Running,
    Paused,
    Stopped,
}

struct AppState {
    machine: Arc<Mutex<HarmonyMachine>>,
    sessions: Mutex<std::collections::HashMap<String, SessionConfig>>,
    global_config: Mutex<GlobalConfig>,
    chat_tx: tokio::sync::broadcast::Sender<ChatMessage>,
    chat_log: Mutex<Vec<ChatMessage>>,
    chat_status: Mutex<ChatStatus>,
}

struct Handler {
    chat_tx: tokio::sync::broadcast::Sender<ChatMessage>,
    channel_id: u64,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, _ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }
        if msg.channel_id.get() == self.channel_id {
            let chat_msg = ChatMessage {
                author: msg.author.name.clone(),
                content: msg.content.clone(),
                timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                from_discord: true,
                channel: "group".to_string(),
            };
            let _ = self.chat_tx.send(chat_msg);
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        tracing::info!("Discord bot connected as {}", ready.user.name);
        if let Err(e) = serenity::model::id::ChannelId::new(self.channel_id).say(&ctx.http, "Harmony Online!").await {
            tracing::error!("Failed to send Harmony Online message: {:?}", e);
        }
    }
}

async fn start_discord_bot(token: String, channel_id: u64, chat_tx: tokio::sync::broadcast::Sender<ChatMessage>, mut chat_rx: tokio::sync::broadcast::Receiver<ChatMessage>) {
    let token_prefix = if token.starts_with("Bot ") { token.clone() } else { format!("Bot {}", token.trim()) };
    let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT;
    let mut client = match Client::builder(&token_prefix, intents)
        .event_handler(Handler { chat_tx: chat_tx.clone(), channel_id })
        .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to create Discord client: {:?}", e);
            return;
        }
    };

    drop(chat_rx);

    if let Err(why) = client.start().await {
        tracing::error!("Discord client error: {:?}", why);
    }
}

#[derive(serde::Deserialize)]
struct ChatQuery {
    view_id: Option<String>,
}

async fn chat_ws(
    ws: WebSocketUpgrade,
    axum::extract::Query(query): axum::extract::Query<ChatQuery>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let view_id = query.view_id.unwrap_or_else(|| "unknown".to_string());
    ws.on_upgrade(move |socket| handle_chat_socket(socket, state, view_id))
}

async fn handle_chat_socket(socket: WebSocket, state: Arc<AppState>, view_id: String) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.chat_tx.subscribe();
    let state_for_send = state.clone();
    let view_id_for_send = view_id.clone();

    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let status = *state_for_send.chat_status.lock().await;
            if status != ChatStatus::Running && msg.author != "System" {
                continue;
            }

            let mut should_send = false;
            
            if msg.channel == "group" {
                should_send = true;
            } else {
                let sessions = state_for_send.sessions.lock().await;
                if let Some(receiver_session) = sessions.get(&view_id_for_send) {
                    if receiver_session.is_gm || receiver_session.ally_groups.contains(&msg.channel) {
                        should_send = true;
                    }
                }
                if view_id_for_send == msg.author {
                    should_send = true;
                }
            }

            if should_send {
                if let Ok(json) = serde_json::to_string(&msg) {
                    if sender.send(WsMessage::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let tx = state.chat_tx.clone();
    let state_for_recv = state.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(WsMessage::Text(text))) = receiver.next().await {
            match serde_json::from_str::<ChatMessage>(&text) {
                Ok(mut msg) => {
                    let status = *state_for_recv.chat_status.lock().await;
                    if status == ChatStatus::Running {
                        msg.from_discord = false;
                        msg.timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let _ = tx.send(msg);
                    }
                },
                Err(e) => {
                    tracing::error!("Failed to parse ChatMessage: {} from payload: {}", e, text);
                }
            }
        }
    });

    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    }
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

        std::env::set_var("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay");

        let mut cam = match VideoCapture::from_file(&final_cam_path, CAP_ANY) {
            Ok(mut c) => {
                let _ = c.set(opencv::videoio::CAP_PROP_BUFFERSIZE, 1.0);
                c
            },
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
                                    if let Ok(c0) = contours.get(0) {
                                        if let Ok(rect) = opencv::imgproc::bounding_rect(&c0) {
                                            if rect.width > 0 && rect.height > 0 {
                                                if let Ok(roi) = opencv::core::Mat::roi(&dst, rect) {
                                                    if let Ok(cloned) = roi.try_clone() {
                                                        processed_frame = cloned;
                                                    } else {
                                                        processed_frame = dst.clone();
                                                    }
                                                } else {
                                                    processed_frame = dst.clone();
                                                }
                                            } else {
                                                processed_frame = dst.clone();
                                            }
                                        } else {
                                            processed_frame = dst.clone();
                                        }
                                    } else {
                                        processed_frame = dst.clone();
                                    }
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

    let cc = HexCaptureConfiguration::load_from_file("observerConfiguration.json")
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
                discord_token: None,
                discord_channel_id: None,
                discord_client_id: None,
                discord_client_secret: None,
                embed_compcon: false,
            }
        });
        
    let machine = Arc::new(Mutex::new(HarmonyMachine::new(cc)));

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

    let (chat_tx, _) = tokio::sync::broadcast::channel(100);

    let state = Arc::new(AppState {
        machine: machine.clone(),
        sessions: Mutex::new(std::collections::HashMap::new()),
        global_config: Mutex::new(GlobalConfig::default()),
        chat_tx: chat_tx.clone(),
        chat_log: Mutex::new(Vec::new()),
        chat_status: Mutex::new(ChatStatus::Stopped),
    });

    let mut central_rx = chat_tx.subscribe();
    let state_for_log = state.clone();
    tokio::spawn(async move {
        while let Ok(msg) = central_rx.recv().await {
            if msg.author != "System" {
                let mut log = state_for_log.chat_log.lock().await;
                log.push(msg.clone());
                
                if msg.channel == "group" && !msg.from_discord {
                    let m = state_for_log.machine.lock().await;
                    let token = m.cc.discord_token.clone();
                    let channel_id_str = m.cc.discord_channel_id.clone();
                    if let (Some(t), Some(c_str)) = (token, channel_id_str) {
                        if let Ok(cid) = c_str.trim().parse::<u64>() {
                            let content = format!("**{}**: {}", msg.author, msg.content);
                            tokio::spawn(async move {
                                let token_prefix = if t.starts_with("Bot ") { t.clone() } else { format!("Bot {}", t.trim()) };
                                let http = serenity::http::Http::new(&token_prefix);
                                if let Err(e) = serenity::model::id::ChannelId::new(cid).say(&http, &content).await {
                                    tracing::error!("Failed to post group message to Discord: {:?}", e);
                                }
                            });
                        }
                    }
                }
            }
        }
    });

    let tx_for_bot = chat_tx.clone();
    let machine_for_bot = machine.clone();
    tokio::spawn(async move {
        let mut last_token = None;
        let mut last_channel_id = None;
        let mut current_bot_task: Option<tokio::task::JoinHandle<()>> = None;

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            
            let (token, channel_id_str) = {
                let m = machine_for_bot.lock().await;
                (m.cc.discord_token.clone(), m.cc.discord_channel_id.clone())
            };

            if token != last_token || channel_id_str != last_channel_id {
                if let Some(handle) = current_bot_task.take() {
                    tracing::info!("Stopping old Discord bot task...");
                    handle.abort();
                }

                last_token = token.clone();
                last_channel_id = channel_id_str.clone();

                if let (Some(t), Some(c_str)) = (token, channel_id_str) {
                    if let Ok(cid) = c_str.trim().parse::<u64>() {
                        tracing::info!("Starting Discord bot with channel ID: {}", cid);
                        let token_clone = t.clone();
                        let rx = tx_for_bot.subscribe();
                        let tx_clone = tx_for_bot.clone();
                        let handle = tokio::spawn(async move {
                            start_discord_bot(token_clone, cid, tx_clone, rx).await;
                        });
                        current_bot_task = Some(handle);
                    } else {
                        tracing::warn!("Failed to parse Discord channel ID: {:?}", c_str);
                    }
                }
            }
        }
    });

    tokio::spawn(async move {
        loop {
            {
                let mut m = machine.lock().await;
                if let Err(e) = m.cycle() {
                    tracing::error!("Machine cycle error: {}", e);
                }
            }
            // In python, the calibrator pool time was 1.0s and observer was 0.01s.
            // We can just sleep for a short duration.
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
    });

    let static_dir = std::env::var("HARMONY_STATIC_DIR").unwrap_or_else(|_| {
        if std::path::Path::new("harmony-web/static").exists() {
            "harmony-web/static".to_string()
        } else {
            "static".to_string()
        }
    });

    let admin_app = Router::new()
        .route("/", get(|| async { axum::response::Redirect::to("/harmony") }))
        .route("/harmony", get(harmony))
        .route("/harmony/chat_ws", axum::routing::get(chat_ws))
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
        .route("/harmony/chat/start", post(start_chat))
        .route("/harmony/chat/pause", post(pause_chat))
        .route("/harmony/chat/stop", post(stop_chat))
        .route("/harmony/set_overlays", post(set_overlays))
        .route("/harmony/control", get(session_list))
        .route("/harmony/control/:view_id", get(session_control_panel))
        .route("/harmony/control/:view_id/update", post(update_session_config))
        .route("/harmony/control/world/update", post(update_world_config))
        .route("/harmony/publish_selection", post(publish_selection))
        .route("/harmony/clear_published_selection", post(clear_published_selection))
        .route("/harmony/clear_all_published_selections", post(clear_all_published_selections))
        .route("/harmony/update_session_id", post(update_session_id))
        .route("/harmony/objects/:oid/rename", post(rename_object))
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
        .route("/compcon/", get(proxy_compcon_root))
        .route("/assets/*path", get(proxy_compcon_assets))
        .route("/icons/*path", get(proxy_compcon_icons))
        .route("/manifest.webmanifest", get(proxy_compcon_manifest))
        .nest_service("/static", ServeDir::new(static_dir.clone()))
        .fallback_service(ServeDir::new(static_dir.clone()))
        .route("/configurator/", get(configurator).post(configurator_save))
        .route("/configurator/clear_calibration/:name", post(configurator_clear_calibration))
        .route("/configurator/discord", post(configurator_discord))
        .route("/configurator/embed_compcon", post(configurator_embed_compcon))
        .with_state(state.clone());

    let user_app = Router::new()
        .route("/", get(harmony_user))
        .route("/harmony_user", get(harmony_user))
        .route("/harmony/chat_ws", axum::routing::get(chat_ws))
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
        .route("/harmony/update_session_id", post(update_session_id))
        .route("/harmony/objects/:oid/rename", post(rename_object))
        .route("/harmony/camWithChanges/:cam_name/:view_id", get(cam_with_changes_feed))
        .route("/compcon/", get(proxy_compcon_root))
        .route("/assets/*path", get(proxy_compcon_assets))
        .route("/icons/*path", get(proxy_compcon_icons))
        .route("/manifest.webmanifest", get(proxy_compcon_manifest))
        .nest_service("/static", ServeDir::new(static_dir.clone()))
        .fallback_service(ServeDir::new(static_dir.clone()))
        .with_state(state.clone());

    let discord_app = Router::new()
        .route("/", get(discord_activity))
        .route("/api/discord/token-exchange", post(token_exchange))
        .route("/harmony/chat_ws", axum::routing::get(chat_ws))
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
        .route("/harmony/update_session_id", post(update_session_id))
        .route("/harmony/objects/:oid/rename", post(rename_object))
        .route("/harmony/camWithChanges/:cam_name/:view_id", get(cam_with_changes_feed))
        .route("/compcon/", get(proxy_compcon_root))
        .route("/assets/*path", get(proxy_compcon_assets))
        .route("/icons/*path", get(proxy_compcon_icons))
        .route("/manifest.webmanifest", get(proxy_compcon_manifest))
        .nest_service("/static", ServeDir::new(static_dir.clone()))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state);

    let admin_addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let user_addr = SocketAddr::from(([0, 0, 0, 0], 8081));
    let discord_addr = SocketAddr::from(([0, 0, 0, 0], 8082));
    
    tracing::info!("Admin server listening on {}", admin_addr);
    tracing::info!("User server listening on {}", user_addr);
    tracing::info!("Discord server listening on {}", discord_addr);

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(&admin_addr).await.unwrap();
        axum::serve(listener, admin_app).await.unwrap();
    });

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(&discord_addr).await.unwrap();
        axum::serve(listener, discord_app).await.unwrap();
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
    discord_token: String,
    discord_channel_id: String,
    discord_client_id: String,
    discord_client_secret: String,
    embed_compcon: bool,
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
    let discord_token = machine.cc.discord_token.clone().unwrap_or_default();
    let discord_channel_id = machine.cc.discord_channel_id.clone().unwrap_or_default();
    let discord_client_id = machine.cc.discord_client_id.clone().unwrap_or_default();
    let discord_client_secret = machine.cc.discord_client_secret.clone().unwrap_or_default();

    let template = ConfiguratorTemplate { 
        cameras, 
        camera_names_json, 
        calibration_pts_json,
        grid_polys_json,
        show_grid, 
        show_objects, 
        hex,
        discord_token,
        discord_channel_id,
        discord_client_id,
        discord_client_secret,
        embed_compcon: machine.cc.embed_compcon,
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
struct DiscordForm {
    discord_token: String,
    discord_channel_id: String,
    discord_client_id: String,
    discord_client_secret: String,
}

async fn configurator_discord(
    State(state): State<Arc<AppState>>,
    Form(form): Form<DiscordForm>
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    tracing::info!("Updating discord config");
    machine.cc.discord_token = if form.discord_token.is_empty() { None } else { Some(form.discord_token.clone()) };
    machine.cc.discord_channel_id = if form.discord_channel_id.is_empty() { None } else { Some(form.discord_channel_id.clone()) };
    machine.cc.discord_client_id = if form.discord_client_id.is_empty() { None } else { Some(form.discord_client_id.clone()) };
    machine.cc.discord_client_secret = if form.discord_client_secret.is_empty() { None } else { Some(form.discord_client_secret.clone()) };
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save configuration: {}", e);
    }
    axum::response::Html(format!(
        r#"<form hx-post="/configurator/discord" hx-swap="outerHTML">
            <div class="d-flex flex-column gap-3 mb-3">
                <div>
                    <label class="form-label mb-0 fw-bold" for="discord_token">Bot Token</label>
                    <input type="password" class="form-control bg-light text-dark" name="discord_token" value="{}" placeholder="Enter Discord Bot Token">
                </div>
                <div>
                    <label class="form-label mb-0 fw-bold" for="discord_channel_id">Game Channel ID</label>
                    <input type="text" class="form-control bg-light text-dark" name="discord_channel_id" value="{}" placeholder="Enter Channel ID">
                </div>
                <div>
                    <label class="form-label mb-0 fw-bold" for="discord_client_id">Discord Client ID</label>
                    <input type="text" class="form-control bg-light text-dark" name="discord_client_id" value="{}" placeholder="Enter Discord Client ID (for Activities)">
                </div>
                <div>
                    <label class="form-label mb-0 fw-bold" for="discord_client_secret">Discord Client Secret</label>
                    <input type="password" class="form-control bg-light text-dark" name="discord_client_secret" value="{}" placeholder="Enter Discord Client Secret (for Activities)">
                </div>
            </div>
            <input type="submit" class="btn btn-primary" value="Save Discord Configuration">
            <div class="alert alert-success mt-2 p-1 text-center" style="font-size: 0.9rem; border-radius: 0;">Settings Saved!</div>
        </form>"#,
        form.discord_token, form.discord_channel_id, form.discord_client_id, form.discord_client_secret
    ))
}

#[derive(serde::Deserialize)]
struct EmbedCompconForm {
    embed_compcon: Option<String>,
}

async fn configurator_embed_compcon(
    State(state): State<Arc<AppState>>,
    Form(form): Form<EmbedCompconForm>,
) -> impl IntoResponse {
    tracing::info!("Route hit: /configurator/embed_compcon");
    let mut machine = state.machine.lock().await;
    machine.cc.embed_compcon = form.embed_compcon.is_some();
    if let Err(e) = machine.cc.save_to_file("observerConfiguration.json") {
        tracing::error!("Failed to save state to observerConfiguration.json: {:?}", e);
    }
    axum::response::Html(format!(
        r#"<form hx-post="/configurator/embed_compcon" hx-swap="outerHTML">
            <div class="form-check form-switch mb-3">
                <input class="form-check-input" type="checkbox" id="embed_compcon" name="embed_compcon" {}>
                <label class="form-check-label fw-bold" for="embed_compcon">Embed Comp/Con Interface</label>
            </div>
            <input type="submit" class="btn btn-primary" value="Save Comp/Con Settings">
            <div class="alert alert-success mt-2 p-1 text-center" style="font-size: 0.9rem; border-radius: 0;">Settings Saved!</div>
        </form>"#,
        if machine.cc.embed_compcon { "checked" } else { "" }
    ))
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
    mode: Option<String>,
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
    embed_compcon: bool,
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
    embed_compcon: bool,
}

#[derive(Template)]
#[template(path = "SessionList.html")]
struct SessionListTemplate {
    sessions: Vec<String>,
    objects: Vec<harmony_core::machine::TrackedObject>,
    global: GlobalConfig,
    object_groups: std::collections::HashMap<String, String>,
}

#[derive(Template)]
#[template(path = "ControlPanel.html")]
#[allow(non_snake_case)]
struct ControlPanelTemplate {
    viewId: String,
    config: SessionConfig,
    objects: Vec<harmony_core::machine::TrackedObject>,
    object_groups: std::collections::HashMap<String, String>,
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
    }).map(|s| s.trim().to_lowercase());

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
        r#"<input type="button" class="btn btn-info cam-btn" value="All Perspectives" data-camera="All"> "#
    ));
    camera_buttons.push_str(&format!(
        r#"<input type="button" class="btn btn-info cam-btn" value="Virtual Map" data-camera="VirtualMap"> "#
    ));
    for cam in &cams {
        camera_buttons.push_str(&format!(
            r#"<input type="button" class="btn btn-info cam-btn" value="Camera {}" data-camera="{}"> "#,
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
        embed_compcon: {
            let machine = state.machine.lock().await;
            machine.cc.embed_compcon
        },
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

#[derive(serde::Deserialize)]
struct TokenExchangeRequest {
    code: String,
}

async fn token_exchange(
    State(state): State<Arc<AppState>>,
    axum::extract::Json(payload): axum::extract::Json<TokenExchangeRequest>,
) -> impl IntoResponse {
    let (client_id, client_secret) = {
        let machine = state.machine.lock().await;
        (
            machine.cc.discord_client_id.clone().unwrap_or_default(),
            machine.cc.discord_client_secret.clone().unwrap_or_default(),
        )
    };

    if client_id.is_empty() || client_secret.is_empty() {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Discord client id or secret not configured").into_response();
    }

    let client = reqwest::Client::new();
    let res = client.post("https://discord.com/api/oauth2/token")
        .form(&[
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("grant_type", "authorization_code"),
            ("code", payload.code.as_str()),
        ])
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(json) = response.text().await {
                (axum::http::StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], json).into_response()
            } else {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to read Discord response").into_response()
            }
        },
        Err(e) => {
            tracing::error!("Discord token exchange failed: {}", e);
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to exchange token with Discord").into_response()
        }
    }
}

#[derive(Template)]
#[template(path = "discord_activity.html")]
struct DiscordActivityTemplate {
    view_id: String,
    default_camera: String,
    camera_buttons: String,
    harmony_url: String,
    show_grid_checked: String,
    show_objects_checked: String,
    embed_compcon: bool,
    discord_client_id: String,
}

async fn discord_activity(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HarmonyQuery>,
    jar: CookieJar,
) -> axum::response::Response {
    if query.mode.as_deref() == Some("compcon") {
        return proxy_compcon_root().await;
    }
    tracing::info!("Route hit: /discord_activity");
    let (view_id, default_camera, camera_buttons, show_grid_checked, show_objects_checked) =
        build_harmony_context(&state, &query, &jar).await;

    let (embed_compcon, discord_client_id) = {
        let machine = state.machine.lock().await;
        (machine.cc.embed_compcon, machine.cc.discord_client_id.clone().unwrap_or_default())
    };

    let template = DiscordActivityTemplate {
        view_id: view_id.clone(),
        default_camera,
        camera_buttons,
        harmony_url: "/harmony/".to_string(),
        show_grid_checked,
        show_objects_checked,
        embed_compcon,
        discord_client_id,
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
        embed_compcon: {
            let machine = state.machine.lock().await;
            machine.cc.embed_compcon
        },
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
    ally_groups: Vec<String>,
    selection: serde_json::Value,
    constituent_axials: std::collections::HashMap<String, Vec<(i32, i32)>>,
    published_selections: Vec<Vec<(i32, i32)>>,
    virtual_map_boundary: Vec<Vec<(i32, i32)>>,
    virtual_map_rect: (i32, i32, i32, i32),
    camera_rects: std::collections::HashMap<String, (i32, i32, i32, i32)>,
    chat_status: String,
}

async fn canvas_data(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let view_id = view_id.trim().to_lowercase();
    tracing::info!("Route hit: /harmony/canvas_data/{}", view_id);
    let machine = state.machine.lock().await;
    let sessions = state.sessions.lock().await;
    let session = sessions.get(&view_id).cloned().unwrap_or_default();
    let global_config = state.global_config.lock().await;

    let mut comp_cams = vec!["VirtualMap".to_string()];
    let mut other_cams: Vec<String> = machine.cc.cameras.keys().cloned().collect();
    other_cams.sort();
    comp_cams.extend(other_cams);
    while comp_cams.len() < 4 {
        comp_cams.push("No Camera".to_string());
    }
    comp_cams.truncate(4);

    let hex = machine.cc.hex.clone().unwrap_or_default();
    
    let mut selection_json = serde_json::json!({
        "firstCell": null,
        "additionalCells": []
    });
    
    let gen_cell_map = |q: i32, r: i32| -> serde_json::Value {
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

    let mut terrain = global_config.terrain.clone();
    let mut moveable = session.effective_moveable(&global_config);
    let mut allies = session.effective_allies(&global_config);
    let mut enemies = session.effective_enemies(&global_config);
    let mut targetable = enemies.clone();
    let mut selectable = Vec::new();
    
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
    
    selectable.extend(moveable.clone());
    selectable.extend(allies.clone());
    selectable.extend(enemies.clone());
    selectable.extend(targetable.clone());
    selectable.sort();
    selectable.dedup();

    let mut published_selections = Vec::new();
    for session in sessions.values() {
        if let Some(sel) = &session.published_selection {
            published_selections.push(sel.clone());
        }
    }

    let mut virtual_map_boundary = Vec::new();
    let mut virtual_map_rect = (0, 0, 1200, 1200);
    
    if let Ok((b_polys, b_rect)) = machine.cc.get_virtual_map_boundary() {
        for poly in b_polys {
            let mut pt_vec = Vec::new();
            for pt in poly {
                pt_vec.push((pt.x, pt.y));
            }
            virtual_map_boundary.push(pt_vec);
        }
        if b_rect.width > 0 && b_rect.height > 0 {
            virtual_map_rect = (b_rect.x, b_rect.y, b_rect.width, b_rect.height);
        }
    }

    let mut camera_rects = std::collections::HashMap::new();
    for (cam_name, cam) in &machine.cc.cameras {
        let mut pts = opencv::core::Vector::<opencv::core::Point>::new();
        for p in &cam.active_zone {
            pts.push(*p);
        }
        if let Ok(rect) = opencv::imgproc::bounding_rect(&pts) {
            camera_rects.insert(cam_name.clone(), (rect.x, rect.y, rect.width, rect.height));
        }
    }

    let chat_status = {
        let status = state.chat_status.lock().await;
        match *status {
            ChatStatus::Running => "Running",
            ChatStatus::Paused => "Paused",
            ChatStatus::Stopped => "Stopped",
        }.to_string()
    };

    let data = CanvasData {
        grid_cache_key: "default".to_string(), // Rust backend doesn't implement caching yet
        objects: objects_json,
        cameras: comp_cams,
        selectable,
        moveable,
        terrain,
        targetable,
        enemies,
        allies,
        ally_groups: session.ally_groups.clone(),
        selection: selection_json,
        constituent_axials,
        published_selections,
        virtual_map_boundary,
        virtual_map_rect,
        camera_rects,
        chat_status,
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
    let view_id_lower = payload.viewId.trim().to_lowercase();
    tracing::info!("Route hit: /harmony/select_pixel by view_id {}", view_id_lower);
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
        } else {
            if let Some(hex_cfg) = &machine.cc.hex {
                axial_coord = hex_cfg.pixel_to_axial(x, y);
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
        let session = sessions.entry(view_id_lower).or_insert_with(SessionConfig::default);
        
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
        let html = {
            let global = state.global_config.lock().await;
            render_interactor(&session, &machine, &global, &payload.viewId, is_admin)
        };
        axum::response::Html(html)
    }
}

pub fn render_interactor(session: &SessionConfig, machine: &harmony_core::machine::HarmonyMachine, global_config: &GlobalConfig, view_id: &str, is_admin: bool) -> String {
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

            if is_multi_cell && (session.effective_moveable(global_config).contains(oid) || is_admin) {
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
                if session.effective_moveable(global_config).contains(oid) || is_admin {
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
                    <button type='submit' class='btn btn-primary btn-sm w-100'>Broadcast Selection</button>
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

#[derive(serde::Deserialize)]
struct ClearSelectionPayload {
    viewId: String,
}

async fn clear_selection(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(payload): axum::extract::Form<ClearSelectionPayload>,
) -> impl IntoResponse {
    let view_id_lower = payload.viewId.trim().to_lowercase();
    tracing::info!("Route hit: /harmony/clear_selection by view_id {}", view_id_lower);
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&view_id_lower) {
        session.selection.first_cell = None;
        session.selection.additional_cells.clear();
        session.selected_oid = None;
    }
    axum::response::Html("".to_string())
}


struct ObjectGroup {
    name: String,
    id_name: String,
    objects: Vec<harmony_core::machine::TrackedObject>,
}

#[derive(Template)]
#[template(path = "objects.html")]
struct ObjectsTemplate {
    groups: Vec<ObjectGroup>,
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
        let global = state.global_config.lock().await;
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
                let is_terrain = global.terrain.contains(&obj.oid);
                let is_moveable = session.effective_moveable(&global).contains(&obj.oid);
                let is_ally = session.effective_allies(&global).contains(&obj.oid);
                let is_enemy = session.effective_enemies(&global).contains(&obj.oid);
                let is_targetable = is_enemy;
                let is_selectable = is_moveable || is_ally || is_enemy || is_targetable;
                
                if is_selectable { highest_tag = Some("Selectable"); }
                if is_terrain { highest_tag = Some("Terrain"); }
                if is_targetable { highest_tag = Some("Targetable"); }
                if is_enemy { highest_tag = Some("Enemies"); }
                if is_ally { highest_tag = Some("Allies"); }
                if is_moveable { highest_tag = Some("Moveable"); }
                
                if let Some(tag) = highest_tag {
                    obj.object_type = tag.to_string();
                }
            }
        }
    }
    
    let mut groups_map: std::collections::HashMap<String, Vec<harmony_core::machine::TrackedObject>> = std::collections::HashMap::new();
    for obj in objects {
        groups_map.entry(obj.object_type.clone()).or_default().push(obj);
    }
    
    let mut groups = Vec::new();
    let order = ["Moveable", "Allies", "Enemies", "Targetable", "Terrain", "Selectable"];
    for name in order {
        if let Some(objs) = groups_map.remove(name) {
            groups.push(ObjectGroup { 
                name: name.to_string(), 
                id_name: name.replace(" ", "-"),
                objects: objs 
            });
        }
    }
    let mut remaining: Vec<_> = groups_map.keys().cloned().collect();
    remaining.sort();
    for name in remaining {
        if let Some(objs) = groups_map.remove(&name) {
            groups.push(ObjectGroup { 
                name: name.clone(), 
                id_name: name.replace(" ", "-"),
                objects: objs 
            });
        }
    }
    
    let template = ObjectsTemplate {
        groups,
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
    
    let mut sessions = state.sessions.lock().await;
    for session in sessions.values_mut() {
        session.moveable_objects.retain(|x| x != &oid);
    }
    
    let mut global = state.global_config.lock().await;
    global.terrain.retain(|x| x != &oid);
    for group_oids in global.groups.values_mut() {
        group_oids.retain(|x| x != &oid);
    }
    
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
    
    let (mut sessions, machine, global) = (
        state.sessions.lock().await,
        state.machine.lock().await,
        state.global_config.lock().await,
    );
    
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        if let Some(obj) = machine.memory.get(&oid) {
            if !obj.constituent_axials.is_empty() {
                session.selection.first_cell = Some(obj.constituent_axials[0]);
                session.selection.additional_cells = obj.constituent_axials[1..].to_vec();
            }
        }
        axum::response::Html(render_interactor(session, &machine, &global, &payload.viewId, is_admin))
    } else {
        axum::response::Html("<div id='interactor'><h5>Failed to move</h5></div>".to_string())
    }
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
    let is_admin = payload.is_admin.as_deref() == Some("true");
    
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
            }
        }
    }
    
    let (mut sessions, machine, global) = (
        state.sessions.lock().await,
        state.machine.lock().await,
        state.global_config.lock().await,
    );
    
    if let Some(session) = sessions.get_mut(&payload.viewId) {
        if let Some(obj) = machine.memory.get(&oid) {
            if !obj.constituent_axials.is_empty() {
                session.selection.first_cell = Some(obj.constituent_axials[0]);
                session.selection.additional_cells = obj.constituent_axials[1..].to_vec();
            }
        }
        axum::response::Html(render_interactor(session, &machine, &global, &payload.viewId, is_admin))
    } else {
        axum::response::Html("<div id='interactor'><h5>Failed to rotate</h5></div>".to_string())
    }
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
            session.moveable_objects.retain(|x| x != &oid);
        }
        
        let mut global = state.global_config.lock().await;
        global.terrain.retain(|x| x != &oid);
        for group_oids in global.groups.values_mut() {
            group_oids.retain(|x| x != &oid);
        }
        
        axum::response::Html(format!("Updated type for {}", oid))
    } else {
        axum::response::Html("Object not found".to_string())
    }
}

async fn reset_game(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    tracing::info!("Route hit: /harmony/reset_game");
    
    let mut machine = state.machine.lock().await;
    machine.memory.clear();
    
    let mut sessions = state.sessions.lock().await;
    for session in sessions.values_mut() {
        *session = SessionConfig::default();
    }
    
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("HX-Refresh", axum::http::HeaderValue::from_static("true"));
    (axum::http::StatusCode::OK, headers, "Game Reset").into_response()
}

async fn start_chat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut status = state.chat_status.lock().await;
    *status = ChatStatus::Running;
    axum::response::Html("Chat Started".to_string())
}

async fn pause_chat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut status = state.chat_status.lock().await;
    *status = ChatStatus::Paused;
    axum::response::Html("Chat Paused".to_string())
}

async fn stop_chat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut status = state.chat_status.lock().await;
    *status = ChatStatus::Stopped;

    let mut collated_text = String::new();
    {
        let mut log = state.chat_log.lock().await;
        if !log.is_empty() {
            collated_text.push_str("**=== Session Ended: Chat Log ===**\n");
            
            let group_msgs: Vec<_> = log.iter().filter(|m| m.channel == "group").collect();
            if !group_msgs.is_empty() {
                collated_text.push_str("\n**Session Log:**\n");
                for m in group_msgs {
                    collated_text.push_str(&format!("[{}] {}: {}\n", m.timestamp, m.author, m.content));
                }
            }

            let mut team_channels: std::collections::HashMap<String, Vec<&ChatMessage>> = std::collections::HashMap::new();
            for m in log.iter().filter(|m| m.channel != "group") {
                team_channels.entry(m.channel.clone()).or_default().push(m);
            }

            for (channel, msgs) in team_channels {
                collated_text.push_str(&format!("\n**Team/Faction Log: {}**\n", channel));
                for m in msgs {
                    collated_text.push_str(&format!("[{}] {}: {}\n", m.timestamp, m.author, m.content));
                }
            }

            log.clear();
        }
    }

    if !collated_text.is_empty() {
        let m = state.machine.lock().await;
        let token = m.cc.discord_token.clone();
        let channel_id_str = m.cc.discord_channel_id.clone();
        if let (Some(t), Some(c_str)) = (token, channel_id_str) {
            if let Ok(cid) = c_str.trim().parse::<u64>() {
                let mut chunks = vec![];
                let mut current_chunk = String::new();
                for line in collated_text.lines() {
                    if current_chunk.len() + line.len() > 1900 {
                        chunks.push(current_chunk.clone());
                        current_chunk = String::new();
                    }
                    current_chunk.push_str(line);
                    current_chunk.push('\n');
                }
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                }
                
                tokio::spawn(async move {
                    let token_prefix = if t.starts_with("Bot ") { t.clone() } else { format!("Bot {}", t.trim()) };
                    let http = serenity::http::Http::new(&token_prefix);
                    for chunk in chunks {
                        if let Err(e) = serenity::model::id::ChannelId::new(cid).say(&http, &chunk).await {
                            tracing::error!("Failed to post session collated log to Discord: {:?}", e);
                        }
                    }
                });
            }
        }
    }

    axum::response::Html("Chat Stopped & Logs Collated".to_string()).into_response()
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
    let view_id_lower = form.view_id.trim().to_lowercase();
    tracing::info!("Route hit: /harmony/set_overlays for view {}", view_id_lower);
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&view_id_lower) {
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
        let mut width = 1200;
        let mut height = 1200;
        {
            let machine = state.machine.lock().await;
            if let Ok((_, rect)) = machine.cc.get_virtual_map_boundary() {
                if rect.width > 0 && rect.height > 0 {
                    width = rect.width;
                    height = rect.height;
                }
            }
        }
        let stream = async_stream::stream! {
            let img = opencv::core::Mat::new_rows_cols_with_default(height, width, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0)).unwrap();
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
    
    let global = state.global_config.lock().await.clone();
    let mut objects: Vec<harmony_core::machine::TrackedObject> = {
        let machine = state.machine.lock().await;
        machine.memory.values().cloned().collect()
    };
    objects.sort_by(|a, b| a.oid.cmp(&b.oid));
    
    let mut object_groups = std::collections::HashMap::new();
    for obj in &objects {
        let mut groups = Vec::new();
        for (gname, goids) in &global.groups {
            if goids.contains(&obj.oid) {
                groups.push(gname.clone());
            }
        }
        object_groups.insert(obj.oid.clone(), groups.join(","));
    }

    let template = SessionListTemplate { 
        sessions: sids,
        objects,
        global,
        object_groups,
    };
    match template.render() {
        Ok(html) => axum::response::Html(html).into_response(),
        Err(err) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {}", err)).into_response(),
    }
}

async fn session_control_panel(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let view_id_lower = view_id.trim().to_lowercase();
    let config = {
        let sessions = state.sessions.lock().await;
        sessions.get(&view_id_lower).cloned()
    };
    
    if let Some(config) = config {
        let mut objects: Vec<harmony_core::machine::TrackedObject> = {
            let machine = state.machine.lock().await;
            machine.memory.values().cloned().collect()
        };
        objects.sort_by(|a, b| a.oid.cmp(&b.oid));
        
        let groups = state.global_config.lock().await.groups.clone();
        let mut object_groups = std::collections::HashMap::new();
        for (group_name, oids) in groups {
            for oid in oids {
                object_groups.insert(oid, group_name.clone());
            }
        }
        
        let template = ControlPanelTemplate {
            viewId: view_id_lower.clone(),
            config: config.clone(),
            objects,
            object_groups,
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
    axum::extract::Form(payload): axum::extract::Form<PublishSelectionForm>,
) -> impl IntoResponse {
    let view_id_lower = payload.viewId.trim().to_lowercase();
    let mut sessions = state.sessions.lock().await;
    let mut published = None;
    if let Some(session) = sessions.get(&view_id_lower) {
        if session.can_publish_selection {
            if let Some(first) = session.selection.first_cell {
                let mut cells = vec![first];
                cells.extend(&session.selection.additional_cells);
                published = Some(cells);
            }
        }
    }
    
    if let Some(session) = sessions.get_mut(&view_id_lower) {
        session.published_selection = published;
    }
    
    axum::response::Html("<div id='publishedFeedback' class='alert alert-success mt-2 p-1'>Selection broadcasted globally.</div>")
}

async fn clear_published_selection(
    State(state): State<Arc<AppState>>,
    axum::extract::Form(payload): axum::extract::Form<PublishSelectionForm>,
) -> impl IntoResponse {
    let view_id_lower = payload.viewId.trim().to_lowercase();
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&view_id_lower) {
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

async fn update_world_config(
    State(state): State<Arc<AppState>>,
    Form(form): Form<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let mut global_terrain = Vec::new();
    let mut global_groups: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

    for (k, v) in form {
        if let Some(oid) = k.strip_suffix("_terrain") {
            global_terrain.push(oid.to_string());
        } else if k.starts_with("group_") {
            let oid = k.strip_prefix("group_").unwrap().to_string();
            for group in v.split(',') {
                let group = group.trim().to_string();
                if !group.is_empty() {
                    global_groups.entry(group).or_default().push(oid.clone());
                }
            }
        }
    }
    
    {
        let mut global = state.global_config.lock().await;
        global.terrain = global_terrain;
        global.groups = global_groups;
    }
    axum::response::Redirect::to("/harmony/control").into_response()
}

async fn update_session_config(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(view_id): axum::extract::Path<String>,
    Form(form): Form<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let view_id_lower = view_id.trim().to_lowercase();
    let mut config = SessionConfig::default();
    let mut can_publish = false;
    let mut is_gm = false;

    for (k, v) in form {
        if k == "can_publish_selection" {
            can_publish = true;
        } else if k == "is_gm" {
            is_gm = true;
        } else if let Some(oid) = k.strip_suffix("_moveable") {
            config.moveable_objects.push(oid.to_string());
        } else if k == "moveable_groups" {
            config.moveable_groups = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        } else if k == "ally_groups" {
            config.ally_groups = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        } else if k == "enemy_groups" {
            config.enemy_groups = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }
    }
    
    config.can_publish_selection = can_publish;
    config.is_gm = is_gm;

    {
        let mut sessions = state.sessions.lock().await;
        let session = sessions.entry(view_id_lower.clone()).or_insert_with(SessionConfig::default);
        config.selection = session.selection.clone();
        config.selected_oid = session.selected_oid.clone();
        config.show_grid = session.show_grid;
        config.show_objects = session.show_objects;
        config.published_selection = session.published_selection.clone();
        *session = config;
    }
    
    axum::response::Redirect::to(&format!("/harmony/control/{}", view_id_lower)).into_response()
}

#[derive(serde::Deserialize)]
struct UpdateSessionIdForm {
    viewId: String,
    newViewId: String,
}

async fn update_session_id(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Form(form): Form<UpdateSessionIdForm>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.lock().await;
    let old_vid = form.viewId.trim().to_lowercase();
    let new_vid = form.newViewId.trim().to_lowercase();
    if !new_vid.is_empty() && new_vid != old_vid {
        if let Some(session) = sessions.remove(&old_vid) {
            sessions.insert(new_vid.to_string(), session);
        }
    }
    
    let mut redirect_url = format!("/?view_id={}", new_vid);
    if let Some(referer) = headers.get(axum::http::header::REFERER) {
        if let Ok(referer_str) = referer.to_str() {
            if let Some(base) = referer_str.split('?').next() {
                redirect_url = format!("{}?view_id={}", base, new_vid);
            }
        }
    }
    
    let mut resp_headers = axum::http::HeaderMap::new();
    resp_headers.insert(axum::http::header::LOCATION, axum::http::HeaderValue::from_str(&redirect_url).unwrap());
    resp_headers.insert(axum::http::header::SET_COOKIE, axum::http::HeaderValue::from_str(&format!("session_view_id={}; Path=/", new_vid)).unwrap());
    
    (axum::http::StatusCode::SEE_OTHER, resp_headers)
}

#[derive(serde::Deserialize)]
struct RenameObjectForm {
    new_oid: String,
}

async fn rename_object(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(old_oid): axum::extract::Path<String>,
    Form(form): Form<RenameObjectForm>,
) -> impl IntoResponse {
    let new_oid = form.new_oid.trim();
    if new_oid.is_empty() || new_oid == old_oid {
        return (axum::http::StatusCode::OK, "No change").into_response();
    }
    
    let mut machine = state.machine.lock().await;
    if let Some(mut obj) = machine.memory.remove(&old_oid) {
        obj.oid = new_oid.to_string();
        machine.memory.insert(new_oid.to_string(), obj);
    }
    
    let replace = |list: &mut Vec<String>| {
        if let Some(pos) = list.iter().position(|x| x == &old_oid) {
            list[pos] = new_oid.to_string();
        }
    };
    
    {
        let mut sessions = state.sessions.lock().await;
        for session in sessions.values_mut() {
            replace(&mut session.moveable_objects);
        }
    }
    {
        let mut global = state.global_config.lock().await;
        replace(&mut global.terrain);
        for group_oids in global.groups.values_mut() {
            replace(group_oids);
        }
    }
    
    (axum::http::StatusCode::OK, "Renamed").into_response()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GameSave {
    memory: std::collections::HashMap<String, harmony_core::machine::TrackedObject>,
    sessions: std::collections::HashMap<String, SessionConfig>,
    global_config: GlobalConfig,
}

#[derive(serde::Deserialize)]
struct GameSaveForm {
    game_name: String,
}

fn sanitize_filename(name: &str) -> String {
    let mut safe_name: String = name.chars().filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
    if safe_name.is_empty() {
        safe_name = "Harmony".to_string();
    }
    safe_name
}

async fn save_game(
    State(state): State<Arc<AppState>>,
    Form(form): Form<GameSaveForm>,
) -> impl IntoResponse {
    let filename = format!("{}.json", sanitize_filename(&form.game_name));
    
    let save_data = GameSave {
        memory: state.machine.lock().await.memory.clone(),
        sessions: state.sessions.lock().await.clone(),
        global_config: state.global_config.lock().await.clone(),
    };
    
    if let Ok(json_str) = serde_json::to_string_pretty(&save_data) {
        if let Err(e) = std::fs::write(&filename, json_str) {
            tracing::error!("Failed to save game to {}: {}", filename, e);
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to write save file").into_response();
        }
    } else {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to serialize save data").into_response();
    }
    
    (axum::http::StatusCode::OK, format!("Saved to {}", filename)).into_response()
}

async fn load_game(
    State(state): State<Arc<AppState>>,
    Form(form): Form<GameSaveForm>,
) -> impl IntoResponse {
    let filename = format!("{}.json", sanitize_filename(&form.game_name));
    
    match std::fs::read_to_string(&filename) {
        Ok(json_str) => {
            match serde_json::from_str::<GameSave>(&json_str) {
                Ok(save_data) => {
                    let mut machine = state.machine.lock().await;
                    let mut sessions = state.sessions.lock().await;
                    let mut global_config = state.global_config.lock().await;
                    
                    machine.memory = save_data.memory;
                    *sessions = save_data.sessions;
                    *global_config = save_data.global_config;
                    
                    let mut headers = axum::http::HeaderMap::new();
                    headers.insert("HX-Refresh", axum::http::HeaderValue::from_static("true"));
                    (axum::http::StatusCode::OK, headers, format!("Loaded {}", filename)).into_response()
                }
                Err(e) => {
                    tracing::error!("Failed to deserialize game save {}: {}", filename, e);
                    (axum::http::StatusCode::BAD_REQUEST, "Invalid save file format").into_response()
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to read game save {}: {}", filename, e);
            (axum::http::StatusCode::NOT_FOUND, "Save file not found").into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_id_case_insensitivity() {
        let (chat_tx, _) = tokio::sync::broadcast::channel(16);
        let state = AppState {
            machine: Arc::new(tokio::sync::Mutex::new(harmony_core::machine::HarmonyMachine::new(
                harmony_core::observer::HexCaptureConfiguration {
                    cameras: std::collections::HashMap::new(),
                    rsc: std::collections::HashMap::new(),
                    hex: None,
                    show_grid: false,
                    show_objects: false,
                    grid_polys_cache: std::collections::HashMap::new(),
                    calibration_plan: serde_json::Value::Null,
                    discord_token: None,
                    discord_channel_id: None,
                    discord_client_id: None,
                    discord_client_secret: None,
                    embed_compcon: false,
                }
            ))),
            sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            global_config: tokio::sync::Mutex::new(GlobalConfig::default()),
            chat_tx,
            chat_log: tokio::sync::Mutex::new(Vec::new()),
            chat_status: tokio::sync::Mutex::new(ChatStatus::Stopped),
        };

        // 1. Build context with mixed-case session
        let query = HarmonyQuery {
            view_id: Some("Test-Session-123".to_string()),
            mode: None,
        };
        let jar = axum_extra::extract::CookieJar::new();
        let (vid_1, _, _, _, _) = build_harmony_context(&state, &query, &jar).await;

        // The returned ID should be lowercase
        assert_eq!(vid_1, "test-session-123");

        // 2. Build context with lowercase session ID
        let query_lower = HarmonyQuery {
            view_id: Some("test-session-123".to_string()),
            mode: None,
        };
        let (vid_2, _, _, _, _) = build_harmony_context(&state, &query_lower, &jar).await;

        assert_eq!(vid_1, vid_2);

        // Verify only 1 session was created in the state map
        let sessions = state.sessions.lock().await;
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key("test-session-123"));
    }
}


async fn proxy_compcon_root() -> axum::response::Response {
    proxy_request("https://compcon.app/").await
}

async fn proxy_compcon_assets(axum::extract::Path(path): axum::extract::Path<String>) -> axum::response::Response {
    proxy_request(&format!("https://compcon.app/assets/{}", path)).await
}

async fn proxy_compcon_icons(axum::extract::Path(path): axum::extract::Path<String>) -> axum::response::Response {
    proxy_request(&format!("https://compcon.app/icons/{}", path)).await
}

async fn proxy_compcon_manifest() -> axum::response::Response {
    proxy_request("https://compcon.app/manifest.webmanifest").await
}

async fn proxy_request(url: &str) -> axum::response::Response {
    use axum::response::IntoResponse;
    let client = reqwest::Client::new();
    match client.get(url).send().await {
        Ok(res) => {
            let status = axum::http::StatusCode::from_u16(res.status().as_u16()).unwrap();
            let mut headers = axum::http::HeaderMap::new();
            for (key, value) in res.headers() {
                let key_str = key.as_str().to_lowercase();
                if key_str != "content-encoding" 
                    && key_str != "transfer-encoding" 
                    && key_str != "content-length" 
                    && key_str != "x-frame-options"
                    && key_str != "content-security-policy"
                {
                    if let Ok(val) = axum::http::HeaderValue::from_bytes(value.as_bytes()) {
                        headers.insert(axum::http::HeaderName::from_bytes(key.as_str().as_bytes()).unwrap(), val);
                    }
                }
            }
            if let Ok(bytes) = res.bytes().await {
                (status, headers, bytes).into_response()
            } else {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to read body").into_response()
            }
        },
        Err(e) => {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to proxy: {}", e)).into_response()
        }
    }
}
