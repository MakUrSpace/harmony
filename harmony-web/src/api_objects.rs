use axum::{
    extract::{Path, State, Form},
    response::IntoResponse,
};
use std::sync::Arc;
use crate::AppState;
use harmony_core::machine::TrackedObject;

#[derive(serde::Deserialize)]
pub struct DefineObjectForm {
    name: String,
    cells: String, // JSON array of [q, r] tuples
}

pub async fn define_object(
    State(state): State<Arc<AppState>>,
    Form(form): Form<DefineObjectForm>,
) -> impl IntoResponse {
    let cells: Result<Vec<(i32, i32)>, _> = serde_json::from_str(&form.cells);
    let axials = cells.unwrap_or_default();
    
    let mut machine = state.machine.lock().await;
    let oid = if form.name.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        form.name.clone()
    };
    
    let obj = machine.memory.entry(oid.clone()).or_insert_with(|| TrackedObject {
        oid: oid.clone(),
        object_type: "None".to_string(),
        constituent_axials: vec![],
    });
    
    obj.constituent_axials = axials;
    
    axum::response::Html(format!("Object defined: {}", oid))
}

pub async fn delete_object(
    State(state): State<Arc<AppState>>,
    Path(oid): Path<String>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    machine.memory.remove(&oid);
    axum::response::Html(format!("Object deleted: {}", oid))
}

#[derive(serde::Deserialize)]
pub struct UpdateTypeForm {
    #[serde(rename = "objectType")]
    object_type: String,
}

pub async fn update_object_type(
    State(state): State<Arc<AppState>>,
    Path(oid): Path<String>,
    Form(form): Form<UpdateTypeForm>,
) -> impl IntoResponse {
    let mut machine = state.machine.lock().await;
    if let Some(obj) = machine.memory.get_mut(&oid) {
        obj.object_type = form.object_type;
        axum::response::Html(format!("Updated type for {}", oid))
    } else {
        axum::response::Html("Object not found".to_string())
    }
}
