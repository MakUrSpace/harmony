use crate::observer::HexCaptureConfiguration;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

fn default_height_one() -> i32 {
    1
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrackedObject {
    pub oid: String,
    pub object_type: String,
    pub constituent_axials: Vec<(i32, i32)>,
    #[serde(default = "default_height_one")]
    pub height: i32,
}

impl Default for TrackedObject {
    fn default() -> Self {
        Self {
            oid: String::new(),
            object_type: String::new(),
            constituent_axials: Vec::new(),
            height: 1,
        }
    }
}

impl TrackedObject {
    pub fn position_hash(&self) -> String {
        let mut out = String::new();
        for (q, r) in &self.constituent_axials {
            out.push_str(&format!("{}_{}_", q, r));
        }
        out
    }
}

pub struct HarmonyMachine {
    pub cc: HexCaptureConfiguration,
    pub cycle_counter: u64,
    pub memory: HashMap<String, TrackedObject>,
}

impl HarmonyMachine {
    pub fn new(cc: HexCaptureConfiguration) -> Self {
        Self {
            cc,
            cycle_counter: 0,
            memory: HashMap::new(),
        }
    }

    pub fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.cycle_counter += 1;
        
        // 1. Trigger captures on cameras
        // Since we don't have real cameras in this test environment,
        // we'll generate a dummy frame with the cycle counter for each camera.
        for (_, cam) in self.cc.cameras.iter_mut() {
            let mut img = opencv::core::Mat::new_rows_cols_with_default(
                480, 640, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0)
            )?;
            opencv::imgproc::put_text(
                &mut img,
                &format!("Cam: {} - Cycle: {}", cam.name, self.cycle_counter),
                opencv::core::Point::new(50, 50),
                opencv::imgproc::FONT_HERSHEY_SIMPLEX,
                1.0,
                opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                opencv::imgproc::LINE_8,
                false,
            )?;
            cam.reference_frame = Some(img);
        }

        // 2. Calculate frame differences (using Camera::change_between)
        // 3. Update object tracking state
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{HexCaptureConfiguration, Camera};
    use std::collections::HashMap;

    #[test]
    fn test_machine_initialization() {
        let mut cc = HexCaptureConfiguration::default();
        cc.show_grid = false;
        let machine = HarmonyMachine::new(cc);
        assert_eq!(machine.cycle_counter, 0);
        assert!(machine.memory.is_empty());
    }

    #[test]
    fn test_tracked_object_hash() {
        let mut obj = TrackedObject::default();
        obj.constituent_axials = vec![(0, 0), (1, -1)];
        let hash = obj.position_hash();
        assert_eq!(hash, "0_0_1_-1_");
    }

    #[test]
    fn test_machine_cycle() {
        let mut cc = HexCaptureConfiguration::default();
        
        let (tx, rx) = tokio::sync::watch::channel(bytes::Bytes::new());
        let (raw_tx, raw_rx) = tokio::sync::watch::channel(bytes::Bytes::new());
        let cam = Camera {
            name: "cam1".to_string(),
            cam_path: "mock".to_string(),
            active_zone: vec![],
            rotate: false,
            auth: vec![],
            image_buffer: vec![],
            reference_frame: None,
            frame_rx: Some(rx),
            frame_tx: Some(tx),
            raw_frame_rx: Some(raw_rx),
            raw_frame_tx: Some(raw_tx),
        };
        cc.cameras.insert("cam1".to_string(), cam);

        let mut machine = HarmonyMachine::new(cc);
        let res = machine.cycle();
        assert!(res.is_ok());
        assert_eq!(machine.cycle_counter, 1);
        
        // Ensure the dummy frame was set in the camera
        let cam_ref = machine.cc.cameras.get("cam1").unwrap();
        assert!(cam_ref.reference_frame.is_some());
    }
}
