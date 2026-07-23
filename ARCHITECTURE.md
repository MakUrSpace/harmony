# Harmony Architecture

Harmony integrates tabletop gaming utilities into a game recording and refereeing system. Following a recent migration, the system's core backend is written entirely in Rust, utilizing the Axum web framework and OpenCV for computer vision.

This document serves to provide a high-level overview of the system and clarify several of the more complex engineering choices within the codebase.

## System Overview

The project is split into two primary Rust crates:

- **`harmony-core`**: The underlying engine of the system. This crate contains the `HarmonyMachine` state, handles the Hex grid mathematics, and parses observer configurations (such as camera settings and spatial mappings).
- **`harmony-web`**: The user-facing web server built on Axum. It serves the HTML UI (via Askama templates and HTMX), handles WebSocket connections for real-time chat and state synchronization, and integrates with Discord.

## Anticipatory Documentation

Some areas of the codebase are dense or rely on specific environmental configurations. The sections below provide context to help new developers understand these systems.

### Nix, Rust, and OpenCV Integration

Harmony relies on [OpenCV](https://opencv.org/) for processing camera streams and tracking objects. The Rust bindings for OpenCV require linking against C++ libraries and generating FFI (Foreign Function Interface) bindings on the fly during the build process.

If you examine `flake.nix`, you will notice the inclusion of `clang`, `pkg-config`, and `rustPlatform.bindgenHook` in the `harmony-rs` package definition. 
- `pkg-config` is used to locate the system's OpenCV installation.
- `clang` and `rustPlatform.bindgenHook` (which configures `LIBCLANG_PATH`) are required by the `rust-bindgen` tool to parse the C++ headers and generate the Rust bindings.

When developing locally using `nix develop`, the `shellHook` explicitly exports `LIBCLANG_PATH` so that `cargo build` and IDE tools (like rust-analyzer) can successfully compile the `opencv` crate.

### Camera Stream Threading Model

In `harmony-web/src/main.rs`, you will find a function named `spawn_camera_stream`. This function is responsible for connecting to a camera's RTSP stream, fetching frames, cropping them to the user-defined `Active Zone`, and distributing the frames to connected clients.

**Why `tokio::task::spawn_blocking`?**
Because the `opencv` crate's `VideoCapture::read` function is synchronous and blocking, running it directly within Axum's async runtime would stall the web server. To prevent this, each camera stream is spawned onto a dedicated blocking thread via `tokio::task::spawn_blocking`. 

These threads continuously read frames in a `loop`, process them, and then use asynchronous channels (`tokio::sync::watch`) to broadcast the JPEG-encoded bytes back to the Axum handlers, which can then stream them to the browser without blocking.

### Multi-Port Routing (ngrok)

When running the `harmony-ngrok` command via Nix, a `tmux` session is launched that sets up multiple `ngrok` tunnels pointing to different ports.

While the primary Rust server runs on a single port (default `8081` internally for standard Harmony), the script allocates several logical domains/ports for different aspects of the application. This allows administrators to share specific URLs with players while keeping GM tools private:

- **Port `8081` (Harmony URL)**: The main player-facing web interface.
- **Port `8080` (Admin URL)**: The GM (Game Master) configuration and administration panel.
- **Port `8082` (Discord URL)**: Endpoints for Discord activity integration.
- **Port `8083` (VR URL)**: Experimental endpoints for Virtual Reality views.

If you are modifying the networking stack or adding new proxy routes, ensure you test across the relevant ports mapped in the `flake.nix` tmux script.
