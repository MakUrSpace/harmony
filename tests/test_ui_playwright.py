import pytest
from playwright.sync_api import Page, expect
import time
import os
import socket
import threading
import numpy as np
from unittest import mock
from collections import namedtuple

HarmonyServers = namedtuple("HarmonyServers", ["admin", "user"])


@pytest.fixture(scope="module")
def harmony_servers(mock_cv2):
    """Start both the Admin and User Harmony servers in-process using mocked hardware."""
    import uvicorn
    import harmony.harmonyServer as hs
    from harmony.harmonyServer import create_harmony_app

    # Reset module-level singleton so create_harmony_app initialises fresh
    hs._cc = None
    hs._cm = None
    hs.APPS.clear()

    p_hcc = mock.patch("harmony.harmonyServer.HexCaptureConfiguration")
    p_hm = mock.patch("harmony.harmonyServer.HarmonyMachine")
    p_rcs = mock.patch("harmony.harmonyServer.registerCaptureService")

    MockHexCC = p_hcc.start()
    MockHM = p_hm.start()
    p_rcs.start()

    mock_cam = mock.MagicMock()
    mock_cam.mostRecentFrame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.camName = "Camera 0"
    mock_cam.activeZone = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    mock_cam.activeZoneBoundingBox = (0, 0, 640, 480)
    mock_cam.cropToActiveZone.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.convertRealToCameraSpace.return_value = (10, 10)

    mock_cc = MockHexCC.return_value
    mock_cc.cameras = {"Camera 0": mock_cam}
    mock_cc.rsc = mock.MagicMock()
    mock_cc.rsc.closestConverterToRealCoord.return_value = mock_cam
    mock_cc.hex = mock.MagicMock()
    mock_cc.hex.size = 10.0
    mock_cc.hex.width = 1600.0
    mock_cc.hex.height = 1600.0
    mock_cc.hex.hex_nudges = {}
    mock_cc.camCoordToAxial.return_value = (0, 0)
    mock_cc.axial_to_pixel.return_value = (10, 10)
    mock_cc.cam_hex_at_axial.return_value = np.array([[10, 10], [20, 10], [25, 20], [20, 30], [10, 30], [5, 20]])
    mock_cc.show_grid = True
    mock_cc.show_objects = True
    mock_cc.memory = []
    mock_cc.objects = []
    mock_cc.grid_overlay.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cc.realSpaceBoundingBox.return_value = (0, 0, 1600, 1600)

    mock_cm = MockHM.return_value
    mock_cm.cc = mock_cc
    mock_cm.memory = []

    # 1. Admin App
    admin_app = create_harmony_app(template_name="Harmony.html")
    # Expose state
    admin_app.state.cc = mock_cc
    admin_app.state.cm = mock_cm

    # 2. User App (shares the same hs._cc and hs._cm singleton)
    user_app = create_harmony_app(template_name="HarmonyUser.html")
    # Expose state
    user_app.state.cc = mock_cc
    user_app.state.cm = mock_cm

    # Find free port for Admin
    with socket.socket() as s:
        s.bind(("", 0))
        admin_port = s.getsockname()[1]

    # Find free port for User
    with socket.socket() as s:
        s.bind(("", 0))
        user_port = s.getsockname()[1]

    admin_config = uvicorn.Config(
        admin_app, host="127.0.0.1", port=admin_port, log_level="error", loop="asyncio"
    )
    admin_server = uvicorn.Server(admin_config)

    user_config = uvicorn.Config(
        user_app, host="127.0.0.1", port=user_port, log_level="error", loop="asyncio"
    )
    user_server = uvicorn.Server(user_config)

    admin_thread = threading.Thread(target=admin_server.run, daemon=True)
    user_thread = threading.Thread(target=user_server.run, daemon=True)

    admin_thread.start()
    user_thread.start()

    # Wait for both servers to be ready
    ready = False
    for _ in range(40):
        admin_ready = False
        user_ready = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", admin_port)) == 0:
                admin_ready = True
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", user_port)) == 0:
                user_ready = True
        if admin_ready and user_ready:
            ready = True
            break
        time.sleep(0.25)

    if not ready:
        admin_server.should_exit = True
        user_server.should_exit = True
        p_hcc.stop()
        p_hm.stop()
        p_rcs.stop()
        pytest.fail("Harmony servers failed to start within 10 seconds")

    yield HarmonyServers(
        admin=f"http://127.0.0.1:{admin_port}",
        user=f"http://127.0.0.1:{user_port}"
    )

    admin_server.should_exit = True
    user_server.should_exit = True
    admin_thread.join(timeout=5)
    user_thread.join(timeout=5)
    p_hcc.stop()
    p_hm.stop()
    p_rcs.stop()


@pytest.fixture(scope="module")
def harmony_server(harmony_servers):
    return harmony_servers.admin


def test_homepage_loads(page: Page, harmony_server):
    page.goto(f"{harmony_server}/harmony/")
    expect(page.locator("h1")).to_contain_text("Harmony")


def test_camera_switching(page: Page, harmony_server):
    page.add_init_script("HTMLImageElement.prototype.decode = () => Promise.resolve();")
    page.goto(f"{harmony_server}/harmony/")

    # Find the Virtual Map camera button
    vmap_btn = page.locator("input[type='button'][value='Virtual Map']")
    expect(vmap_btn).to_be_visible()
    vmap_btn.click()
    expect(page.locator("#GameWorldHeader")).to_contain_text("VirtualMap")

    # Find the Camera 0 camera button
    cam_btn = page.locator("input[type='button'][value='Camera Camera 0']")
    expect(cam_btn).to_be_visible()
    cam_btn.click()
    expect(page.locator("#GameWorldHeader")).to_contain_text("Camera 0")

    # Find the All Views camera button (which should now be hidden/disabled)
    all_btn = page.locator("input[type='button'][value='All Views']")
    expect(all_btn).not_to_be_visible()


def test_admin_vs_user_permissions_js(page: Page, harmony_servers):
    admin_url = harmony_servers.admin
    user_url = harmony_servers.user

    page.add_init_script("HTMLImageElement.prototype.decode = () => Promise.resolve();")

    # 1. Verify Admin UI elements
    page.goto(f"{admin_url}/harmony/")
    expect(page.locator("input[value='Save Game']")).to_be_visible()
    expect(page.locator("input[value='Load Game']")).to_be_visible()
    expect(page.locator("button:has-text('Session Control Panels')")).to_be_visible()
    expect(page.locator("button:has-text('Configurator')")).to_be_visible()
    expect(page.locator("input[value='Reset Game']")).to_be_visible()

    # 2. Verify User UI elements (admin controls must be absent)
    page.goto(f"{user_url}/harmony/")
    expect(page.locator("input[value='Save Game']")).not_to_be_visible()
    expect(page.locator("input[value='Load Game']")).not_to_be_visible()
    expect(page.locator("button:has-text('Session Control Panels')")).not_to_be_visible()
    expect(page.locator("button:has-text('Configurator')")).not_to_be_visible()
    expect(page.locator("input[value='Reset Game']")).not_to_be_visible()

    # 3. Verify Admin vs User selection permissions (e.g. Delete Object button presence)
    # Import harmony server globals to mock object selection
    import harmony.harmonyServer as hs

    # Add a mock object in the shared HarmonyMachine memory
    mock_obj = mock.MagicMock()
    mock_obj.oid = "E2EObj"
    mock_obj.constituent_axials = [(0, 0)]
    hs._cm.memory = [mock_obj]

    # Configure coordinate tracking mock to return (0, 0)
    hs._cm.cc.camCoordToAxial.return_value = (0, 0)
    hs._cm.cc.changeSetToAxialCoord.return_value = (0, 0)
    hs._cc.changeSetToAxialCoord.return_value = (0, 0)

    # Admin selection
    page.goto(f"{admin_url}/harmony/?viewId=admin_e2e_session")
    # Submit coordinate (0, 0) click
    page.evaluate("""
        document.getElementById('selectedCamera').value = 'Camera 0';
        document.getElementById('selectedPixel').value = '[100, 200]';
        document.getElementById('appendPixel').value = 'false';
        document.getElementById('selectPixelForm').requestSubmit();
    """)
    expect(page.locator("#interactor")).to_contain_text("Delete Object")

    # User selection
    page.goto(f"{user_url}/harmony/?viewId=user_e2e_session")
    # Submit same coordinate (0, 0) click
    page.evaluate("""
        document.getElementById('selectedCamera').value = 'Camera 0';
        document.getElementById('selectedPixel').value = '[100, 200]';
        document.getElementById('appendPixel').value = 'false';
        document.getElementById('selectPixelForm').requestSubmit();
    """)
    expect(page.locator("#interactor")).not_to_contain_text("Delete Object")


def test_ui_latency(page: Page, harmony_server):
    # Measures the load time of the main page
    start_time = time.time()
    page.goto(f"{harmony_server}/harmony/")
    expect(page.locator("body")).to_be_visible()
    load_time = time.time() - start_time
    assert load_time < 3.0, f"UI page load time is too slow: {load_time}s"

    # Measures API latency of canvas data retrieval
    start_time = time.time()
    response = page.request.get(f"{harmony_server}/harmony/canvas_data/test_view")
    assert response.ok
    api_time = time.time() - start_time
    assert api_time < 0.5, f"Canvas data API response is too slow: {api_time}s"

def test_ui_cell_selection_drawing_sub100ms(page: Page, harmony_server):
    import time
    page.add_init_script("HTMLImageElement.prototype.decode = () => Promise.resolve();")
    page.goto(f"{harmony_server}/harmony/?viewId=perf_test")
    expect(page.locator("body")).to_be_visible()
    
    # Measure Selection API
    start = time.time()
    resp = page.request.post(
        f"{harmony_server}/harmony/select_pixel",
        form={
            "viewId": "perf_test",
            "selectedPixel": "[100, 200]",
            "selectedCamera": "Camera 0",
            "appendPixel": ""
        }
    )
    assert resp.ok
    sel_time = time.time() - start
    assert sel_time < 0.1, f"Selection took {sel_time}s, expected <0.1s"
    
    # Measure Draw API
    start = time.time()
    resp = page.request.post(
        f"{harmony_server}/harmony/draw_region/perf_test",
        form={
            "radius": "2",
            "region_type": "Burst"
        }
    )
    assert resp.ok
    draw_time = time.time() - start
    assert draw_time < 0.1, f"Drawing took {draw_time}s, expected <0.1s"
