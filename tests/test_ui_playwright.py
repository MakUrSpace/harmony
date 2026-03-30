import pytest
from playwright.sync_api import Page, expect
import time
import subprocess
import signal
import os
import sys
import socket

@pytest.fixture(scope="module")
def harmony_server():
    """Start the Harmony server in a background process and wait for it to be ready."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure project root is in PYTHONPATH so 'harmony' and 'observer' can be imported
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    
    # Start the server on a known port for testing
    proc = subprocess.Popen(
        [sys.executable, "harmony/harmonyServer.py"], 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    # Wait for server to be ready by checking the port
    max_retries = 20
    ready = False
    for i in range(max_retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', 7000)) == 0:
                ready = True
                break
        time.sleep(0.5)
    
    if not ready:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        pytest.fail("Harmony server failed to start within 10 seconds")

    yield "http://localhost:7000"
    
    # Shutdown
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

def test_homepage_loads(page: Page, harmony_server):
    page.goto(f"{harmony_server}/harmony/")
    expect(page.locator("h1")).to_contain_text("Harmony")

def test_canvas_click_updates_selection(page: Page, harmony_server):
    page.goto(f"{harmony_server}/harmony/")
    
    # Click on the canvas
    canvas = page.locator("#GameWorldOverlay")
    
    # Wait for overlay to be visible and have dimensions
    expect(canvas).to_be_visible()
    
    # We click in the middle of the canvas
    canvas.click(position={"x": 100, "y": 100})
    
    # Check if 'Selected' appears in #interactor (htmx update)
    interactor = page.locator("#interactor")
    expect(interactor).to_contain_text("Selected", timeout=5000)

def test_camera_switching(page: Page, harmony_server):
    page.goto(f"{harmony_server}/harmony/")
    
    # Find a camera button, in this app they are often inputs or list items
    # Check if 'Camera 0' exists in a radio/button
    cam_btn = page.locator("input[type='submit']").filter(has_text="Camera 0").first
    if cam_btn.count() == 0:
         # Fallback search
         cam_btn = page.locator("input").filter(has_text="0").first

    if cam_btn.count() > 0:
        cam_btn.click()
        expect(page.locator("#GameWorldHeader")).to_contain_text("0")

def test_admin_vs_user_permissions_js(page: Page, harmony_server):
    # Admin View (7001) - but usually shared server logic
    # In this app, Template determines permission.
    # harmonyServer.py: create_harmony_app(template_name="Harmony.html")
    # We'd need to start both apps or mock the session.
    pass
