import pytest
from playwright.sync_api import Page, expect
import time
import os
import socket
import threading
import numpy as np
from unittest import mock
from collections import namedtuple

import subprocess

HarmonyServers = namedtuple("HarmonyServers", ["admin", "user"])

@pytest.fixture(scope="module")
def harmony_servers():
    """Start the Rust Harmony server in a subprocess."""
    # Write a dummy config for testing
    import json
    config = {
        "hex": {"size": 10.0, "width": 1600, "height": 1600, "offset_xy": [0,0], "anchor_xy": [0,0]},
        "show_grid": True,
        "show_objects": True,
        "Camera 0": {
            "addr": "mock",
            "rot": False,
            "az": "[[0,0],[0,480],[640,480],[640,0]]"
        },
        "rsc": []
    }
    with open("observerConfiguration.json", "w") as f:
        json.dump(config, f)
        
    # Launch cargo run inside nix develop
    proc = subprocess.Popen(
        ["nix", "develop", ".", "-c", "cargo", "run", "--manifest-path", "harmony-web/Cargo.toml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    admin_port = 8080
    user_port = 8081

    # Wait for the server to be ready
    ready = False
    for _ in range(120):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", admin_port)) == 0:
                ready = True
                break
        time.sleep(1)

    if not ready:
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        pytest.fail(f"Rust Harmony server failed to start within 120 seconds\nSTDOUT:\n{out.decode()}\nSTDERR:\n{err.decode()}")

    yield HarmonyServers(
        admin=f"http://127.0.0.1:{admin_port}",
        user=f"http://127.0.0.1:{user_port}"
    )

    proc.terminate()
    proc.wait(timeout=5)

@pytest.fixture(scope="module")
def harmony_server(harmony_servers):
    return harmony_servers.admin

def test_homepage_loads(page: Page, harmony_server):
    page.goto(f"{harmony_server}/harmony", wait_until="domcontentloaded")
    expect(page.locator("h1")).to_contain_text("Harmony")


def test_camera_switching(page: Page, harmony_server):
    page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Page error: {err}"))
    page.on("response", lambda res: print(f"Response: {res.url} - {res.status}") if res.status >= 400 else None)
    page.add_init_script("HTMLImageElement.prototype.decode = () => Promise.resolve();")
    page.goto(f"{harmony_server}/harmony", wait_until="domcontentloaded")
    page.wait_for_function("typeof window.harmonyCanvasData !== 'undefined'")

    # Find the Virtual Map camera button
    vmap_btn = page.locator("input[type='button'][value='Virtual Map']")
    expect(vmap_btn).to_be_visible()
    vmap_btn.click()
    expect(page.locator("#GameWorldHeader")).to_contain_text("VirtualMap")

    print("AVAILABLE BUTTONS:")
    for btn in page.locator(".cam-btn").all():
        print(btn.get_attribute("value"))

    # Find the Camera 0 camera button
    cam_btn = page.locator("input[type='button'][value='Camera Camera 0']")
    expect(cam_btn).to_be_visible()
    cam_btn.click()
    expect(page.locator("#GameWorldHeader")).to_contain_text("Camera 0")

    # Find the All Views camera button (which should now be hidden/disabled)
    all_btn = page.locator("input[type='button'][value='All Views']")
    expect(all_btn).not_to_be_visible()


def test_admin_vs_user_permissions_js(page: Page, harmony_servers):
    page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Page error: {err}"))
    page.on("response", lambda res: print(f"Response: {res.url} - {res.status}") if res.status >= 400 else None)
    admin_url = harmony_servers.admin
    user_url = harmony_servers.user

    page.add_init_script("HTMLImageElement.prototype.decode = () => Promise.resolve();")

    # 1. Verify Admin UI elements
    page.goto(f"{admin_url}/harmony", wait_until="domcontentloaded")
    expect(page.locator("input[value='Save Game']")).to_be_visible()
    expect(page.locator("input[value='Load Game']")).to_be_visible()
    expect(page.locator("button:has-text('Session Control Panels')")).to_be_visible()
    expect(page.locator("button:has-text('Configurator')")).to_be_visible()
    expect(page.locator("input[value='Reset Game']")).to_be_visible()

    # 2. Verify User UI elements (admin controls must be absent)
    page.goto(f"{user_url}/harmony_user", wait_until="domcontentloaded")
    page.wait_for_function("typeof window.harmonyCanvasData !== 'undefined'")
    expect(page.locator("input[value='Save Game']")).not_to_be_visible()
    expect(page.locator("input[value='Load Game']")).not_to_be_visible()
    expect(page.locator("button:has-text('Session Control Panels')")).not_to_be_visible()
    expect(page.locator("button:has-text('Configurator')")).not_to_be_visible()
    expect(page.locator("input[value='Reset Game']")).not_to_be_visible()




def test_ui_latency(page: Page, harmony_server):
    # Measures the load time of the main page
    start_time = time.time()
    page.goto(f"{harmony_server}/harmony", wait_until="domcontentloaded")
    expect(page.locator("body")).to_be_visible()
    load_time = time.time() - start_time
    assert load_time < 3.0, f"UI page load time is too slow: {load_time}s"

    # Measures API latency of canvas data retrieval
    start_time = time.time()
    response = page.request.get(f"{harmony_server}/harmony/canvas_data/test_view")
    assert response.ok
    api_time = time.time() - start_time
    assert api_time < 0.5, f"Canvas data API response is too slow: {api_time}s"


