import { describe, it, expect, beforeEach, vi } from 'vitest';
import { JSDOM } from 'jsdom';
import fs from 'fs';
import path from 'path';

const scriptPath = path.resolve(__dirname, '../../harmony/harmony_templates/HarmonyCanvas.js');
const scriptContent = fs.readFileSync(scriptPath, 'utf8');

describe('HarmonyCanvas.js Unit Tests', () => {
    let dom;
    let window;
    let document;

    beforeEach(() => {
        // Setup a basic DOM structure that matches Harmony.html expectations
        dom = new JSDOM(`
            <!DOCTYPE html>
            <html>
                <body>
                    <input type="hidden" id="selectedCamera" value="Camera 0">
                    <img id="GameWorld" src="" style="width: 1920px; height: 1080px;">
                    <canvas id="GameWorldOverlay"></canvas>
                    <input type="hidden" id="selectedPixel">
                    <input type="hidden" id="appendPixel">
                    <form id="selectPixelForm"></form>
                </body>
            </html>
        `);
        window = dom.window;
        document = window.document;

        // Mock global canvas context
        window.mockCtx = {
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            closePath: vi.fn(),
            stroke: vi.fn(),
            fill: vi.fn(),
            arc: vi.fn(),
            setLineDash: vi.fn()
        };
        window.HTMLCanvasElement.prototype.getContext = vi.fn(() => window.mockCtx);

        // Mock getBoundingClientRect
        window.HTMLCanvasElement.prototype.getBoundingClientRect = vi.fn(() => ({
            left: 0,
            top: 0,
            width: 1920,
            height: 1080
        }));

        window.HTMLImageElement.prototype.getBoundingClientRect = vi.fn(() => ({
            left: 0,
            top: 0,
            width: 1920,
            height: 1080
        }));

        // Expose globals for the script to use
        global.window = window;
        global.document = document;
        global.navigator = window.navigator;
        global.setInterval = vi.fn();
        global.console = { ...console, error: vi.fn(), log: vi.fn() };

        // Eval the script content to load functions into global
        // We use a vm-like approach or just append to window
        // In HarmonyCanvas.js, functions are defined as "function name() { ... }"
        // which in a browser script puts them on window.
        // We can simulate this by wrapping the script and explicitly assigning to window.
        try {
            const evalScript = `
                (function(window, document, console) {
                    ${scriptContent}
                    // Capture defined functions and put them on window
                    window.initCanvasEditor = typeof initCanvasEditor !== 'undefined' ? initCanvasEditor : undefined;
                    window.dist = typeof dist !== 'undefined' ? dist : undefined;
                    window.handlePixelSelection = typeof handlePixelSelection !== 'undefined' ? handlePixelSelection : undefined;
                    window.translatePolyToQuadrant = typeof translatePolyToQuadrant !== 'undefined' ? translatePolyToQuadrant : undefined;
                    window.getNaturalDims = typeof getNaturalDims !== 'undefined' ? getNaturalDims : undefined;
                })(window, document, console);
            `;
            const fn = new Function('window', 'document', 'console', evalScript);
            fn(window, document, global.console);
        } catch (e) {
            console.error("Error evaluating script:", e);
        }
    });

    it('dist() correctly calculates Euclidean distance', () => {
        const p1 = { x: 0, y: 0 };
        const p2 = { x: 3, y: 4 };
        expect(window.dist(p1, p2)).toBe(5);
        
        const p1_arr = [10, 20];
        const p2_arr = [10, 30];
        expect(window.dist(p1_arr, p2_arr)).toBe(10);
    });

    it('initCanvasEditor throws error if IDs not found', () => {
        window.initCanvasEditor('nonExistent', {});
        expect(global.console.error).toHaveBeenCalledWith("Canvas or Image not found");
    });

    it('handlePixelSelection maps screen coordinates correctly for Default (1920x1080)', () => {
        const mockEvent = {
            clientX: 960, // Middle of 1920
            clientY: 540, // Middle of 1080
            shiftKey: false
        };
        
        // Mock clientWidth/Height via defineProperty
        const img = document.getElementById('GameWorld');
        Object.defineProperty(img, 'clientWidth', { value: 1920, configurable: true });
        Object.defineProperty(img, 'clientHeight', { value: 1080, configurable: true });

        window.handlePixelSelection(mockEvent);

        const pixelVal = JSON.parse(document.getElementById('selectedPixel').value);
        expect(pixelVal[0]).toEqual(960);
        expect(pixelVal[1]).toEqual(540);
        expect(document.getElementById('appendPixel').value).toEqual("false");
    });

    it('handlePixelSelection maps screen coordinates correctly for VirtualMap (1200x1200)', () => {
        document.getElementById('selectedCamera').value = "VirtualMap";
        const mockEvent = {
            clientX: 600, 
            clientY: 600,
            shiftKey: true
        };
        
        const img = document.getElementById('GameWorld');
        Object.defineProperty(img, 'clientWidth', { value: 1200, configurable: true });
        Object.defineProperty(img, 'clientHeight', { value: 1200, configurable: true });

        window.handlePixelSelection(mockEvent);

        const pixelVal = JSON.parse(document.getElementById('selectedPixel').value);
        expect(pixelVal[0]).toEqual(600);
        expect(pixelVal[1]).toEqual(600);
        expect(document.getElementById('appendPixel').value).toEqual("true");
    });

    it('translatePolyToQuadrant maps poly coordinates correctly under dynamic grid layout', () => {
        // Setup 4 cameras + 1 VirtualMap = 5 cameras (no No Camera padding)
        window.harmonyCanvasData = {
            cameras: ["Cam0", "Cam1", "Cam2", "Cam3", "VirtualMap"]
        };

        const poly = [{ x: 100, y: 200 }];
        
        // Test a camera quadrant (e.g. quad index 2, "Cam2", which is row 0, col 2)
        // Row 0 has 3 cams (fully filled), left_pad = 0.
        // offsetX = 2 * 960 = 1920. offsetY = 0 * 540 = 0.
        // scaleX = 0.5, scaleY = 0.5.
        // x = 100 * 0.5 + 1920 = 1970.
        // y = 200 * 0.5 + 0 = 100.
        const translatedCam = window.translatePolyToQuadrant(poly, 2, "Cam2");
        expect(translatedCam[0].x).toEqual(1970);
        expect(translatedCam[0].y).toEqual(100);

        // Test VirtualMap quadrant (e.g. quad index 4, "VirtualMap", which is row 1, index 1 within row)
        // Row 1 has 2 elements (Cam3, VirtualMap), so left_pad = (3 - 2)*960/2 = 480.
        // offsetX = 480 + 1 * 960 = 1440. offsetY = 1 * 540 = 540.
        // scaleX = 960 / 1200 = 0.8. scaleY = 540 / 1200 = 0.45.
        // x = 100 * 0.8 + 1440 = 1520.
        // y = 200 * 0.45 + 540 = 630.
        const translatedVM = window.translatePolyToQuadrant(poly, 4, "VirtualMap");
        expect(translatedVM[0].x).toEqual(1520);
        expect(translatedVM[0].y).toEqual(630);
    });

    it('handlePixelSelection maps screen coordinates correctly for dynamic grid layout All Views', () => {
        document.getElementById('selectedCamera').value = "All";
        window.harmonyCanvasData = {
            cameras: ["Cam0", "Cam1", "Cam2", "Cam3", "VirtualMap"]
        };

        // Middle of quadrant 4 ("VirtualMap", row 1, index 1 within row)
        // Row 1 has left padding 480. Active area starts at 480.
        // VirtualMap starts at 480 + 960 = 1440.
        // Let's click at local coordinate x = 480, y = 270 (relative to quadrant).
        // Click position in composite: clientX = 1440 + 480 = 1920. clientY = 540 + 270 = 810.
        const mockEvent = {
            clientX: 1920,
            clientY: 810,
            shiftKey: false
        };

        const img = document.getElementById('GameWorld');
        // Composite natural size is cols * 960 by rows * 540 = 3 * 960 by 2 * 540 = 2880 x 1080.
        // Let's set actual image client dimensions equal to natural dimensions so scale is 1.
        Object.defineProperty(img, 'clientWidth', { value: 2880, configurable: true });
        Object.defineProperty(img, 'clientHeight', { value: 1080, configurable: true });

        // Mock bounding rect to match
        img.getBoundingClientRect = vi.fn(() => ({
            left: 0,
            top: 0,
            width: 2880,
            height: 1080
        }));

        window.handlePixelSelection(mockEvent);

        // Clicked inside quadrant 4 ("VirtualMap").
        // Click was at local_x = 480, local_y = 270.
        // VirtualMap scaling: local_x * (1200 / 960) = 480 * 1.25 = 600.
        // local_y * (1200 / 540) = 270 * 2.2222 = 600.
        const pixelVal = JSON.parse(document.getElementById('selectedPixel').value);
        expect(pixelVal[0]).toEqual(600);
        expect(pixelVal[1]).toEqual(600);
        expect(document.getElementById('selectedCamera').value).toEqual("All");
    });

    it('initCanvasEditor correctly draws objects even with mismatched camera names', () => {
        // Setup image dimensions
        const img = document.getElementById('GameWorld');
        Object.defineProperty(img, 'clientWidth', { value: 1920, configurable: true });
        Object.defineProperty(img, 'clientHeight', { value: 1080, configurable: true });

        // Create virtual map image so that it exists if looked up
        const imgVM = document.createElement('img');
        imgVM.id = 'GameWorld_VirtualMap';
        Object.defineProperty(imgVM, 'clientWidth', { value: 1200, configurable: true });
        Object.defineProperty(imgVM, 'clientHeight', { value: 1200, configurable: true });
        document.body.appendChild(imgVM);

        // Mock canvas dimensions
        const canvas = document.getElementById('GameWorldOverlay');
        canvas.width = 1920;
        canvas.height = 1080;

        // Mock canvas context methods
        window.mockCtx.lineTo.mockClear();

        // 1. Camera suffix match (e.g. RTSPCamera0 mapping to suffix "0")
        document.getElementById('selectedCamera').value = "RTSPCamera0";
        const data = {
            objects: {
                Obj1: {
                    "0": [[100, 100], [200, 200], [300, 100]]
                }
            },
            selectable: ["Obj1"]
        };
        
        const editor = window.initCanvasEditor('GameWorldOverlay', data);
        expect(window.mockCtx.lineTo).toHaveBeenCalled();

        // 2. Case insensitive match (e.g. "virtualmap" matching "VirtualMap")
        window.mockCtx.lineTo.mockClear();
        document.getElementById('selectedCamera').value = "virtualmap";
        const dataVM = {
            objects: {
                ObjVM: {
                    "VirtualMap": [[10, 10], [20, 20], [30, 10]]
                }
            },
            selectable: ["ObjVM"]
        };
        const editorVM = window.initCanvasEditor('GameWorldOverlay', dataVM, null, null, "VirtualMap");
        expect(window.mockCtx.lineTo).toHaveBeenCalled();
    });
});


