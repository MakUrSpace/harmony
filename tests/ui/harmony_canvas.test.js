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
        window.HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            closePath: vi.fn(),
            stroke: vi.fn(),
            fill: vi.fn(),
            arc: vi.fn(),
            setLineDash: vi.fn()
        }));

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
});
