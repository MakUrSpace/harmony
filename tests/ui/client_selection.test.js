import { describe, it, expect, beforeEach, vi } from 'vitest';
import { JSDOM } from 'jsdom';
import fs from 'fs';
import path from 'path';

const scriptPath = path.resolve(__dirname, '../../harmony/harmony_templates/HarmonyCanvas.js');
const scriptContent = fs.readFileSync(scriptPath, 'utf8');

describe('Client-side Cell Selection', () => {
    let dom;
    let window;
    let document;

    beforeEach(() => {
        dom = new JSDOM(`
            <!DOCTYPE html>
            <html>
                <body>
                    <input type="hidden" id="selectedCamera" value="Camera 0">
                    <img id="GameWorld" src="" style="width: 1920px; height: 1080px;">
                    <canvas id="GameWorldOverlay"></canvas>
                    <input type="hidden" id="selectedPixel">
                    <input type="hidden" id="appendPixel">
                    <input type="hidden" id="viewId" value="test-view">
                    <form id="selectPixelForm" hx-post="/harmony/select_pixel"></form>
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
        global.console = { ...console, error: vi.fn(), log: vi.fn(), warn: vi.fn(), time: vi.fn(), timeEnd: vi.fn() };

        // Mock form requestSubmit so the htmx-undefined fallback path works
        const form = document.getElementById('selectPixelForm');
        form.requestSubmit = vi.fn();
        form.getAttribute = vi.fn(() => '/harmony/select_pixel');

        try {
            const evalScript = `
                (function(window, document, console) {
                    ${scriptContent}
                    window.initCanvasEditor = typeof initCanvasEditor !== 'undefined' ? initCanvasEditor : undefined;
                    window.dist = typeof dist !== 'undefined' ? dist : undefined;
                    window.handlePixelSelection = typeof handlePixelSelection !== 'undefined' ? handlePixelSelection : undefined;
                    window.pointInPolygon = typeof pointInPolygon !== 'undefined' ? pointInPolygon : undefined;
                    window.syncCanvasData = typeof syncCanvasData !== 'undefined' ? syncCanvasData : undefined;
                })(window, document, console);
            `;
            const fn = new Function('window', 'document', 'console', evalScript);
            fn(window, document, global.console);
        } catch (e) {
            console.error("Error evaluating script:", e);
        }

        // Set up image clientWidth/clientHeight
        const img = document.getElementById('GameWorld');
        Object.defineProperty(img, 'clientWidth', { value: 1920, configurable: true });
        Object.defineProperty(img, 'clientHeight', { value: 1080, configurable: true });
    });

    // ---- Helper to set up editor + canvasData ----
    function setupEditor() {
        const canvasData = { objects: {}, moveable: [], selection: null };
        const editor = window.initCanvasEditor('GameWorldOverlay', canvasData,
            vi.fn(), vi.fn()
        );
        window.harmonyEditor = editor;
        window.harmonyCanvasData = canvasData;
        return { canvasData, editor };
    }

    function makeClickEvent(x, y, opts = {}) {
        return {
            clientX: x,
            clientY: y,
            shiftKey: opts.shiftKey || false,
            ctrlKey: opts.ctrlKey || false,
            metaKey: opts.metaKey || false
        };
    }

    // ================================================================
    // 1. pointInPolygon correctness
    // ================================================================
    describe('pointInPolygon', () => {
        it('returns true for a point inside a square polygon', () => {
            const square = [[0, 0], [100, 0], [100, 100], [0, 100]];
            expect(window.pointInPolygon([50, 50], square)).toBe(true);
        });

        it('returns false for a point outside a square polygon', () => {
            const square = [[0, 0], [100, 0], [100, 100], [0, 100]];
            expect(window.pointInPolygon([150, 50], square)).toBe(false);
        });

        it('does not crash for a point on the edge', () => {
            const square = [[0, 0], [100, 0], [100, 100], [0, 100]];
            const result = window.pointInPolygon([0, 50], square);
            expect(typeof result).toBe('boolean');
        });
    });

    // ================================================================
    // 2. First cell click sets firstCell
    // ================================================================
    it('first cell click sets firstCell and empty additionalCells', () => {
        setupEditor();
        window.gridPolys = {
            'Camera 0': [{ q: 0, r: 0, poly: [[0, 0], [100, 0], [100, 100], [0, 100]] }]
        };

        window.handlePixelSelection(makeClickEvent(50, 50));

        const sel = window.harmonyCanvasData.selection;
        expect(sel).toBeTruthy();
        expect(sel.firstCell).toEqual({ 'Camera 0': [[0, 0], [100, 0], [100, 100], [0, 100]] });
        expect(sel.additionalCells).toEqual([]);
    });

    // ================================================================
    // 3. Second cell click (no append) sets additionalCells
    // ================================================================
    it('second click without shift sets additionalCells with 1 entry', () => {
        setupEditor();
        window.gridPolys = {
            'Camera 0': [
                { q: 0, r: 0, poly: [[0, 0], [100, 0], [100, 100], [0, 100]] },
                { q: 1, r: 0, poly: [[200, 200], [300, 200], [300, 300], [200, 300]] }
            ]
        };

        // First click — sets firstCell
        window.handlePixelSelection(makeClickEvent(50, 50));
        // Second click on a different hex — no shift
        window.handlePixelSelection(makeClickEvent(250, 250));

        const sel = window.harmonyCanvasData.selection;
        expect(sel.firstCell).toEqual({ 'Camera 0': [[0, 0], [100, 0], [100, 100], [0, 100]] });
        expect(sel.additionalCells).toHaveLength(1);
        expect(sel.additionalCells[0]).toEqual({ 'Camera 0': [[200, 200], [300, 200], [300, 300], [200, 300]] });
    });

    // ================================================================
    // 4. Append mode (shift+click) accumulates additionalCells
    // ================================================================
    it('shift+click appends cells with most recent at index 0', () => {
        setupEditor();
        window.gridPolys = {
            'Camera 0': [
                { q: 0, r: 0, poly: [[0, 0], [100, 0], [100, 100], [0, 100]] },
                { q: 1, r: 0, poly: [[200, 200], [300, 200], [300, 300], [200, 300]] },
                { q: 2, r: 0, poly: [[400, 400], [500, 400], [500, 500], [400, 500]] }
            ]
        };

        // First click — sets firstCell
        window.handlePixelSelection(makeClickEvent(50, 50));
        // Shift+click second hex
        window.handlePixelSelection(makeClickEvent(250, 250, { shiftKey: true }));
        // Shift+click third hex
        window.handlePixelSelection(makeClickEvent(450, 450, { shiftKey: true }));

        const sel = window.harmonyCanvasData.selection;
        expect(sel.firstCell).toEqual({ 'Camera 0': [[0, 0], [100, 0], [100, 100], [0, 100]] });
        expect(sel.additionalCells).toHaveLength(2);
        // Most recent at index 0 (unshift)
        expect(sel.additionalCells[0]).toEqual({ 'Camera 0': [[400, 400], [500, 400], [500, 500], [400, 500]] });
        expect(sel.additionalCells[1]).toEqual({ 'Camera 0': [[200, 200], [300, 200], [300, 300], [200, 300]] });
    });

    // ================================================================
    // 5. updateData preserves client selection
    // ================================================================
    it('updateData preserves existing client selection', () => {
        const { canvasData, editor } = setupEditor();

        const clientSelection = {
            firstCell: { 'Camera 0': [[10, 10], [20, 10], [20, 20], [10, 20]] },
            additionalCells: [{ 'Camera 0': [[30, 30], [40, 30], [40, 40], [30, 40]] }]
        };
        canvasData.selection = clientSelection;

        // Simulate server data arriving with a different (or empty) selection
        editor.updateData({
            objects: { newObj: { 'Camera 0': [[0, 0], [10, 0], [10, 10]] } },
            selection: { firstCell: null, additionalCells: [] }
        });

        // Client selection must be UNCHANGED
        expect(canvasData.selection).toBe(clientSelection);
        expect(canvasData.selection.firstCell).toEqual({ 'Camera 0': [[10, 10], [20, 10], [20, 20], [10, 20]] });
        expect(canvasData.selection.additionalCells).toHaveLength(1);
        // Also verify the new objects were merged
        expect(canvasData.objects.newObj).toBeTruthy();
    });

    // ================================================================
    // 6. Selection performance: must complete in under 50ms
    // ================================================================
    it('handlePixelSelection completes in under 50ms with 2000 hex cells', () => {
        setupEditor();

        // Generate 2000 hex cells spread across a large area
        const hexes = [];
        for (let i = 0; i < 2000; i++) {
            const col = i % 50;
            const row = Math.floor(i / 50);
            const x = col * 40;
            const y = row * 40;
            hexes.push({
                q: col,
                r: row,
                poly: [[x, y], [x + 30, y], [x + 30, y + 30], [x, y + 30]]
            });
        }
        window.gridPolys = { 'Camera 0': hexes };

        // Click on the last hex to force full iteration
        const lastHex = hexes[hexes.length - 1];
        const cx = lastHex.poly[0][0] + 15;
        const cy = lastHex.poly[0][1] + 15;

        const start = performance.now();
        window.handlePixelSelection(makeClickEvent(cx, cy));
        const elapsed = performance.now() - start;

        expect(elapsed).toBeLessThan(50);
        // Also verify it actually selected the right cell
        const sel = window.harmonyCanvasData.selection;
        expect(sel).toBeTruthy();
        expect(sel.firstCell['Camera 0']).toEqual(lastHex.poly);
    });
});
