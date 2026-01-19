
/**
 * Initialize the Canvas Editor.
 * 
 * @param {string} canvasId - The ID of the canvas element.
 * @param {object} data - The data object containing scene information.
 * @param {object} data.objects - Dictionary of objects by name. Each value is a dictionary mapping { cameraName: polygon }.
 * @param {string[]} data.moveable - List of object names this session can edit.
 * @param {string[]} data.selectable - List of object names this session can inspect.
 * @param {string[]} data.terrain - List of object names that are terrain.
 * @param {string[]} data.allies - List of object names (Blue).
 * @param {string[]} data.targetable - List of object names this session can target.
 * @param {string[]} data.enemies - List of object names (Red).
 * @param {string} data.viewId - The current Session/View ID for Selection state.
 * @param {string} data.cameraName - The current Camera/Perspective Name to render.
 * @param {object} data.selected_cells - Dictionary of selected cells keyed by viewId. 
 *                                      Value: { 
 *                                          firstCell: { cameraName: poly }, 
 *                                          secondCell: { cameraName: poly } | null,
 *                                          additionalCells: [ { cameraName: poly }, ... ] 
 *                                      }
 * @param {function} [onUpdate] - Callback when data is modified (e.g. vertex dragged).
 */
function initCanvasEditor(canvasId, data, onUpdate) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error("Canvas not found:", canvasId);
        return;
    }
    const ctx = canvas.getContext('2d');

    // Default cameraName if not provided (safety fallback, modify as needed)
    const currentCam = data.cameraName || "Camera 0";

    // State for dragging
    let isDragging = false;
    let draggedObject = null;
    let draggedPointIndex = -1;
    let dragStartPos = { x: 0, y: 0 };
    const VERTEX_RADIUS = 5;

    // Helper to get mouse position
    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) * (canvas.width / rect.width),
            y: (evt.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    // Helper to check distance
    function dist(p1, p2) {
        if (Array.isArray(p1)) p1 = { x: p1[0], y: p1[1] };
        if (Array.isArray(p2)) p2 = { x: p2[0], y: p2[1] };
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    // Helper to get polygon for current perspective
    function getPoly(objOrMap) {
        if (!objOrMap) return null;
        // Check if it's already a poly (array) or a map
        if (Array.isArray(objOrMap)) {
            // Unlikely given the new requirement, but graceful fallback
            return objOrMap;
        }
        return objOrMap[currentCam];
    }

    // Main Draw Function
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 1. Draw Terrain (Bottom Layer)
        if (data.terrain) {
            data.terrain.forEach(name => {
                drawObject(name, 'rgba(128, 128, 128, 0.5)', 'rgba(100, 100, 100, 0.8)');
            });
        }

        // 2. Draw Allies (Blue)
        if (data.allies) {
            data.allies.forEach(name => {
                drawObject(name, 'rgba(0, 0, 255, 0.3)', 'rgba(0, 0, 255, 0.8)');
            });
        }

        // 3. Draw Enemies (Red)
        if (data.enemies) {
            data.enemies.forEach(name => {
                drawObject(name, 'rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 0, 0.8)');
            });
        }

        // 4. Draw Selected Cells logic
        if (data.viewId && data.selected_cells && data.selected_cells[data.viewId]) {
            const sel = data.selected_cells[data.viewId];

            // First Cell (Green)
            if (sel.firstCell) {
                drawPoly(getPoly(sel.firstCell), 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 0, 1)');
            }
            // Second Cell (Orange)
            if (sel.secondCell) {
                drawPoly(getPoly(sel.secondCell), 'rgba(255, 165, 0, 0.5)', 'rgba(255, 165, 0, 1)');
            }
            // Additional Cells (Orange/Yellow?) - Assuming same style as second, or slightly different
            if (sel.additionalCells && Array.isArray(sel.additionalCells)) {
                sel.additionalCells.forEach(cellObj => {
                    drawPoly(getPoly(cellObj), 'rgba(255, 200, 0, 0.5)', 'rgba(255, 200, 0, 1)');
                });
            }
        }

        // 5. Draw Vertices for Moveable Objects (if interactive)
        if (data.moveable) {
            data.moveable.forEach(name => {
                if (data.objects[name]) {
                    const poly = getPoly(data.objects[name]);
                    if (!poly) return;

                    ctx.fillStyle = 'yellow';
                    poly.forEach(p => {
                        const pt = Array.isArray(p) ? { x: p[0], y: p[1] } : p;
                        ctx.beginPath();
                        ctx.arc(pt.x, pt.y, VERTEX_RADIUS, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                    });
                }
            });
        }
    }

    function drawObject(name, fillColor, strokeColor) {
        const objMap = data.objects[name];
        if (!objMap) return;
        const poly = getPoly(objMap);
        drawPoly(poly, fillColor, strokeColor);
    }

    function drawPoly(poly, fillColor, strokeColor) {
        if (!poly || poly.length === 0) return;
        ctx.beginPath();
        const start = Array.isArray(poly[0]) ? { x: poly[0][0], y: poly[0][1] } : poly[0];
        ctx.moveTo(start.x, start.y);
        for (let i = 1; i < poly.length; i++) {
            const p = Array.isArray(poly[i]) ? { x: poly[i][0], y: poly[i][1] } : poly[i];
            ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = strokeColor;
        ctx.stroke();
    }

    // Interaction Handlers
    canvas.onmousedown = function (e) {
        const m = getMousePos(e);

        // Check for Vertex Dragging (Moveable)
        if (data.moveable) {
            for (let name of data.moveable) {
                const objMap = data.objects[name];
                if (!objMap) continue;
                const poly = getPoly(objMap); // Get poly for CURRENT PERSECTIVE
                if (!poly) continue;

                for (let i = 0; i < poly.length; i++) {
                    const p = Array.isArray(poly[i]) ? { x: poly[i][0], y: poly[i][1] } : poly[i];
                    if (dist(m, p) <= VERTEX_RADIUS * 2) {
                        isDragging = true;
                        draggedObject = name;
                        draggedPointIndex = i;
                        dragStartPos = m;
                        return;
                    }
                }
            }
        }
    };

    canvas.onmousemove = function (e) {
        if (isDragging && draggedObject) {
            const m = getMousePos(e);

            // Update the data
            // We only update the polygon for the CURRENT perspective
            const objMap = data.objects[draggedObject];
            const poly = getPoly(objMap);

            // Handle array vs object format
            if (Array.isArray(poly[draggedPointIndex])) {
                poly[draggedPointIndex] = [m.x, m.y];
            } else {
                poly[draggedPointIndex] = m;
            }

            // Note: This only updates the local JS object for the current camera.
            // Client code using 'onUpdate' is responsible for syncing this back to the server
            // or updating other perspective coordinates if necessary (which is hard in 2D space without 3D model).

            if (onUpdate) onUpdate(data);
            draw();
        }
    };

    canvas.onmouseup = function (e) {
        if (isDragging) {
            isDragging = false;
            draggedObject = null;
            draggedPointIndex = -1;
        }
    };

    canvas.onmouseleave = function (e) {
        isDragging = false;
        draggedObject = null;
        draggedPointIndex = -1;
    };

    // Initial Draw
    draw();

    // Export a render function
    return {
        render: draw,
        updateData: function (newData) {
            Object.assign(data, newData);
            draw();
        }
    };
}
