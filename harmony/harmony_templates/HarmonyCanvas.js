/**
 * Initialize the Canvas Editor.
 */
function initCanvasEditor(canvasId, data, onUpdate, onClick) {
    const canvas = document.getElementById(canvasId);
    const imgElem = document.getElementById('GameWorld');

    if (!canvas || !imgElem) {
        console.error("Canvas or Image not found");
        return;
    }

    const ctx = canvas.getContext('2d');
    let isDragging = false;
    let potentialDrag = false;
    let dragStartPos = { x: 0, y: 0 };
    let isClickCandidate = false;
    let draggedObject = null;
    let draggedPointIndex = -1;
    const VERTEX_RADIUS = 5;
    const DRAG_THRESHOLD = 3;

    // Resize canvas to match image
    function resizeCanvas() {
        canvas.width = imgElem.clientWidth;
        canvas.height = imgElem.clientHeight;
        canvas.style.left = (imgElem.offsetLeft + imgElem.clientLeft) + "px";
        canvas.style.top = (imgElem.offsetTop + imgElem.clientTop) + "px";
        // Re-draw if needed
        if (typeof draw === 'function') draw();
    }

    // Initial resize and listener
    if (imgElem.complete) {
        resizeCanvas();
    } else {
        imgElem.onload = resizeCanvas;
    }
    window.addEventListener('resize', resizeCanvas);
    // Also resize periodically in case of layout shifts
    setInterval(resizeCanvas, 1000);

    function getNaturalDims() {
        const currentCam = document.getElementById('selectedCamera').value;
        if (currentCam === "VirtualMap") {
            return { w: 1200, h: 1200 };
        }
        return { w: 1920, h: 1080 };
    }

    function getScale() {
        const dims = getNaturalDims();
        // avoid div by zero
        if (dims.w === 0 || dims.h === 0) return { x: 1, y: 1 };
        return {
            x: canvas.width / dims.w,
            y: canvas.height / dims.h
        };
    }

    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) * (canvas.width / rect.width),
            y: (evt.clientY - rect.top) * (canvas.height / rect.height),
            originalEvent: evt
        };
    }

    function dist(p1, p2) {
        if (Array.isArray(p1)) p1 = { x: p1[0], y: p1[1] };
        if (Array.isArray(p2)) p2 = { x: p2[0], y: p2[1] };
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    function getPoly(objOrMap) {
        if (!objOrMap) return null;
        if (Array.isArray(objOrMap)) return objOrMap;
        return objOrMap[document.getElementById(`selectedCamera`).value];
    }


    function drawPoly(poly, scale, fill = true, r = 255, g = 255, b = 0) {
        // Draw edges
        if (poly.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 1)`;
            ctx.lineWidth = 2;
            var transparency = 0.3;
            if (document.getElementById(`selectedCamera`).value == 'VirtualMap') {
                transparency = 1;
            }
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${transparency})`;
            const start = Array.isArray(poly[0]) ? { x: poly[0][0], y: poly[0][1] } : poly[0];
            ctx.moveTo(start.x * scale.x, start.y * scale.y);
            for (let i = 1; i < poly.length; i++) {
                const pt = Array.isArray(poly[i]) ? { x: poly[i][0], y: poly[i][1] } : poly[i];
                ctx.lineTo(pt.x * scale.x, pt.y * scale.y);
            }
            ctx.closePath();
            if (fill) ctx.fill();
            ctx.stroke();
        }
    }

    function drawGroup(group, r, g, b, drawnSet) {
        const scale = getScale();
        group.forEach(name => {
            if (drawnSet && drawnSet.has(name)) return;
            if (data.objects[name]) {
                const poly = getPoly(data.objects[name]);
                if (!poly) return;
                drawPoly(poly, scale, true, r, g, b);
                if (drawnSet) drawnSet.add(name);
            }
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scale = getScale();
        const drawnSet = new Set();

        // selectable (lowest priority, “neutral hint”): RGB(180, 190, 205)
        // terrain (background structure, stone/metal vibe): RGB(120, 125, 135)
        // targetable (actionable, “UI accent” cyan): RGB(0, 200, 255)
        // enemies (danger, strong red): RGB(230, 60, 70)
        // allies (safe/positive, strong green): RGB(0, 210, 120)
        // moveable (interactive edit mode, vivid purple): RGB(170, 80, 255)
        // selection (highest priority, unmistakable gold): RGB(255, 210, 70)

        if (data.moveable) {
            drawGroup(data.moveable, 170, 80, 255, drawnSet);
        }
        if (data.allies) {
            drawGroup(data.allies, 0, 210, 120, drawnSet);
        }
        if (data.enemies) {
            drawGroup(data.enemies, 230, 60, 70, drawnSet);
        }
        if (data.targetable) {
            drawGroup(data.targetable, 0, 200, 255, drawnSet);
        }
        if (data.terrain) {
            drawGroup(data.terrain, 210, 105, 30, drawnSet);
        }
        if (data.selectable) {
            drawGroup(data.selectable, 255, 105, 180, drawnSet);
        }
        if (data.selection) {
            let firstCenter = null;
            if (data.selection.firstCell) {
                const poly = getPoly(data.selection.firstCell);
                if (poly) {
                    drawPoly(poly, scale, true, 255, 210, 70); // Gold for first
                    // Calculate center
                    let cx = 0, cy = 0;
                    poly.forEach(p => {
                        const pt = Array.isArray(p) ? { x: p[0], y: p[1] } : p;
                        cx += pt.x;
                        cy += pt.y;
                    });
                    firstCenter = { x: cx / poly.length, y: cy / poly.length };
                }
            }

            if (data.selection.additionalCells) {
                data.selection.additionalCells.forEach(cell => {
                    const poly = getPoly(cell);
                    if (poly) {
                        drawPoly(poly, scale, true, 255, 0, 255); // Magenta for additional

                        // Draw line to first center
                        if (firstCenter) {
                            let cx = 0, cy = 0;
                            poly.forEach(p => {
                                const pt = Array.isArray(p) ? { x: p[0], y: p[1] } : p;
                                cx += pt.x;
                                cy += pt.y;
                            });
                            const currentCenter = { x: cx / poly.length, y: cy / poly.length };

                            ctx.beginPath();
                            ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
                            ctx.lineWidth = 2;
                            ctx.setLineDash([5, 5]);
                            ctx.moveTo(firstCenter.x * scale.x, firstCenter.y * scale.y);
                            ctx.lineTo(currentCenter.x * scale.x, currentCenter.y * scale.y);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }
                    }
                });
            }
        }
    }

    canvas.onmousedown = function (e) {
        const m = getMousePos(e); // Screen coords
        isClickCandidate = true;
        const scale = getScale();

        if (data.moveable) {
            for (let name of data.moveable) {
                const objMap = data.objects[name];
                if (!objMap) continue;
                const poly = getPoly(objMap);
                if (!poly) continue;
                for (let i = 0; i < poly.length; i++) {
                    const p = Array.isArray(poly[i]) ? { x: poly[i][0], y: poly[i][1] } : poly[i];
                    // Convert point to screen space for check
                    const p_screen = { x: p.x * scale.x, y: p.y * scale.y };

                    if (dist(m, p_screen) <= VERTEX_RADIUS * 2) {
                        potentialDrag = true;
                        dragStartPos = m;
                        draggedObject = name;
                        draggedPointIndex = i;
                        return;
                    }
                }
            }
        }
    };

    canvas.onmousemove = function (e) {
        const m = getMousePos(e); // Screen coords
        const scale = getScale();

        if (potentialDrag && !isDragging) {
            if (dist(m, dragStartPos) > DRAG_THRESHOLD) {
                isDragging = true;
                potentialDrag = false;
                isClickCandidate = false;
            }
        }

        if (isDragging && draggedObject) {
            const objMap = data.objects[draggedObject];
            const poly = getPoly(objMap);

            // Convert mouse screen coord back to image coord
            const imagePt = { x: m.x / scale.x, y: m.y / scale.y };

            if (Array.isArray(poly[draggedPointIndex])) {
                poly[draggedPointIndex] = [imagePt.x, imagePt.y];
            } else {
                poly[draggedPointIndex] = imagePt;
            }
            if (onUpdate) onUpdate(data);
            draw();
        }
    };

    canvas.onmouseup = function (e) {
        if (isDragging) {
            isDragging = false;
            draggedObject = null;
            draggedPointIndex = -1;
        } else {
            // Either Click Candidate (no object hit) OR Potential Drag (object hit but not moved)
            if (isClickCandidate || potentialDrag) {
                if (onClick) onClick(e);
            }
        }
        potentialDrag = false;
        isClickCandidate = false;
    };

    canvas.onmouseleave = function () {
        isDragging = false;
        isClickCandidate = false;
    };

    draw();

    return {
        render: draw,
        updateData: function (newData) {
            Object.assign(data, newData);
            draw();
        }
    };
}

// Existing logic adapted for external call
function handlePixelSelection(event) {
    console.log("Canvas clicked!!", event);
    // Event is the mouse event derived from canvas

    // We need to match the original coordinate calculation
    // original: event.x - left (where left is bounds.left)
    // The event passed here is the mouseup event on the canvas.

    const imgElem = document.getElementById(`GameWorld`)
    const bounds = imgElem.getBoundingClientRect();


    // Use clientX/Y for consistent screen coords
    const clientX = event.clientX;
    const clientY = event.clientY;

    const left = bounds.left;
    const top = bounds.top;

    const x = clientX - left;
    const y = clientY - top;

    const cw = imgElem.clientWidth
    const ch = imgElem.clientHeight

    const selectedCamera = document.getElementById(`selectedCamera`).value
    var natural_width = 1920
    var natural_height = 1080
    if (selectedCamera === "VirtualMap") {
        natural_width = 1200
        natural_height = 1200
    }

    // Safety check for div-by-zero
    if (cw === 0 || ch === 0) return;

    const px = x / cw * natural_width
    const py = y / ch * natural_height

    const image_x = px // (px - 0) * 1
    const image_y = py // (py - 0) * 1

    const pixelField = document.getElementById(`selectedPixel`)
    const appendPixelField = document.getElementById(`appendPixel`)

    // Check for modifier keys
    if (event.shiftKey || event.ctrlKey || event.metaKey) {
        appendPixelField.value = "true";
    } else {
        appendPixelField.value = "false";
    }

    const selectPixelForm = document.getElementById(`selectPixelForm`)
    pixelField.value = JSON.stringify([~~image_x, ~~image_y])
    selectPixelForm.requestSubmit()
}

// Initialization function to be called from the HTML
function initHarmonyCanvas() {
    // Initialization logic
    const canvasData = {
        objects: {},
        moveable: [],
        cameraName: document.getElementById('selectedCamera').value
    };

    var editor = null;

    // Initialize once DOM is ready (or script runs)
    editor = initCanvasEditor("GameWorldOverlay", canvasData,
        function (updatedData) { console.log("Data updated", updatedData); },
        function (clickEvent) { handlePixelSelection(clickEvent); }
    );

    // Expose editor for other functions to use
    window.harmonyEditor = editor;

    // Also Expose canvasData if needed, or better, keep it in closure but accessible via editor?
    // The original code accessed 'canvasData' via closure in initCanvasEditor but also outside.
    // 'gameWorldClick' needs to update 'canvasData.cameraName'.
    // So let's attach it to the window or return it.
    window.harmonyCanvasData = canvasData;
}

function gameWorldClick(camNum) {
    const view_id = document.getElementById(`viewId`).value;
    var img = document.querySelector("#GameWorld");

    // Set src first to start loading
    const newSrc = `/harmony/camWithChanges/${camNum}/${view_id}`;
    if (img.src !== newSrc) {
        img.src = newSrc;
    }

    // Use decode() to wait for image frame to be ready
    img.decode().then(() => {
        var header = document.querySelector("#GameWorldHeader");
        header.innerText = `Game World View -- ${camNum}`;

        const camField = document.getElementById(`selectedCamera`);
        camField.value = camNum;

        // Update Editor Context
        if (window.harmonyCanvasData) {
            window.harmonyCanvasData.cameraName = camNum;
        }
        if (window.harmonyEditor) window.harmonyEditor.render();
    }).catch((encodingError) => {
        console.error("Image decode error", encodingError);
    });
}

function syncCanvasData(viewId) {
    // viewId can be passed or retrieved from DOM
    if (!viewId) {
        const val = document.getElementById("viewId");
        if (val) viewId = val.value;
    }

    if (!viewId) return;

    fetch(`/harmony/canvas_data/${viewId}`)
        .then(response => response.json())
        .then(data => {
            if (window.harmonyEditor) {
                window.harmonyEditor.updateData(data);
            }
        })
        .catch(err => console.error("Error syncing canvas data:", err));
}
