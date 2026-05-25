/**
 * Initialize the Canvas Editor.
 */
function initCanvasEditor(canvasId, data, onUpdate, onClick, camName) {
    const canvas = document.getElementById(canvasId);
    let imgElem = null;
    if (camName) {
        imgElem = document.getElementById('GameWorld_' + camName);
    } else {
        imgElem = document.getElementById('GameWorld');
    }

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

    function getPoly(objOrMap) {
        if (!objOrMap) return null;
        if (Array.isArray(objOrMap)) return objOrMap;
        const currentCam = camName || document.getElementById('selectedCamera').value;
        return objOrMap[currentCam];
    }



    function drawPoly(poly, scale, fill = true, r = 255, g = 255, b = 0) {
        // Draw edges
        if (poly.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 1)`;
            ctx.lineWidth = 2;
            var transparency = 0.3;
            const currentCam = camName || document.getElementById(`selectedCamera`).value;
            if (currentCam == 'VirtualMap') {
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

    function translatePolyToQuadrant(poly, quadIndex, cName) {
        if (!poly || poly.length === 0) return null;
        const half_w = 960;
        const half_h = 540;
        const offsetX = (quadIndex % 2) * half_w;
        const offsetY = Math.floor(quadIndex / 2) * half_h;
        
        const scaleX = (cName === "VirtualMap") ? (960 / 1200) : 0.5;
        const scaleY = (cName === "VirtualMap") ? (540 / 1200) : 0.5;
        
        return poly.map(pt => {
            const x = Array.isArray(pt) ? pt[0] : pt.x;
            const y = Array.isArray(pt) ? pt[1] : pt.y;
            return {
                x: x * scaleX + offsetX,
                y: y * scaleY + offsetY
            };
        });
    }

    function drawGroup(group, r, g, b, drawnSet) {
        const scale = getScale(canvas, camName);
        const currentCam = camName || document.getElementById('selectedCamera').value;
        group.forEach(name => {
            if (drawnSet && drawnSet.has(name)) return;
            if (data.objects[name]) {
                if (currentCam === "All") {
                    const cams = data.cameras || [];
                    cams.forEach((cName, quadIdx) => {
                        const poly = data.objects[name][cName];
                        if (poly && poly.length > 0) {
                            const translated = translatePolyToQuadrant(poly, quadIdx, cName);
                            drawPoly(translated, scale, true, r, g, b);
                        }
                    });
                } else {
                    const poly = getPoly(data.objects[name]);
                    if (poly) {
                        drawPoly(poly, scale, true, r, g, b);
                    }
                }
                if (drawnSet) drawnSet.add(name);
            }
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scale = getScale(canvas, camName);
        const drawnSet = new Set();


        // selectable (lowest priority, “neutral hint”): RGB(180, 190, 205)
        // terrain (background structure, stone/metal vibe): RGB(120, 125, 135)
        // targetable (actionable, “UI accent” cyan): RGB(0, 200, 255)
        // enemies (danger, strong red): RGB(230, 60, 70)
        // allies (safe/positive, strong green): RGB(0, 210, 120)
        // moveable (interactive edit mode, vivid purple): RGB(170, 80, 255)
        // selection (highest priority, unmistakable gold): RGB(255, 210, 70)

        const showObjectsElem = document.getElementById('showObjectsHarmony');
        const showObjects = showObjectsElem ? showObjectsElem.checked : true;

        if (showObjects) {
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
        }

        if (data.selection) {
            const currentCam = camName || document.getElementById('selectedCamera').value;
            let firstCenters = [];
            let firstCenterSingle = null;

            if (data.selection.firstCell) {
                if (currentCam === "All") {
                    const cams = data.cameras || [];
                    cams.forEach((cName, quadIdx) => {
                        const poly = data.selection.firstCell[cName];
                        if (poly && poly.length > 0) {
                            const translated = translatePolyToQuadrant(poly, quadIdx, cName);
                            drawPoly(translated, scale, true, 255, 210, 70); // Gold for first
                            
                            // Calculate center
                            let cx = 0, cy = 0;
                            translated.forEach(p => {
                                cx += p.x;
                                cy += p.y;
                            });
                            firstCenters[quadIdx] = { x: cx / translated.length, y: cy / translated.length };
                        }
                    });
                } else {
                    const poly = getPoly(data.selection.firstCell);
                    if (poly) {
                        drawPoly(poly, scale, true, 255, 210, 70); // Gold for first
                        let cx = 0, cy = 0;
                        poly.forEach(p => {
                            const pt = Array.isArray(p) ? { x: p[0], y: p[1] } : p;
                            cx += pt.x;
                            cy += pt.y;
                        });
                        firstCenterSingle = { x: cx / poly.length, y: cy / poly.length };
                    }
                }
            }

            if (data.selection.additionalCells) {
                data.selection.additionalCells.forEach(cell => {
                    if (currentCam === "All") {
                        const cams = data.cameras || [];
                        cams.forEach((cName, quadIdx) => {
                            const poly = cell[cName];
                            if (poly && poly.length > 0) {
                                const translated = translatePolyToQuadrant(poly, quadIdx, cName);
                                drawPoly(translated, scale, true, 255, 0, 255); // Magenta for additional
                                
                                const fCenter = firstCenters[quadIdx];
                                if (fCenter) {
                                    let cx = 0, cy = 0;
                                    translated.forEach(p => {
                                        cx += p.x;
                                        cy += p.y;
                                    });
                                    const currentCenter = { x: cx / translated.length, y: cy / translated.length };

                                    ctx.beginPath();
                                    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
                                    ctx.lineWidth = 2;
                                    ctx.setLineDash([5, 5]);
                                    ctx.moveTo(fCenter.x * scale.x, fCenter.y * scale.y);
                                    ctx.lineTo(currentCenter.x * scale.x, currentCenter.y * scale.y);
                                    ctx.stroke();
                                    ctx.setLineDash([]);
                                }
                            }
                        });
                    } else {
                        const poly = getPoly(cell);
                        if (poly) {
                            drawPoly(poly, scale, true, 255, 0, 255); // Magenta for additional
                            if (firstCenterSingle) {
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
                                ctx.moveTo(firstCenterSingle.x * scale.x, firstCenterSingle.y * scale.y);
                                ctx.lineTo(currentCenter.x * scale.x, currentCenter.y * scale.y);
                                ctx.stroke();
                                ctx.setLineDash([]);
                            }
                        }
                    }
                });
            }
        }
    }

    canvas.onmousedown = function (e) {
        const m = getMousePos(canvas, e); // Screen coords
        isClickCandidate = true;
        const scale = getScale(canvas, camName);

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
        const m = getMousePos(canvas, e); // Screen coords
        const scale = getScale(canvas, camName);

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
        },
        getData: function () {
            return data;
        }
    };
}

function getNaturalDims(camName) {
    const currentCam = camName || document.getElementById('selectedCamera').value;
    if (currentCam === "VirtualMap") {
        return { w: 1200, h: 1200 };
    }
    return { w: 1920, h: 1080 };
}

function getScale(canvas, camName) {
    const dims = getNaturalDims(camName);
    // avoid div by zero
    if (dims.w === 0 || dims.h === 0) return { x: 1, y: 1 };
    return {
        x: canvas.width / dims.w,
        y: canvas.height / dims.h
    };
}

function getMousePos(canvas, evt) {
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


// Existing logic adapted for external call
function handlePixelSelection(event, camNameOverride) {
    console.log("Canvas clicked!!", event);
    // Event is the mouse event derived from canvas

    const camName = camNameOverride || document.getElementById('selectedCamera').value;
    let imgElem = null;
    if (camNameOverride) {
        imgElem = document.getElementById('GameWorld_' + camNameOverride);
    } else {
        imgElem = document.getElementById('GameWorld');
    }

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

    var natural_width = 1920
    var natural_height = 1080
    if (camName === "VirtualMap") {
        natural_width = 1200
        natural_height = 1200
    }

    // Safety check for div-by-zero
    if (cw === 0 || ch === 0) return;

    const px = x / cw * natural_width
    const py = y / ch * natural_height

    const image_x = px // (px - 0) * 1
    const image_y = py // (py - 0) * 1

    let finalCamName = camName;
    let final_x = image_x;
    let final_y = image_y;

    if (camName === "All") {
        const cams = (window.harmonyCanvasData && window.harmonyCanvasData.cameras) ? window.harmonyCanvasData.cameras : [];
        if (cams.length > 0) {
            const half_w = 960;
            const half_h = 540;
            const quad_col = Math.floor(image_x / half_w);
            const quad_row = Math.floor(image_y / half_h);
            const quad_idx = quad_row * 2 + quad_col;
            if (quad_idx >= 0 && quad_idx < cams.length) {
                finalCamName = cams[quad_idx];
                if (finalCamName === "VirtualMap") {
                    final_x = (image_x % half_w) * (1200 / 960);
                    final_y = (image_y % half_h) * (1200 / 540);
                } else {
                    final_x = (image_x % half_w) * 2;
                    final_y = (image_y % half_h) * 2;
                }
            } else {
                finalCamName = cams[0];
            }
        }
    }

    const pixelField = document.getElementById(`selectedPixel`);
    const appendPixelField = document.getElementById(`appendPixel`);
    const isAppend = (event.shiftKey || event.ctrlKey || event.metaKey);
    if (appendPixelField) appendPixelField.value = isAppend ? "true" : "false";

    // Client-side optimistic update
    if (window.gridPolys && window.gridPolys[finalCamName]) {
        let clickedHex = null;
        for (let hex of window.gridPolys[finalCamName]) {
            if (pointInPolygon([final_x, final_y], hex.poly)) {
                clickedHex = hex;
                break;
            }
        }
        
        if (clickedHex) {
            let sel = {};
            if (window.harmonyEditor && window.harmonyEditor.getData) {
                sel = window.harmonyEditor.getData().selection || {};
            } else {
                sel = window.harmonyCanvasData.selection || {};
            }
            let polyObj = {};
            // Optimistically populate polygon for ALL cameras using axial coordinates (q, r)
            for (let cName in window.gridPolys) {
                const hexArr = window.gridPolys[cName];
                const matchingHex = hexArr.find(h => h.q === clickedHex.q && h.r === clickedHex.r);
                if (matchingHex) {
                    polyObj[cName] = matchingHex.poly;
                }
            }
            
            if (appendPixelField.value === "true") {
                if (sel.firstCell) {
                    sel.additionalCells = sel.additionalCells || [];
                    // Insert at front to match server's insert(0, ...)
                    sel.additionalCells.unshift(polyObj);
                } else {
                    sel.firstCell = polyObj;
                    sel.additionalCells = [];
                }
            } else {
                sel.firstCell = polyObj;
                sel.additionalCells = [];
            }
            
            if (window.harmonyEditor && window.harmonyEditor.getData) {
                window.harmonyEditor.getData().selection = sel;
            }
            window.harmonyCanvasData.selection = sel;
            if (window.harmonyEditor) window.harmonyEditor.render();
        }
    }

    const selectPixelForm = document.getElementById(`selectPixelForm`);
    const viewIdVal = document.getElementById(`viewId`) ? document.getElementById(`viewId`).value : "";

    // Temporarily set the selectedCamera form field to the clicked camera (fallback only)
    const selectedCameraInput = document.getElementById('selectedCamera');
    
    // Use HTMX api if available, otherwise fallback to requestSubmit
    if (typeof htmx !== 'undefined') {
        queuedClicks++;
        htmx.ajax('POST', selectPixelForm.getAttribute('hx-post'), {
            target: '#interactor',
            source: '#selectPixelForm',
            values: {
                viewId: viewIdVal,
                selectedPixel: JSON.stringify([~~final_x, ~~final_y]),
                selectedCamera: finalCamName,
                appendPixel: isAppend ? "true" : "false"
            }
        });
    } else {
        pixelField.value = JSON.stringify([~~final_x, ~~final_y]);
        selectedCameraInput.value = finalCamName;
        selectPixelForm.requestSubmit();
    }

    // Restore if needed
    if (camNameOverride) {
        selectedCameraInput.value = 'All';
    } else if (camName === 'All') {
        selectedCameraInput.value = 'All';
    }
}

document.addEventListener('htmx:afterRequest', function(evt) {
    if (evt.detail.elt.id === 'selectPixelForm') {
        if (queuedClicks > 0) queuedClicks--;
        if (queuedClicks === 0 && queuedSyncCanvas) {
            queuedSyncCanvas = false;
            syncCanvasData(document.getElementById("viewId") ? document.getElementById("viewId").value : null);
        }
    }
});

function pointInPolygon(point, vs) {
    var x = point[0], y = point[1];
    var inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        var xi = vs[i][0], yi = vs[i][1];
        var xj = vs[j][0], yj = vs[j][1];
        var intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

window.gridPolys = window.gridPolys || {};

function fetchGridPolys(camName) {
    if (!camName) return;
    if (camName === "All") {
        const cams = (window.harmonyCanvasData && window.harmonyCanvasData.cameras) ? window.harmonyCanvasData.cameras : [];
        for (let c of cams) {
            if (c && c !== "No Camera" && !window.gridPolys[c]) {
                fetch(`/harmony/grid_polys/${c}`)
                    .then(r => r.json())
                    .then(data => { window.gridPolys[c] = data; })
                    .catch(e => console.error(e));
            }
        }
        return;
    }
    if (window.gridPolys[camName]) return;
    fetch(`/harmony/grid_polys/${camName}`)
        .then(r => r.json())
        .then(data => {
            window.gridPolys[camName] = data;
        })
        .catch(e => console.error("Failed to fetch grid polys", e));
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
    
    fetchGridPolys(canvasData.cameraName);
}

function gameWorldClick(camNum) {
    const view_id = document.getElementById(`viewId`).value;
    var container = document.querySelector("#GameWorldViewer");
    var header = document.querySelector("#GameWorldHeader");

    const showObjectsElem = document.getElementById('showObjectsHarmony');
    const showObjects = showObjectsElem ? showObjectsElem.checked : true;

    if (camNum === 'All') {
        header.innerText = `Game World View -- All Views`;
        const camField = document.getElementById(`selectedCamera`);
        camField.value = 'All';

        container.innerHTML = `<img id="GameWorld" class="img-responsive border border-3 border-info bg-primary" src="/harmony/camWithChanges/All/${view_id}" style="border-radius: 40px; max-width: 90%;">
                               <canvas id="GameWorldOverlay" style="position:absolute; left:0; top:0; pointer-events:auto; display: block;"></canvas>`;
        if (window.harmonyCanvasData) {
            window.harmonyCanvasData.cameraName = 'All';
            setTimeout(() => {
                window.harmonyEditor = initCanvasEditor("GameWorldOverlay", window.harmonyCanvasData,
                    function (updatedData) { console.log("Data updated", updatedData); },
                    function (clickEvent) { handlePixelSelection(clickEvent); }
                );
            }, 50);
        }
        window.harmonyEditors = []; // clear multi-view mode
    } else {
        if (!document.getElementById('GameWorld') || (window.harmonyCanvasData && window.harmonyCanvasData.cameraName === 'All')) {
            container.innerHTML = `<img id="GameWorld" class="img-responsive border border-3 border-info bg-primary" src="/harmony/camWithChanges/${camNum}/${view_id}" style="border-radius: 40px; max-width: 90%;">
                                   <canvas id="GameWorldOverlay" style="position:absolute; left:0; top:0; pointer-events:auto; display: block;"></canvas>`;
            if (window.harmonyCanvasData) {
                window.harmonyCanvasData.cameraName = camNum;
                fetchGridPolys(camNum);
                setTimeout(() => {
                    window.harmonyEditor = initCanvasEditor("GameWorldOverlay", window.harmonyCanvasData,
                        function (updatedData) { console.log("Data updated", updatedData); },
                        function (clickEvent) { handlePixelSelection(clickEvent); }
                    );
                }, 50);
            }
        }

        var img = document.querySelector("#GameWorld");
        const newSrc = `/harmony/camWithChanges/${camNum}/${view_id}`;
        if (img.src !== newSrc) {
            img.src = newSrc;
        }

        img.decode().then(() => {
            header.innerText = `Game World View -- ${camNum}`;
            const camField = document.getElementById(`selectedCamera`);
            camField.value = camNum;

            if (window.harmonyCanvasData) {
                window.harmonyCanvasData.cameraName = camNum;
                fetchGridPolys(camNum);
            }
            if (window.harmonyEditor) window.harmonyEditor.render();
            window.harmonyEditors = []; // clear multi-view mode
        }).catch((err) => {
            console.error("Image decode error", err);
        });
    }
}

let isSyncingCanvas = false;
let queuedSyncCanvas = false;
let queuedClicks = 0;

function syncCanvasData(viewId) {
    if (!viewId) {
        const val = document.getElementById("viewId");
        if (val) viewId = val.value;
    }
    if (!viewId) return;

    if (isSyncingCanvas || queuedClicks > 0) {
        queuedSyncCanvas = true;
        return; // Prevent request stacking over high-latency connections
    }

    isSyncingCanvas = true;
    fetch(`/harmony/canvas_data/${viewId}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.cameras) {
                if (window.harmonyCanvasData) {
                    window.harmonyCanvasData.cameras = data.cameras;
                    if (window.harmonyCanvasData.cameraName === "All") {
                        fetchGridPolys("All");
                    }
                }
            }
            if (window.harmonyEditor && queuedClicks === 0) {
                window.harmonyEditor.updateData(data);
            }
        })
        .catch(err => console.error("Error syncing canvas data:", err))
        .finally(() => {
            isSyncingCanvas = false;
            if (queuedSyncCanvas) {
                queuedSyncCanvas = false;
                syncCanvasData(viewId);
            }
        });
}
