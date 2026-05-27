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
        updateData: function (newData, forceSelection) {
            // Client owns selection state — never let server overwrite it
            var clientSelection = data.selection;
            Object.assign(data, newData);
            if (!forceSelection) {
                data.selection = clientSelection;
            }
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
async function handlePixelSelection(event, camNameOverride) {
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
        if (cams.length > 0) {
            const numCams = cams.length;
            const w = event.target.width;
            const h = event.target.height;
            const half_w = w / 2;
            const half_h = h / 2;
            
            let cIndex = -1;
            if (numCams === 2) {
                cIndex = image_x < half_w ? 0 : 1;
            } else if (numCams > 2) {
                if (image_y < half_h) {
                    cIndex = image_x < half_w ? 0 : 1;
                } else {
                    cIndex = image_x < half_w ? 2 : 3;
                }
            }
            
            if (cIndex >= 0 && cIndex < cams.length) {
                finalCamName = cams[cIndex];
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

    // Client-side cell selection — this is the ONLY place selection state is set.
    // The server sync (syncCanvasData) never touches selection.
    console.time('cellSelect');
    try {
        if (!window.gridPolys || !window.gridPolys[finalCamName] || !Array.isArray(window.gridPolys[finalCamName]) || window.gridPolys[finalCamName].length === 0) {
            console.log("Grid polygons not loaded yet. Fetching synchronously for click...");
            try {
                const response = await fetch(`/harmony/grid_polys/${finalCamName}`);
                const data = await response.json();
                window.gridPolys = window.gridPolys || {};
                window.gridPolys[finalCamName] = data;
            } catch (err) {
                console.error("Failed to fetch grid polys during click", err);
            }
        }

        if (window.gridPolys && window.gridPolys[finalCamName] && Array.isArray(window.gridPolys[finalCamName])) {
            let clickedHex = null;
            const polys = window.gridPolys[finalCamName];
            for (let i = 0; i < polys.length; i++) {
                if (polys[i] && polys[i].poly && pointInPolygon([final_x, final_y], polys[i].poly)) {
                    clickedHex = polys[i];
                    break;
                }
            }

            if (clickedHex) {
                // Build the polygon object for all loaded cameras at this axial coord
                let polyObj = {};
                for (let cName of Object.keys(window.gridPolys)) {
                    const hexArr = window.gridPolys[cName];
                    if (Array.isArray(hexArr)) {
                        const match = hexArr.find(h => h.q === clickedHex.q && h.r === clickedHex.r);
                        if (match) polyObj[cName] = match.poly;
                    }
                }
                polyObj._q = clickedHex.q;
                polyObj._r = clickedHex.r;

                // Read current client-owned selection
                let editorData = window.harmonyEditor ? window.harmonyEditor.getData() : window.harmonyCanvasData;
                let sel = editorData.selection || {};

                // Prevent repeatedly selecting the same cell
                let alreadySelected = false;
                if (sel.firstCell && sel.firstCell._q === clickedHex.q && sel.firstCell._r === clickedHex.r) {
                    alreadySelected = true;
                }
                if (sel.additionalCells) {
                    for (let cell of sel.additionalCells) {
                        if (cell._q === clickedHex.q && cell._r === clickedHex.r) {
                            alreadySelected = true;
                            break;
                        }
                    }
                }

                if (alreadySelected) {
                    console.timeEnd('cellSelect');
                    return;
                }

                if (isAppend) {
                    if (sel.firstCell) {
                        let adds = sel.additionalCells ? [...sel.additionalCells] : [];
                        adds.unshift(polyObj);
                        sel = { firstCell: sel.firstCell, additionalCells: adds };
                    } else {
                        sel = { firstCell: polyObj, additionalCells: [] };
                    }
                } else {
                    if (sel.firstCell) {
                        sel = { firstCell: sel.firstCell, additionalCells: [polyObj] };
                    } else {
                        sel = { firstCell: polyObj, additionalCells: [] };
                    }
                }

                // Write selection to both references so draw() sees it immediately
                if (window.harmonyEditor && window.harmonyEditor.getData) {
                    window.harmonyEditor.getData().selection = sel;
                }
                window.harmonyCanvasData.selection = sel;

                // Draw immediately — no server round-trip needed
                if (window.harmonyEditor) window.harmonyEditor.render();
                console.timeEnd('cellSelect');
            } else {
                console.timeEnd('cellSelect');
                console.log('No hex found at click coordinates', final_x, final_y);
                return;
            }
        } else {
            console.timeEnd('cellSelect');
            console.warn('gridPolys not loaded for', finalCamName, '— available:', Object.keys(window.gridPolys || {}));
        }
    } catch (err) {
        console.timeEnd('cellSelect');
        console.error('Client cell selection error:', err);
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

function fetchGridPolys(camName, force = false) {
    if (!camName) return;
    if (camName === "All") {
        const cams = (window.harmonyCanvasData && window.harmonyCanvasData.cameras) ? window.harmonyCanvasData.cameras : [];
        for (let c of cams) {
            if (c && c !== "No Camera" && (force || !window.gridPolys[c] || !Array.isArray(window.gridPolys[c]) || window.gridPolys[c].length === 0)) {
                fetch(`/harmony/grid_polys/${c}`)
                    .then(r => r.json())
                    .then(data => { window.gridPolys[c] = data; })
                    .catch(e => console.error(e));
            }
        }
        return;
    }
    if (!force && window.gridPolys[camName] && Array.isArray(window.gridPolys[camName]) && window.gridPolys[camName].length > 0) return;
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
        return;
    }

    isSyncingCanvas = true;
    fetch(`/harmony/canvas_data/${viewId}`)
        .then(response => response.json())
        .then(serverData => {
            if (serverData && serverData.cameras) {
                if (window.harmonyCanvasData) {
                    window.harmonyCanvasData.cameras = serverData.cameras;
                    
                    let refetchAll = false;
                    if (serverData.grid_cache_key && serverData.grid_cache_key !== window.gridCacheKey) {
                        window.gridCacheKey = serverData.grid_cache_key;
                        refetchAll = true;
                    }
                    
                    if (window.harmonyCanvasData.cameraName === "All") {
                        fetchGridPolys("All", refetchAll);
                    } else if (refetchAll) {
                        fetchGridPolys(window.harmonyCanvasData.cameraName, true);
                    }
                }
            }
            // Strip selection from server data — client owns selection exclusively
            delete serverData.selection;
            if (window.harmonyEditor) {
                window.harmonyEditor.updateData(serverData);
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

window.clearSelectionFrontEnd = function(viewId) {
    var emptySel = { firstCell: null, additionalCells: [] };
    if (window.harmonyEditor && window.harmonyEditor.getData) {
        window.harmonyEditor.getData().selection = emptySel;
    }
    if (window.harmonyCanvasData) {
        window.harmonyCanvasData.selection = emptySel;
    }
    if (window.harmonyEditor) window.harmonyEditor.render();
    
    // Also tell the backend
    if (typeof htmx !== 'undefined') {
        htmx.ajax('GET', `/harmony/clear_pixel/${viewId}`, {target: '#interactor'});
    }
};

window.rotateObjectFrontEnd = function(viewId, oid) {
    if (window.harmonyCanvasData && window.harmonyCanvasData.objects && window.harmonyCanvasData.objects[oid]) {
        let axials = window.harmonyCanvasData.constituent_axials ? window.harmonyCanvasData.constituent_axials[oid] : null;
        if (axials && axials.length > 1) {
            let q0 = axials[0][0], r0 = axials[0][1];
            let newAxials = [[q0, r0]];
            for (let i = 1; i < axials.length; i++) {
                let q = axials[i][0], r = axials[i][1];
                let dq = q - q0, dr = r - r0;
                let new_dq = -dr, new_dr = dq + dr;
                newAxials.push([q0 + new_dq, r0 + new_dr]);
            }
            window.harmonyCanvasData.constituent_axials[oid] = newAxials;
            
            // Now update the objects geometry from gridPolys
            let objMap = window.harmonyCanvasData.objects[oid];
            for (let cName in objMap) {
                if (window.gridPolys && window.gridPolys[cName]) {
                    let newPoly = [];
                    newAxials.forEach(ax => {
                        let hex = window.gridPolys[cName].find(h => h.q === ax[0] && h.r === ax[1]);
                        if (hex && hex.poly) {
                            newPoly.push(...hex.poly);
                        }
                    });
                    if (newPoly.length > 0) {
                        objMap[cName] = newPoly;
                    }
                }
            }
            if (window.harmonyEditor) window.harmonyEditor.render();
        }
        
        // Tell the backend
        if (typeof htmx !== 'undefined') {
            htmx.ajax('POST', `/harmony/rotate_object/${viewId}/${oid}`, {target: '#interactor'});
        }
    }
};
