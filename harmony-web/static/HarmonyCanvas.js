/**
 * Initialize the Canvas Editor.
 */
function initCanvasEditor(canvasId, data, onUpdate, onClick, camName) {
    const canvas = document.getElementById(canvasId);
    let imgElem = null;
    if (camName) {
        let safeName = camName.replace(/[^a-zA-Z0-9_-]/g, '_');
        imgElem = document.getElementById('GameWorld_' + safeName);
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
        if (!canvas || !imgElem) return;
        const nw = imgElem.naturalWidth || getNaturalDims(camName).w;
        const nh = imgElem.naturalHeight || getNaturalDims(camName).h;
        canvas.width = nw;
        canvas.height = nh;
        canvas.style.width = (imgElem.clientWidth || 0) + "px";
        canvas.style.height = (imgElem.clientHeight || 0) + "px";
        canvas.style.left = (imgElem.offsetLeft + imgElem.clientLeft) + "px";
        canvas.style.top = (imgElem.offsetTop + imgElem.clientTop) + "px";
        // Re-draw if needed
        if (typeof draw === 'function') draw();
    }

    // Modern, robust resize tracking
    const resizeObserver = new ResizeObserver(() => {
        resizeCanvas();
    });
    if (imgElem) resizeObserver.observe(imgElem);
    
    // Also bind window resize as a fallback
    window.addEventListener('resize', resizeCanvas);

    function getPoly(objOrMap) {
        if (!objOrMap) return null;
        if (Array.isArray(objOrMap)) return objOrMap;
        const currentCam = camName || document.getElementById('selectedCamera').value;
        return objOrMap[currentCam];
    }



    function getDrawOffset(cName) {
        let editorData = window.harmonyEditor ? window.harmonyEditor.getData() : (window.harmonyCanvasData || (typeof data !== 'undefined' ? data : null));
        let vmr = editorData && editorData.virtual_map_rect ? editorData.virtual_map_rect : [0,0,1200,1200];
        let mappedCam = cName;
        if (mappedCam && mappedCam.startsWith("RTSPCamera")) {
            mappedCam = mappedCam.replace("RTSPCamera", "");
        }
        if (cName === 'VirtualMap') {
            return { x: -vmr[0], y: -vmr[1] };
        } else if (editorData && editorData.camera_rects && editorData.camera_rects[mappedCam]) {
            return { x: -editorData.camera_rects[mappedCam][0], y: -editorData.camera_rects[mappedCam][1] };
        }
        return { x: 0, y: 0 };
    }

    function drawPoly(poly, scale, fill = true, r = 255, g = 255, b = 0, customAlpha = null, strokeAlpha = 1) {
        if (!poly || poly.length === 0) return;

        let isMultiPoly = false;
        if (Array.isArray(poly[0])) {
            if (Array.isArray(poly[0][0]) || (typeof poly[0][0] === 'object' && poly[0][0] !== null && 'x' in poly[0][0])) {
                isMultiPoly = true;
            }
        }
        
        if (isMultiPoly) {
            for (let i = 0; i < poly.length; i++) {
                drawPoly(poly[i], scale, fill, r, g, b, customAlpha, strokeAlpha);
            }
            return;
        }

        const currentCam = camName || (document.getElementById(`selectedCamera`) ? document.getElementById(`selectedCamera`).value : null);
        const offset = getDrawOffset(currentCam);
        let offsetX = offset.x;
        let offsetY = offset.y;

        // Draw edges
        if (poly.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${strokeAlpha})`;
            ctx.lineWidth = 2;
            var transparency = 0.3;
            if (currentCam == 'VirtualMap') {
                transparency = 1;
            }
            if (customAlpha !== null) {
                transparency = customAlpha;
            }
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${transparency})`;
            const p0x = Array.isArray(poly[0]) ? poly[0][0] : poly[0].x;
            const p0y = Array.isArray(poly[0]) ? poly[0][1] : poly[0].y;
            ctx.moveTo((p0x + offsetX) * scale.x, (p0y + offsetY) * scale.y);
            for (let i = 1; i < poly.length; i++) {
                const px = Array.isArray(poly[i]) ? poly[i][0] : poly[i].x;
                const py = Array.isArray(poly[i]) ? poly[i][1] : poly[i].y;
                ctx.lineTo((px + offsetX) * scale.x, (py + offsetY) * scale.y);
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
        if (!canvas || !imgElem) return;
        const nw = imgElem.naturalWidth || getNaturalDims(camName).w;
        const nh = imgElem.naturalHeight || getNaturalDims(camName).h;
        
        // Auto-correct if natural dimensions changed after initial load (avoids ResizeObserver missing updates)
        let expectedLeft = imgElem.offsetLeft + imgElem.clientLeft;
        let expectedTop = imgElem.offsetTop + imgElem.clientTop;
        
        let renderWidth = imgElem.clientWidth;
        let renderHeight = imgElem.clientHeight;
        
        // Handle object-fit: contain letterboxing for All Perspectives view
        if (nw > 0 && nh > 0 && renderWidth > 0 && renderHeight > 0) {
            const imgAspect = nw / nh;
            const containerAspect = renderWidth / renderHeight;
            if (imgAspect > containerAspect) {
                // Constrained by width
                const newHeight = renderWidth / imgAspect;
                expectedTop += (renderHeight - newHeight) / 2;
                renderHeight = newHeight;
            } else {
                // Constrained by height
                const newWidth = renderHeight * imgAspect;
                expectedLeft += (renderWidth - newWidth) / 2;
                renderWidth = newWidth;
            }
        }
        
        const styleLeft = expectedLeft + "px";
        const styleTop = expectedTop + "px";
        const styleWidth = renderWidth + "px";
        const styleHeight = renderHeight + "px";
        
        if (canvas.width !== nw || canvas.height !== nh || 
            canvas.style.width !== styleWidth || 
            canvas.style.height !== styleHeight ||
            canvas.style.left !== styleLeft ||
            canvas.style.top !== styleTop) {
            
            canvas.width = nw;
            canvas.height = nh;
            canvas.style.width = styleWidth;
            canvas.style.height = styleHeight;
            canvas.style.left = styleLeft;
            canvas.style.top = styleTop;
        }
        
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

        const currentCam = camName || (document.getElementById('selectedCamera') ? document.getElementById('selectedCamera').value : null);
        
        if (currentCam === 'VirtualMap' && data.virtual_map_boundary) {
            for (let i = 0; i < data.virtual_map_boundary.length; i++) {
                drawPoly(data.virtual_map_boundary[i], scale, false, 255, 255, 255, null, 1);
            }
        }

        const showGridElem = document.getElementById('showGridHarmony') || document.getElementById('showGridConfig');
        const showGrid = showGridElem ? showGridElem.checked : false;
        if (showGrid && window.gridPolys && window.gridPolys[currentCam]) {
            const polys = window.gridPolys[currentCam];
            for (let i = 0; i < polys.length; i++) {
                if (polys[i] && polys[i].poly) {
                    drawPoly(polys[i].poly, scale, false, 255, 255, 255, null, 0.4); 
                }
            }
        }

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
                        const offset = getDrawOffset(currentCam);
                        firstCenterSingle = { x: (cx / poly.length) + offset.x, y: (cy / poly.length) + offset.y };
                    }
                }
            }

            if (!data.selection.additionalCells || data.selection.additionalCells.length === 0) {
                const infoSpan = document.getElementById("selectionDistance");
                if (infoSpan) infoSpan.innerText = "";
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
                                    
                                    if (cell._q !== undefined && cell._r !== undefined && data.selection.firstCell._q !== undefined && data.selection.firstCell._r !== undefined) {
                                        const dist = hexDistance(data.selection.firstCell._q, data.selection.firstCell._r, cell._q, cell._r);
                                        const midX = (fCenter.x + currentCenter.x) / 2 * scale.x;
                                        const midY = (fCenter.y + currentCenter.y) / 2 * scale.y;
                                        ctx.font = "bold 17px Arial";
                                        ctx.fillStyle = "white";
                                        ctx.strokeStyle = "black";
                                        ctx.lineWidth = 3;
                                        ctx.textAlign = "center";
                                        ctx.textBaseline = "middle";
                                        ctx.strokeText(dist, midX, midY - 15);
                                        ctx.fillText(dist, midX, midY - 15);
                                        // Update distance info span in the UI
                                        const infoSpan = document.getElementById("selectionDistance");
                                        if (infoSpan) {
                                            infoSpan.innerText = `Distance: ${dist} hexes`;
                                        }
                                    }
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
                                const offset = getDrawOffset(currentCam);
                                const currentCenter = { x: (cx / poly.length) + offset.x, y: (cy / poly.length) + offset.y };

                                ctx.beginPath();
                                ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
                                ctx.lineWidth = 2;
                                ctx.setLineDash([5, 5]);
                                ctx.moveTo(firstCenterSingle.x * scale.x, firstCenterSingle.y * scale.y);
                                ctx.lineTo(currentCenter.x * scale.x, currentCenter.y * scale.y);
                                ctx.stroke();
                                ctx.setLineDash([]);

                                if (cell._q !== undefined && cell._r !== undefined && data.selection.firstCell._q !== undefined && data.selection.firstCell._r !== undefined) {
                                    const dist = hexDistance(data.selection.firstCell._q, data.selection.firstCell._r, cell._q, cell._r);
                                    const midX = (firstCenterSingle.x + currentCenter.x) / 2 * scale.x;
                                    const midY = (firstCenterSingle.y + currentCenter.y) / 2 * scale.y;
                                    ctx.font = "bold 17px Arial";
                                    ctx.fillStyle = "white";
                                    ctx.strokeStyle = "black";
                                    ctx.lineWidth = 3;
                                    ctx.textAlign = "center";
                                    ctx.textBaseline = "middle";
                                    ctx.strokeText(dist, midX, midY - 15);
                                    ctx.fillText(dist, midX, midY - 15);
                                    const infoSpan = document.getElementById("selectionInfo");
                                    if (infoSpan) infoSpan.innerText = "Distance: " + dist + " hexes";
                                }
                            }
                        }
                    }
                });
            }
            
            // Render Burst and Cone Tools
            if (data.selection.firstCell && window.toolMode && window.toolMode !== 'none' && firstCenterSingle) {
                const currentCam = camName || (document.getElementById('selectedCamera') ? document.getElementById('selectedCamera').value : null);
                if (currentCam && window.gridPolys && window.gridPolys[currentCam]) {
                    const polys = window.gridPolys[currentCam];
                    const selQ = data.selection.firstCell._q;
                    const selR = data.selection.firstCell._r;
                    
                    if (selQ !== undefined && selR !== undefined) {
                        let primaryDir = [1, 0];
                        let secondaryDir = [0, 1];
                        
                        if (window.toolMode === 'cone') {
                            let hoverHex = null;
                            let minDist = Infinity;
                            for (let i = 0; i < polys.length; i++) {
                                const hex = polys[i];
                                let hx = 0, hy = 0;
                                hex.poly.forEach(p => { hx += p[0]; hy += p[1]; });
                                hx = hx / hex.poly.length * scale.x;
                                hy = hy / hex.poly.length * scale.y;
                                let d = Math.hypot(window.mouseX - hx, window.mouseY - hy);
                                if (d < minDist) {
                                    minDist = d;
                                    hoverHex = hex;
                                }
                            }
                            
                            if (hoverHex) {
                                let dq = hoverHex.q - selQ;
                                let dr = hoverHex.r - selR;
                                if (dq !== 0 || dr !== 0) {
                                    const dirs = [[1,0], [0,1], [-1,1], [-1,0], [0,-1], [1,-1]];
                                    for (let i = 0; i < 6; i++) {
                                        let D1 = dirs[i];
                                        let D2 = dirs[(i+1)%6];
                                        let det = D1[0]*D2[1] - D1[1]*D2[0];
                                        let a = (D2[1]*dq - D2[0]*dr) / det;
                                        let b = (-D1[1]*dq + D1[0]*dr) / det;
                                        a = Math.round(a);
                                        b = Math.round(b);
                                        if (a >= 0 && b >= 0) {
                                            if (a >= b) {
                                                primaryDir = D1;
                                                secondaryDir = D2;
                                            } else {
                                                primaryDir = D2;
                                                secondaryDir = D1;
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        window.currentToolHighlight = [[selQ, selR]];
                        for (let i = 0; i < polys.length; i++) {
                            const hex = polys[i];
                            const dist = hexDistance(selQ, selR, hex.q, hex.r);
                            
                            if (dist === 0) continue; // Skip the selected cell itself
                            
                            let highlight = false;
                            if (window.toolMode === 'burst' && dist <= (window.burstRadius || 2)) {
                                highlight = true;
                            } else if (window.toolMode === 'cone' && dist <= (window.coneLength || 3)) {
                                let hq = hex.q - selQ;
                                let hr = hex.r - selR;
                                let det = primaryDir[0]*secondaryDir[1] - primaryDir[1]*secondaryDir[0];
                                let a = (secondaryDir[1]*hq - secondaryDir[0]*hr) / det;
                                let b = (-primaryDir[1]*hq + primaryDir[0]*hr) / det;
                                a = Math.round(a);
                                b = Math.round(b);
                                if (a >= 1 && b >= 0 && (a + b) <= (window.coneLength || 3)) {
                                    highlight = true;
                                }
                            }
                            
                            if (highlight) {
                                window.currentToolHighlight.push([hex.q, hex.r]);
                                drawPoly(hex.poly, scale, true, 0, 150, 255, 0.5);
                            }
                        }
                    }
                }
            }
        }
        
        if (data.published_selections && data.published_selections.length > 0) {
            const currentCam = camName || (document.getElementById('selectedCamera') ? document.getElementById('selectedCamera').value : null);
            if (currentCam && window.gridPolys && window.gridPolys[currentCam]) {
                const polys = window.gridPolys[currentCam];
                
                data.published_selections.forEach(selList => {
                    selList.forEach(cell => {
                        const cellQ = cell[0];
                        const cellR = cell[1];
                        
                        for (let i = 0; i < polys.length; i++) {
                            const hex = polys[i];
                            if (hex.q === cellQ && hex.r === cellR) {
                                // Goldenrod / Orange overlay for published selections
                                drawPoly(hex.poly, scale, true, 218, 165, 32, 0.5); 
                                break;
                            }
                        }
                    });
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
        window.mouseX = m.x;
        window.mouseY = m.y;
        if (window.toolMode === 'cone') {
            if (window.harmonyEditor) window.harmonyEditor.render();
        }

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
            // Client owns selection state locally to prevent flickering, but we want the server's 
            // fully-populated polygons (e.g. for VirtualMap and other cameras) once it catches up.
            var clientSelection = data.selection;
            Object.assign(data, newData);
            if (!forceSelection) {
                let keepClient = true;
                if (clientSelection && clientSelection.firstCell && data.selection && data.selection.firstCell) {
                    if (clientSelection.firstCell._q === data.selection.firstCell._q && 
                        clientSelection.firstCell._r === data.selection.firstCell._r) {
                        keepClient = false; // Server caught up, use server's fully-populated polygons
                    }
                } else if (!clientSelection || !clientSelection.firstCell) {
                    keepClient = false; // Client has no selection, accept server's
                }

                if (keepClient) {
                    data.selection = clientSelection;
                }
            }
            draw();
        },
        getData: function () {
            return data;
        }
    };
}

function getNaturalDims(camName) {
    const currentCam = camName || (document.getElementById('selectedCamera') ? document.getElementById('selectedCamera').value : null);
    if (currentCam === "VirtualMap") {
        let editorData = window.harmonyEditor ? window.harmonyEditor.getData() : window.harmonyCanvasData;
        let vmr = editorData && editorData.virtual_map_rect ? editorData.virtual_map_rect : [0,0,1200,1200];
        return { w: vmr[2] > 0 ? vmr[2] : 1200, h: vmr[3] > 0 ? vmr[3] : 1200 };
    }
    return { w: 1920, h: 1080 };
}

function getScale(canvas, camName) {
    let imgElem = null;
    if (camName) {
        let safeName = camName.replace(/[^a-zA-Z0-9_-]/g, '_');
        imgElem = document.getElementById('GameWorld_' + safeName);
    }
    if (!imgElem) {
        imgElem = document.getElementById('GameWorld');
    }
    
    let nw = imgElem ? imgElem.naturalWidth : 0;
    let nh = imgElem ? imgElem.naturalHeight : 0;
    
    if (nw === 0 || nh === 0) {
        const dims = getNaturalDims(camName);
        nw = dims.w;
        nh = dims.h;
    }
    
    if (nw === 0 || nh === 0) return { x: 1, y: 1 };
    
    return {
        x: canvas.width / nw,
        y: canvas.height / nh
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
        let safeName = camNameOverride.replace(/[^a-zA-Z0-9_-]/g, '_');
        imgElem = document.getElementById('GameWorld_' + safeName);
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

    var natural_width = imgElem.naturalWidth || 1920;
    var natural_height = imgElem.naturalHeight || 1080;
    
    if (natural_width === 0 || natural_height === 0) {
        if (camName === "VirtualMap") {
            natural_width = 1200;
            natural_height = 1200;
        } else {
            natural_width = 1920;
            natural_height = 1080;
        }
    }

    // Safety check for div-by-zero
    if (cw === 0 || ch === 0) return;

    const px = x / cw * natural_width
    const py = y / ch * natural_height

    const image_x = px // (px - 0) * 1
    const image_y = py // (py - 0) * 1

    let finalCamName = camName;
    if (finalCamName && finalCamName.startsWith("RTSPCamera")) {
        finalCamName = finalCamName.replace("RTSPCamera", "");
    }
    
    let editorData = window.harmonyEditor ? window.harmonyEditor.getData() : window.harmonyCanvasData;
    let vmr = editorData && editorData.virtual_map_rect ? editorData.virtual_map_rect : [0,0,1200,1200];
    
    let final_x = image_x;
    let final_y = image_y;

    if (finalCamName === "VirtualMap") {
        final_x += vmr[0];
        final_y += vmr[1];
    } else if (editorData && editorData.camera_rects && editorData.camera_rects[finalCamName]) {
        final_x += editorData.camera_rects[finalCamName][0];
        final_y += editorData.camera_rects[finalCamName][1];
    }

    if (camName === "All") {
        // Obsolete: All view is now implemented as a CSS grid of individual streams.
        return;
    }

    const pixelField = document.getElementById(`selectedPixel`);
    const appendPixelField = document.getElementById(`appendPixel`);
    const isAppend = (event.shiftKey || event.ctrlKey || event.metaKey);
    if (appendPixelField) appendPixelField.value = isAppend ? "true" : "false";
    if (pixelField) pixelField.value = JSON.stringify([final_x, final_y]);

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

                let isActuallyAppending = false;
                if (isAppend) {
                    if (sel.firstCell) {
                        let adds = sel.additionalCells ? [...sel.additionalCells] : [];
                        adds.unshift(polyObj);
                        sel = { firstCell: sel.firstCell, additionalCells: adds };
                        isActuallyAppending = true;
                    } else {
                        sel = { firstCell: polyObj, additionalCells: [] };
                    }
                } else {
                    if (window.toolMode && window.toolMode !== 'none' && sel.firstCell && (!sel.additionalCells || sel.additionalCells.length === 0)) {
                        sel = { firstCell: sel.firstCell, additionalCells: [polyObj] };
                        isActuallyAppending = true;
                    } else {
                        sel = { firstCell: polyObj, additionalCells: [] };
                    }
                }
                
                if (appendPixelField) {
                    appendPixelField.value = isActuallyAppending ? "true" : "false";
                }

                // Write selection to both references so draw() sees it immediately
                if (window.harmonyEditor && window.harmonyEditor.getData) {
                    window.harmonyEditor.getData().selection = sel;
                }
                window.harmonyCanvasData.selection = sel;
                
                const palette = document.getElementById('selectionPalette');
                if (palette) palette.style.display = 'block';

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
            const values = {
                viewId: viewIdVal,
                selectedPixel: JSON.stringify([~~final_x, ~~final_y]),
                selectedCamera: finalCamName,
                appendPixel: isAppend ? "true" : "false"
            };
            const isAdminInput = document.querySelector('input[name="isAdmin"]');
            if (isAdminInput) {
                values.isAdmin = isAdminInput.value;
            }
            
            htmx.ajax('POST', selectPixelForm.getAttribute('hx-post'), {
                target: '#interactor',
                source: '#selectPixelForm',
                values: values
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

document.addEventListener('click', function(e) {
    if (e.target && e.target.classList && e.target.classList.contains('cam-btn')) {
        gameWorldClick(e.target.getAttribute('data-camera'));
    }
});

function gameWorldClick(camNum) {
    const view_id = document.getElementById(`viewId`).value;
    var container = document.querySelector("#GameWorldViewer");
    var header = document.querySelector("#GameWorldHeader");

    const showObjectsElem = document.getElementById('showObjectsHarmony');
    const showObjects = showObjectsElem ? showObjectsElem.checked : true;

    if (camNum === 'All') {
        const headerText = document.querySelector("#GameWorldHeaderText");
        if (headerText) headerText.innerText = `Game World View -- All Views`;
        const camField = document.getElementById(`selectedCamera`);
        camField.value = 'All';

        let html = `<div style="display: grid; grid-template-columns: 1fr 1fr; grid-auto-rows: minmax(0, 1fr); gap: 10px; width: 100%; height: 70vh; max-height: 70vh;">`;
        const allCams = window.harmonyCanvasData && window.harmonyCanvasData.cameras ? window.harmonyCanvasData.cameras : ["VirtualMap"];
        for(let i=0; i < allCams.length; i++) {
            let cName = allCams[i];
            if (!cName || cName === "No Camera") continue;
            let safeName = cName.replace(/[^a-zA-Z0-9_-]/g, '_');
            let encodedCName = encodeURIComponent(cName);
            html += `<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 20px; overflow: hidden; min-width: 0; min-height: 0;" class="border border-3 border-info">
                        <div style="position: relative; display: flex; align-items: center; justify-content: center; width: 100%; height: 100%;">
                            <img id="GameWorld_${safeName}" class="img-responsive" src="/harmony/camWithChanges/${encodedCName}/${view_id}" style="max-width: 100%; max-height: 100%; width: auto; height: auto; display: block; object-fit: contain;">
                            <canvas id="GameWorldOverlay_${safeName}" style="position:absolute; left:0; top:0; pointer-events:auto; display: block;"></canvas>
                        </div>
                     </div>`;
        }
        html += `</div>`;
        container.innerHTML = html;

        if (window.harmonyCanvasData) {
            window.harmonyCanvasData.cameraName = 'All';
            window.harmonyEditors = [];
            fetchGridPolys("All");
            setTimeout(() => {
                for(let i=0; i < allCams.length; i++) {
                    let cName = allCams[i];
                    if (!cName || cName === "No Camera") continue;
                    let safeName = cName.replace(/[^a-zA-Z0-9_-]/g, '_');
                    let ed = initCanvasEditor(`GameWorldOverlay_${safeName}`, window.harmonyCanvasData, 
                        function(d) { console.log("Data updated", d); },
                        function(ev) { handlePixelSelection(ev, cName); },
                        cName
                    );
                    if (ed) window.harmonyEditors.push(ed);
                }
                if (window.harmonyEditors.length > 0) {
                    window.harmonyEditor = {
                        getData: function() { return window.harmonyCanvasData; },
                        updateData: function(d) { window.harmonyEditors.forEach(e => e.updateData(d)); },
                        render: function() { window.harmonyEditors.forEach(e => e.render()); }
                    };
                }
            }, 100);
        }

    } else {
        let encodedCamNum = encodeURIComponent(camNum);
        if (!document.getElementById('GameWorld') || (window.harmonyCanvasData && window.harmonyCanvasData.cameraName === 'All')) {
            container.innerHTML = `<div style="position: relative; width: 100%; height: auto; max-width: 95vw; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 20px; overflow: hidden;" class="border border-3 border-info mx-auto">
                                       <img id="GameWorld" class="img-responsive" src="/harmony/camWithChanges/${encodedCamNum}/${view_id}" style="width: 100%; height: auto; display: block;">
                                       <canvas id="GameWorldOverlay" style="position:absolute; left:0; top:0; pointer-events:auto; display: block;"></canvas>
                                   </div>`;
            if (window.harmonyCanvasData) {
                window.harmonyCanvasData.cameraName = camNum;
                fetchGridPolys(camNum);
                setTimeout(() => {
                    let ed = initCanvasEditor("GameWorldOverlay", window.harmonyCanvasData,
                        function (updatedData) { console.log("Data updated for", camNum, updatedData); },
                        function (clickEvent) { handlePixelSelection(clickEvent, camNum); },
                        null
                    );
                    window.harmonyEditor = ed;
                }, 50);
            }
        }

        var img = document.querySelector("#GameWorld");
        const newSrc = `/harmony/camWithChanges/${encodedCamNum}/${view_id}`;
        if (img.src !== newSrc) {
            img.src = newSrc;
        }
        const headerText = document.querySelector("#GameWorldHeaderText");
        if (headerText) headerText.innerText = `Game World View -- ${camNum}`;
        const camField = document.getElementById(`selectedCamera`);
        camField.value = camNum;

        if (window.harmonyCanvasData) {
            window.harmonyCanvasData.cameraName = camNum;
            fetchGridPolys(camNum);
        }
        if (window.harmonyEditor) window.harmonyEditor.render();
        window.harmonyEditors = []; // clear multi-view mode
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
    fetch(`/harmony/canvas_data/${viewId}`, { cache: "no-store" })
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
            if (serverData && serverData.selection && window.harmonyCanvasData && window.harmonyCanvasData.selection) {
                // Merge missing camera polygons from server selection into client selection (e.g. VirtualMap)
                let sSel = serverData.selection.firstCell;
                let cSel = window.harmonyCanvasData.selection.firstCell;
                if (sSel && cSel && sSel._q === cSel._q && sSel._r === cSel._r) {
                    for (let key in sSel) {
                        if (cSel[key] === undefined) {
                            cSel[key] = sSel[key];
                        }
                    }
                }
            }
            // Strip selection from server data — client owns selection exclusively
            delete serverData.selection;
            if (serverData && serverData.chat_status !== undefined) {
                window.chatStatus = serverData.chat_status;
                if (typeof window.onChatStatusUpdate === 'function') {
                    window.onChatStatusUpdate(serverData.chat_status);
                }
            }
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

window.toolMode = 'none';
window.burstRadius = 2;
window.coneLength = 3;
window.coneAngle = 60;
window.mouseX = 0;
window.mouseY = 0;

function clearSelection() {
    if (window.harmonyCanvasData) {
        window.harmonyCanvasData.selection = null;
    }
    if (window.harmonyEditor && window.harmonyEditor.getData) {
        window.harmonyEditor.getData().selection = null;
    }
    const interactor = document.getElementById('interactor');
    if (interactor) {
        interactor.innerHTML = '';
    }
    const palette = document.getElementById('selectionPalette');
    if (palette) {
        palette.style.display = 'none';
    }
    const viewId = document.getElementById('viewId') ? document.getElementById('viewId').value : '';
    if (viewId) {
        let fd = new URLSearchParams();
        fd.append('viewId', viewId);
        fetch('/harmony/clear_selection', { method: 'POST', body: fd });
    }
    if (window.harmonyEditor) window.harmonyEditor.render();
    if (window.harmonyEditors) window.harmonyEditors.forEach(ed => ed.render());
}

function setToolMode(mode) {
    window.toolMode = mode;
    if (window.harmonyEditor) window.harmonyEditor.render();
    if (window.harmonyEditors) window.harmonyEditors.forEach(ed => ed.render());
}

function updateToolParams() {
    const br = document.getElementById('burstRadius');
    if (br) window.burstRadius = parseFloat(br.value) || 2;
    
    const cl = document.getElementById('coneLength');
    if (cl) window.coneLength = parseFloat(cl.value) || 3;
    
    const ca = document.getElementById('coneAngle');
    if (ca) window.coneAngle = parseFloat(ca.value) || 60;
    
    if (window.harmonyEditor) window.harmonyEditor.render();
    if (window.harmonyEditors) window.harmonyEditors.forEach(ed => ed.render());
}

function hexDistance(q1, r1, q2, r2) {
    return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}
