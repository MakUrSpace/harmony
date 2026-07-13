// HarmonyVR.js
// Handles fetching data from the backend and updating the A-Frame VR scene.

const POLL_INTERVAL = 1000;
const HEX_SIZE = 0.5; // Radius of a hex in meters

function axialToCartesian(q, r, size) {
    const x = size * Math.sqrt(3) * (q + r / 2);
    const z = size * 1.5 * r;
    return { x, z };
}

// Point in polygon for pixel coordinates
function pointInPolygon(point, vs) {
    let x = point[0], y = point[1];
    let inside = false;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        let xi = vs[i][0], yi = vs[i][1];
        let xj = vs[j][0], yj = vs[j][1];
        let intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

let lastObjects = {};
let lastData = null;
let camerasCreated = false;
let boardInitialized = false;

// Store DOM elements of the grid hexes { "q,r": cylinderElement }
let boardHexes = {};
let objectLabels = {};

// Overlay Colors (for 2D camera canvases)
const overlayColors = {
    selectable: [180, 190, 205],
    terrain: [120, 125, 135],
    structures: [139, 69, 19], // SaddleBrown
    targetable: [0, 200, 255],
    enemies: [230, 60, 70],
    allies: [0, 210, 120],
    moveable: [170, 80, 255],
    selection: [255, 210, 70],
    default: [255, 255, 255]
};

// Hexadecimal colors for A-Frame cylinders (3D Board)
const hexColors = {
    selectable: "#B4BECD",
    terrain: "#787D87",
    structures: "#8B4513",
    targetable: "#00C8FF",
    enemies: "#E63C46",
    allies: "#00D278",
    moveable: "#AA50FF",
    selection: "#FFD246",
    default: "#2A2A2A" // Dark transparent gray for empty grid hexes
};

async function fetchCanvasData() {
    try {
        const response = await fetch('/harmony/canvas_data/' + VIEW_ID);
        if (!response.ok) return;
        const data = await response.json();
        lastData = data;
        
        if (!boardInitialized) {
            await initializeBoard(data);
            boardInitialized = true;
        }

        updateObjects(data);
        
        if (!camerasCreated && data.cameras && data.cameras.length > 0) {
            createCameras(data.cameras);
            camerasCreated = true;
        }
    } catch (e) {
        console.error("Error fetching canvas data:", e);
    }
}

async function initializeBoard(data) {
    const virtualBoard = document.getElementById('virtual-board');
    if (!data.virtual_map_boundary || data.virtual_map_boundary.length === 0) {
        console.warn("No virtual map boundary available to generate board.");
        return;
    }

    try {
        const res = await fetch('/harmony/grid_polys/VirtualMap');
        const gridPolys = await res.json();
        
        gridPolys.forEach(hex => {
            // Find center of hex polygon
            let cx = 0, cy = 0;
            hex.poly.forEach(p => { cx += p[0]; cy += p[1]; });
            cx /= hex.poly.length;
            cy /= hex.poly.length;
            
            // Check if center falls inside ANY of the virtual_map_boundary polygons
            const isActive = data.virtual_map_boundary.some(boundary => pointInPolygon([cx, cy], boundary));
            
            if (isActive) {
                const q = hex.q;
                const r = hex.r;
                const key = `${q},${r}`;
                const pos = axialToCartesian(q, r, HEX_SIZE);
                
                const cylinder = document.createElement('a-cylinder');
                cylinder.setAttribute('segments-radial', '6');
                cylinder.setAttribute('radius', HEX_SIZE * 0.95);
                // Unscaled HEX_SIZE is 0.5. Scale is 0.05, so hex diameter is 0.05m (2 inches).
                // Physical thickness requested is 6 inches (0.15m). Unscaled thickness = 0.15 / 0.05 = 3.0.
                cylinder.setAttribute('height', '3.0');
                cylinder.setAttribute('color', hexColors.default);
                cylinder.setAttribute('material', 'opacity: 0.8; transparent: true');
                // Cylinder origin is center, so raise by half height to rest on y=0
                cylinder.setAttribute('position', `${pos.x} 1.5 ${pos.z}`);
                cylinder.setAttribute('class', 'clickable'); // Raycaster target
                
                cylinder.onclick = () => {
                    fetch('/harmony/select_pixel', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            view_id: VIEW_ID,
                            cam_name: 'VirtualMap',
                            x: q, y: r
                        })
                    });
                };
                
                virtualBoard.appendChild(cylinder);
                boardHexes[key] = cylinder;
            }
        });
        
    } catch (e) {
        console.error("Failed to initialize grid board", e);
    }
}

function updateObjects(data) {
    const virtualBoard = document.getElementById('virtual-board');
    
    // Build a map of axial -> { oid, obj, color, isSelectable }
    const hexMap = {};
    const newObjectKeys = new Set(Object.keys(data.objects));
    
    for (const oid in data.objects) {
        const obj = data.objects[oid];
        
        let color = "#FFFFFF";
        let isSelectable = false;
        
        if (data.selectable && (data.selectable.includes(oid) || data.selectable.includes(obj.name))) {
            color = hexColors.selectable;
            isSelectable = true;
        }
        if (data.terrain && (data.terrain.includes(oid) || data.terrain.includes(obj.name))) color = hexColors.terrain;
        if (data.structures && (data.structures.includes(oid) || data.structures.includes(obj.name))) color = hexColors.structures;
        if (data.targetable && (data.targetable.includes(oid) || data.targetable.includes(obj.name))) color = hexColors.targetable;
        if (data.enemies && (data.enemies.includes(oid) || data.enemies.includes(obj.name))) color = hexColors.enemies;
        if (data.allies && (data.allies.includes(oid) || data.allies.includes(obj.name))) color = hexColors.allies;
        if (data.moveable && (data.moveable.includes(oid) || data.moveable.includes(obj.name))) color = hexColors.moveable;
        if (data.selection && data.selection.selected_oid === oid) color = hexColors.selection;
        
        const axials = data.constituent_axials && data.constituent_axials[oid] ? data.constituent_axials[oid] : [];
        if (axials.length === 0) continue;
        
        axials.forEach((axial, idx) => {
            const key = `${axial[0]},${axial[1]}`;
            hexMap[key] = { oid, obj, color, isSelectable, isPrimary: (idx === 0) };
        });
    }
    
    // Update board hexes
    for (const key in boardHexes) {
        const cyl = boardHexes[key];
        if (hexMap[key]) {
            const info = hexMap[key];
            const height = info.isSelectable ? 3.5 : 4.0; // Taller than empty hex
            cyl.setAttribute('color', info.color);
            cyl.setAttribute('height', height);
            cyl.setAttribute('position', `${cyl.getAttribute('position').x} ${height / 2} ${cyl.getAttribute('position').z}`);
            cyl.setAttribute('material', 'opacity: 1; transparent: false');
        } else {
            // Empty hex
            cyl.setAttribute('color', hexColors.default);
            cyl.setAttribute('height', '3.0');
            cyl.setAttribute('position', `${cyl.getAttribute('position').x} 1.5 ${cyl.getAttribute('position').z}`);
            cyl.setAttribute('material', 'opacity: 0.8; transparent: true');
        }
    }
    
    // Handle Labels
    for (const oid in objectLabels) {
        if (!newObjectKeys.has(oid)) {
            const lbl = objectLabels[oid];
            if (lbl && lbl.parentNode) lbl.parentNode.removeChild(lbl);
            delete objectLabels[oid];
        }
    }
    
    for (const oid in data.objects) {
        const obj = data.objects[oid];
        const axials = data.constituent_axials && data.constituent_axials[oid] ? data.constituent_axials[oid] : [];
        if (axials.length === 0) continue;
        
        const key = `${axials[0][0]},${axials[0][1]}`;
        const pos = axialToCartesian(axials[0][0], axials[0][1], HEX_SIZE);
        
        // Only show label if the hex is on the board
        if (!boardHexes[key]) continue;
        
        let lbl = objectLabels[oid];
        if (!lbl) {
            lbl = document.createElement('a-text');
            lbl.setAttribute('align', 'center');
            lbl.setAttribute('scale', '10 10 10'); // Scaled up because board is scaled 0.05
            lbl.setAttribute('color', 'white');
            virtualBoard.appendChild(lbl);
            objectLabels[oid] = lbl;
        }
        
        lbl.setAttribute('value', obj.name || obj.object_type || oid);
        const height = hexMap[key] && hexMap[key].isSelectable ? 3.5 : 4.0;
        lbl.setAttribute('position', `${pos.x} ${height + 0.5} ${pos.z}`);
        
        lbl.setAttribute('rotation', '-90 0 0');
    }
    
    lastObjects = data.objects;
}

window.drawCameraOverlay = function(ctx, camName, width, height) {
    if (!lastData || !lastData.objects) return;
    
    const img = document.getElementById('stream-' + camName.replace(/[^a-zA-Z0-9_-]/g, '_'));
    if (!img) return;
    const nw = img.naturalWidth || 1280;
    const nh = img.naturalHeight || 720;
    
    const scaleX = width / nw;
    const scaleY = height / nh;
    
    function drawPoly(poly, r, g, b, alpha=0.3) {
        if (!poly || poly.length === 0) return;
        
        let isMultiPoly = false;
        if (Array.isArray(poly[0])) {
            if (Array.isArray(poly[0][0]) || (typeof poly[0][0] === 'object' && poly[0][0] !== null && 'x' in poly[0][0])) {
                isMultiPoly = true;
            }
        }
        if (isMultiPoly) {
            for (let i = 0; i < poly.length; i++) {
                drawPoly(poly[i], r, g, b, alpha);
            }
            return;
        }

        ctx.beginPath();
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 1.0)`;
        ctx.lineWidth = 2;
        
        const p0x = Array.isArray(poly[0]) ? poly[0][0] : poly[0].x;
        const p0y = Array.isArray(poly[0]) ? poly[0][1] : poly[0].y;
        
        ctx.moveTo(p0x * scaleX, p0y * scaleY);
        for(let i=1; i<poly.length; i++) {
            const px = Array.isArray(poly[i]) ? poly[i][0] : poly[i].x;
            const py = Array.isArray(poly[i]) ? poly[i][1] : poly[i].y;
            ctx.lineTo(px * scaleX, py * scaleY);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }
    
    // Sort object IDs so drawing order is stable to prevent z-fighting / flashing
    const sortedOids = Object.keys(lastData.objects).sort();
    
    for (const oid of sortedOids) {
        const objData = lastData.objects[oid];
        const poly = objData[camName];
        if (!poly) continue;
        
        let c = overlayColors.default;
        if (lastData.selectable && lastData.selectable.includes(oid)) c = overlayColors.selectable;
        if (lastData.terrain && lastData.terrain.includes(oid)) c = overlayColors.terrain;
        if (lastData.structures && lastData.structures.includes(oid)) c = overlayColors.structures;
        if (lastData.targetable && lastData.targetable.includes(oid)) c = overlayColors.targetable;
        if (lastData.enemies && lastData.enemies.includes(oid)) c = overlayColors.enemies;
        if (lastData.allies && lastData.allies.includes(oid)) c = overlayColors.allies;
        if (lastData.moveable && lastData.moveable.includes(oid)) c = overlayColors.moveable;
        if (lastData.selection && lastData.selection.selected_oid === oid) c = overlayColors.selection;
        
        drawPoly(poly, c[0], c[1], c[2]);
    }
}

function createCameras(cameras) {
    const assets = document.getElementById('vr-assets');
    const streamsDiv = document.getElementById('mjpeg-streams');
    const container = document.getElementById('camera-windows');
    
    let activeCameras = cameras.filter(c => c !== "VirtualMap" && c !== "No Camera");
    let xOffset = - (activeCameras.length * 2.0) / 2;

    activeCameras.forEach((camName, i) => {
        const safeName = camName.replace(/[^a-zA-Z0-9_-]/g, '_');
        
        const img = document.createElement('img');
        img.id = 'stream-' + safeName;
        img.src = '/video_feed/' + camName;
        img.hidden = true;
        img.crossOrigin = "anonymous";
        streamsDiv.appendChild(img);

        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-' + safeName;
        canvas.width = 1280;
        canvas.height = 720;
        assets.appendChild(canvas);

        const plane = document.createElement('a-plane');
        plane.setAttribute('position', `${xOffset + i * 2.2} 0 0`);
        plane.setAttribute('width', '2.0');
        plane.setAttribute('height', '1.125'); // 16:9 ratio
        plane.setAttribute('material', `shader: flat; src: #canvas-${safeName}; side: double`);
        plane.setAttribute('mjpeg-texture', `image: #stream-${safeName}; canvas: #canvas-${safeName}; camName: ${camName}; fps: 15`);
        plane.setAttribute('visible', 'false');
        
        const label = document.createElement('a-text');
        label.setAttribute('value', camName);
        label.setAttribute('position', '0 0.65 0');
        label.setAttribute('align', 'center');
        label.setAttribute('scale', '0.5 0.5 0.5');
        
        plane.appendChild(label);
        container.appendChild(plane);
    });
}

window.onload = () => {
    setInterval(fetchCanvasData, POLL_INTERVAL);
    fetchCanvasData();
};
