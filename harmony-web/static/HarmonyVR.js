// HarmonyVR.js
// Handles fetching data from the backend and updating the A-Frame VR scene.

const POLL_INTERVAL = 1000;
const HEX_SIZE = 0.5; // Radius of a hex in meters

// Pointy top: x = size * sqrt(3) * (q + r/2), z = size * 3/2 * r
function axialToCartesian(q, r, size) {
    const x = size * Math.sqrt(3) * (q + r / 2);
    const z = size * 1.5 * r;
    return { x, z };
}

let lastObjects = {};
let lastData = null;
let camerasCreated = false;

// Overlay Colors
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

// Hexadecimal colors for A-Frame cylinders
const hexColors = {
    selectable: "#B4BECD",
    terrain: "#787D87",
    structures: "#8B4513",
    targetable: "#00C8FF",
    enemies: "#E63C46",
    allies: "#00D278",
    moveable: "#AA50FF",
    selection: "#FFD246",
    default: "#FFFFFF"
};

async function fetchCanvasData() {
    try {
        const response = await fetch('/harmony/canvas_data/' + VIEW_ID);
        if (!response.ok) return;
        const data = await response.json();
        
        lastData = data;
        updateObjects(data);
        if (!camerasCreated && data.cameras && data.cameras.length > 0) {
            createCameras(data.cameras);
            camerasCreated = true;
        }
    } catch (e) {
        console.error("Error fetching canvas data:", e);
    }
}

function updateObjects(data) {
    const scene = document.querySelector('a-scene');
    const virtualMap = document.getElementById('virtual-map');

    const newObjectKeys = new Set(Object.keys(data.objects));
    
    // Remove deleted objects
    for (const oid in lastObjects) {
        if (!newObjectKeys.has(oid)) {
            const el = document.getElementById('obj-' + oid);
            if (el) el.parentNode.removeChild(el);
        }
    }

    // Add or update objects
    for (const oid in data.objects) {
        const obj = data.objects[oid];
        let el = document.getElementById('obj-' + oid);
        
        let color = hexColors.default;
        let isSelectable = false;
        
        // Priority coloring matching HarmonyCanvas.js
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
        
        if (!obj.constituent_axials || obj.constituent_axials.length === 0) continue;
        
        // Render each constituent axial as a separate hex cylinder under the same entity
        // or just render the first one if multi-hex is too complex. We will render all of them.
        
        if (!el) {
            el = document.createElement('a-entity');
            el.setAttribute('id', 'obj-' + oid);
            
            // Text label positioned above the center of the first hex
            const text = document.createElement('a-text');
            text.setAttribute('value', obj.name || obj.object_type);
            text.setAttribute('align', 'center');
            text.setAttribute('scale', '0.5 0.5 0.5');
            text.setAttribute('color', 'white');
            text.setAttribute('class', 'label');
            el.appendChild(text);

            virtualMap.appendChild(el);
        }
        
        // Sync cylinders
        const existingCylinders = el.querySelectorAll('a-cylinder');
        // If the number of hexes changed, it's easier to just recreate them, but let's try to reuse or just wipe and recreate if it mismatches
        if (existingCylinders.length !== obj.constituent_axials.length) {
            existingCylinders.forEach(c => c.parentNode.removeChild(c));
            obj.constituent_axials.forEach((axial, index) => {
                const cylinder = document.createElement('a-cylinder');
                cylinder.setAttribute('segments-radial', '6');
                cylinder.setAttribute('class', 'clickable');
                el.appendChild(cylinder);
            });
        }
        
        const cylinders = el.querySelectorAll('a-cylinder');
        const height = isSelectable ? 0.25 : 0.5;
        const yPos = height / 2;
        
        obj.constituent_axials.forEach((axial, index) => {
            const pos = axialToCartesian(axial[0], axial[1], HEX_SIZE);
            const cyl = cylinders[index];
            if (cyl) {
                cyl.setAttribute('position', `${pos.x} ${yPos} ${pos.z}`);
                cyl.setAttribute('radius', HEX_SIZE * 0.95);
                cyl.setAttribute('height', height);
                cyl.setAttribute('color', color);
                
                if (index === 0) {
                    const text = el.querySelector('.label');
                    if (text) {
                        text.setAttribute('position', `${pos.x} ${height/2 + 0.2} ${pos.z}`);
                    }
                    
                    cyl.onclick = () => {
                        fetch('/harmony/select_pixel', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                view_id: VIEW_ID,
                                cam_name: 'VirtualMap',
                                x: axial[0], y: axial[1] // If API needs axial coords or just select object
                            })
                        });
                    };
                }
            }
        });
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
        
        // Handle multi-poly
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
    
    for (const oid in lastData.objects) {
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
    
    let activeCameras = cameras.filter(c => c !== "VirtualMap");
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
        plane.setAttribute('visible', 'false'); // Hidden until naturalWidth > 0
        
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
