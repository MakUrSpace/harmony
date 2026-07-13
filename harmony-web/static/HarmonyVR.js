// HarmonyVR.js
// Handles fetching data from the backend and updating the A-Frame VR scene.

const POLL_INTERVAL = 1000;
const HEX_SIZE = 0.5; // Radius of a hex in meters

// Flat top hex to pixel conversion (adjust if pointy top)
// Pointy top: x = size * sqrt(3) * (q + r/2), z = size * 3/2 * r
// Flat top:   x = size * 3/2 * q,             z = size * sqrt(3) * (r + q/2)
// Assuming Pointy top by default for typical grids.
function axialToCartesian(q, r, size) {
    const x = size * Math.sqrt(3) * (q + r / 2);
    const z = size * 1.5 * r;
    return { x, z };
}

let lastObjects = {};
let camerasCreated = false;

async function fetchCanvasData() {
    try {
        const response = await fetch('/harmony/canvas_data/' + VIEW_ID);
        if (!response.ok) return;
        const data = await response.json();
        
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
    
    // Group objects by color
    // Colors: self == purple, allies == green, enemies == red, selectable == gray
    const purpleHex = "#800080";
    const greenHex = "#00FF00";
    const redHex = "#FF0000";
    const grayHex = "#AAAAAA";
    const defaultHex = "#FFFFFF";

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
        
        let color = defaultHex;
        let isSelectable = false;
        
        if (data.selectable && (data.selectable.includes(oid) || data.selectable.includes(obj.name))) {
            color = grayHex;
            isSelectable = true;
        }
        if (data.enemies && (data.enemies.includes(oid) || data.enemies.includes(obj.name))) color = redHex;
        if (data.allies && (data.allies.includes(oid) || data.allies.includes(obj.name))) color = greenHex;
        if (data.moveable && (data.moveable.includes(oid) || data.moveable.includes(obj.name))) color = purpleHex;
        
        // Compute position based on first axial coordinate
        // In a real game, multi-hex objects might be larger, but here we place at the center of the first hex.
        if (!obj.constituent_axials || obj.constituent_axials.length === 0) continue;
        
        const q = obj.constituent_axials[0][0];
        const r = obj.constituent_axials[0][1];
        const pos = axialToCartesian(q, r, HEX_SIZE);
        const height = isSelectable ? 0.25 : 0.5;
        const yPos = height / 2; // offset so bottom is on floor

        if (!el) {
            el = document.createElement('a-entity');
            el.setAttribute('id', 'obj-' + oid);
            
            // Visual representation
            const cylinder = document.createElement('a-cylinder');
            cylinder.setAttribute('segments-radial', '6');
            cylinder.setAttribute('radius', HEX_SIZE * 0.9);
            cylinder.setAttribute('height', height);
            cylinder.setAttribute('color', color);
            cylinder.setAttribute('class', 'clickable'); // Raycaster target
            
            // Text label
            const text = document.createElement('a-text');
            text.setAttribute('value', obj.name || obj.object_type);
            text.setAttribute('position', `0 ${height/2 + 0.2} 0`);
            text.setAttribute('align', 'center');
            text.setAttribute('scale', '0.5 0.5 0.5');
            text.setAttribute('color', 'white');
            
            el.appendChild(cylinder);
            el.appendChild(text);
            
            // Click handler
            el.addEventListener('click', () => {
                console.log(`Clicked object ${oid}`);
                // Could call /harmony/objects/${oid}/select or similar backend API
                fetch('/harmony/select_pixel', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        view_id: VIEW_ID,
                        cam_name: 'VirtualMap',
                        // If we had pixel logic, we'd pass it here, or we can send axial coords if API supports it
                        x: 0, y: 0 
                    })
                });
            });

            virtualMap.appendChild(el);
        } else {
            // Update existing
            const cylinder = el.querySelector('a-cylinder');
            if (cylinder) {
                cylinder.setAttribute('color', color);
                cylinder.setAttribute('height', height);
            }
        }
        
        el.setAttribute('position', `${pos.x} ${yPos} ${pos.z}`);
    }
    
    lastObjects = data.objects;
}

function createCameras(cameras) {
    const assets = document.getElementById('vr-assets');
    const streamsDiv = document.getElementById('mjpeg-streams');
    const container = document.getElementById('camera-windows');
    
    let xOffset = - (cameras.length * 2.0) / 2;

    cameras.forEach((camName, i) => {
        if (camName === "VirtualMap") return;

        const safeName = camName.replace(/[^a-zA-Z0-9_-]/g, '_');
        
        // 1. Create the hidden img tag for MJPEG stream
        const img = document.createElement('img');
        img.id = 'stream-' + safeName;
        img.src = '/video_feed/' + camName;
        img.hidden = true;
        img.crossOrigin = "anonymous";
        streamsDiv.appendChild(img);

        // 2. Create the canvas in assets
        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-' + safeName;
        canvas.width = 1280;
        canvas.height = 720;
        assets.appendChild(canvas);

        // 3. Create the A-Frame plane displaying the canvas texture
        const plane = document.createElement('a-plane');
        plane.setAttribute('position', `${xOffset + i * 2.2} 0 0`);
        plane.setAttribute('width', '2.0');
        plane.setAttribute('height', '1.125'); // 16:9 ratio
        plane.setAttribute('material', `shader: flat; src: #canvas-${safeName}; side: double`);
        plane.setAttribute('mjpeg-texture', `image: #stream-${safeName}; canvas: #canvas-${safeName}; fps: 15`);
        
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
    fetchCanvasData(); // Initial fetch
};
