import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Setup Scene, Camera, Renderer
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050510, 0.005);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(60, 30, 80);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050510, 1);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// Shared Materials
const eventMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.8 });
const memoryMat = new THREE.MeshPhongMaterial({ color: 0x3366ff, transparent: true, opacity: 0.7, shininess: 100 });
const gnnMat = new THREE.MeshPhongMaterial({ color: 0xff3366, transparent: true, opacity: 0.8, emissive: 0xff3366, emissiveIntensity: 0.4 });
const lineMat = new THREE.LineBasicMaterial({ color: 0x444466, transparent: true, opacity: 0.4 });

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(50, 100, 50);
scene.add(dirLight);
const pointLight = new THREE.PointLight(0x00ffcc, 2, 100);
pointLight.position.set(0, -20, 0);
scene.add(pointLight);

// Arrays for animation
const eventParticles = [];
const memoryCubes = [];
const gnnNodes = [];
const flowingParticles = [];

const layerYs = {
    events: -40,
    memory: -20,
    gnn: 0,
    scoring: 20,
    gate: 40
};

// --- 1. Event Stream (y = -40) ---
const eventGroup = new THREE.Group();
eventGroup.position.y = layerYs.events;
scene.add(eventGroup);

const sphereGeom = new THREE.SphereGeometry(0.5, 16, 16);
for (let i = 0; i < 0; i++) {
    // Particles are now generated dynamically based on incoming events
}

// Grid for Event Stream
const gridHelper = new THREE.GridHelper(80, 20, 0x00ffcc, 0x222244);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.2;
eventGroup.add(gridHelper);

// --- 2. Memory & Registry (y = -20) ---
const memoryGroup = new THREE.Group();
memoryGroup.position.y = layerYs.memory;
scene.add(memoryGroup);

const cubeGeom = new THREE.BoxGeometry(2, 2, 2);
for (let i = -3; i <= 3; i++) {
    for (let j = -3; j <= 3; j++) {
        const mesh = new THREE.Mesh(cubeGeom, memoryMat.clone());
        mesh.position.set(i * 6, 0, j * 6);
        mesh.userData = {
            baseY: 0,
            phase: Math.random() * Math.PI * 2,
            title: 'Memory Slot ' + (i*7+j+25),
            type: 'TGNMemory + Hashed Identity',
            desc: 'RNN GRU state for historical entity tracking.'
        };
        memoryGroup.add(mesh);
        memoryCubes.push(mesh);
    }
}

// --- 3. Graph Attention (y = 0) ---
const gnnGroup = new THREE.Group();
gnnGroup.position.y = layerYs.gnn;
scene.add(gnnGroup);

const abstractGnnGroup = new THREE.Group();
const realGnnGroup = new THREE.Group();
realGnnGroup.visible = false;
gnnGroup.add(abstractGnnGroup);
gnnGroup.add(realGnnGroup);

const nodeGeom = new THREE.IcosahedronGeometry(1.2, 1);
const numSources = 15;
const numDests = 8;
const nodeCount = numSources + numDests;

// Create Abstract Source nodes (e.g., Devices/IPs)
for (let i = 0; i < numSources; i++) {
    const mesh = new THREE.Mesh(nodeGeom, gnnMat.clone());
    mesh.position.set(-15 + (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 8, (Math.random() - 0.5) * 30);
    mesh.userData = { rotX: Math.random() * 0.02, rotY: Math.random() * 0.02, title: 'Mock IP 10.0.' + i + '.' + Math.floor(Math.random()*255), type: 'Abstract Source', desc: 'Embedded device initiating access.' };
    abstractGnnGroup.add(mesh);
    gnnNodes.push(mesh);
}

// Create Abstract Dest nodes (e.g., Resources)
for (let i = 0; i < numDests; i++) {
    const mesh = new THREE.Mesh(nodeGeom, gnnMat.clone());
    mesh.position.set(15 + (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 8, (Math.random() - 0.5) * 30);
    mesh.material.color.setHex(0xffaa00);
    mesh.material.emissive.setHex(0xffaa00);
    mesh.userData = { rotX: Math.random() * 0.02, rotY: Math.random() * 0.02, title: 'Mock Resource /api/v1/data_' + i, type: 'Abstract Destination', desc: 'Target zero-trust resource.' };
    abstractGnnGroup.add(mesh);
    gnnNodes.push(mesh);
}

// Abstract GNN Edges
const edgesGeometry = new THREE.BufferGeometry();
const edgesPositions = [];
for (let i = 0; i < numSources; i++) {
    const numConnections = Math.floor(Math.random() * 3) + 1;
    for (let c = 0; c < numConnections; c++) {
        const destIdx = numSources + Math.floor(Math.random() * numDests);
        edgesPositions.push(
            gnnNodes[i].position.x, gnnNodes[i].position.y, gnnNodes[i].position.z,
            gnnNodes[destIdx].position.x, gnnNodes[destIdx].position.y, gnnNodes[destIdx].position.z
        );
    }
}
edgesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgesPositions, 3));
const edgesLines = new THREE.LineSegments(edgesGeometry, lineMat);
abstractGnnGroup.add(edgesLines);

// Real Network Data Structures
let isExactTopology = false;
const realNodesMap = new Map();
const realEdgesSet = new Set();
const realEdgesPositions = [];
const realEdgesGeometry = new THREE.BufferGeometry();
realEdgesGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
const realEdgesLines = new THREE.LineSegments(realEdgesGeometry, lineMat);
realGnnGroup.add(realEdgesLines);

function getOrCreateRealNode(key, isSource) {
    if (realNodesMap.has(key)) return realNodesMap.get(key);
    
    const mesh = new THREE.Mesh(nodeGeom, gnnMat.clone());
    if (isSource) {
        mesh.position.set(-15 + (Math.random() - 0.5) * 15, (Math.random() - 0.5) * 15, (Math.random() - 0.5) * 30);
        mesh.userData = { rotX: Math.random() * 0.02, rotY: Math.random() * 0.02, title: 'Source: ' + key, type: 'Real Source Node', desc: 'Observed Source Entity' };
    } else {
        mesh.position.set(15 + (Math.random() - 0.5) * 15, (Math.random() - 0.5) * 15, (Math.random() - 0.5) * 30);
        mesh.material.color.setHex(0xffaa00);
        mesh.material.emissive.setHex(0xffaa00);
        mesh.userData = { rotX: Math.random() * 0.02, rotY: Math.random() * 0.02, title: 'Dest: ' + key, type: 'Real Destination Node', desc: 'Observed Dest Entity' };
    }
    
    realGnnGroup.add(mesh);
    gnnNodes.push(mesh);
    realNodesMap.set(key, mesh);
    
    const statNodes = document.getElementById('stat-nodes');
    if (statNodes) statNodes.textContent = abstractGnnGroup.visible ? 23 : realNodesMap.size;
    return mesh;
}

function processEventForRealNet(srcKey, dstKey) {
    const srcNode = getOrCreateRealNode(srcKey, true);
    const dstNode = getOrCreateRealNode(dstKey, false);
    
    const edgeKey = srcKey + '|' + dstKey;
    if (!realEdgesSet.has(edgeKey)) {
        realEdgesSet.add(edgeKey);
        realEdgesPositions.push(srcNode.position.x, srcNode.position.y, srcNode.position.z, dstNode.position.x, dstNode.position.y, dstNode.position.z);
        realEdgesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(realEdgesPositions, 3));
        realEdgesGeometry.attributes.position.needsUpdate = true;
    }
}

// --- 4. Scoring Heads (y = 20) ---
const scoringGroup = new THREE.Group();
scoringGroup.position.y = layerYs.scoring;
scene.add(scoringGroup);

// Feature Head
const featureMat = new THREE.MeshPhongMaterial({ color: 0xffaa00, transparent: true, opacity: 0.8, wireframe: true });
const featureHead = new THREE.Mesh(new THREE.TorusGeometry(5, 1, 16, 32), featureMat);
featureHead.position.set(-15, 0, 0);
featureHead.userData = { title: 'Feature Head', type: 'LinkPredictor (MLP)', desc: 'Detects contextual and policy violations.' };
scoringGroup.add(featureHead);

// Structural Head
const structuralMat = new THREE.MeshPhongMaterial({ color: 0xaa00ff, transparent: true, opacity: 0.8, wireframe: true });
const structuralHead = new THREE.Mesh(new THREE.TorusKnotGeometry(4, 0.8, 64, 16), structuralMat);
structuralHead.position.set(15, 0, 0);
structuralHead.userData = { title: 'Structural Head', type: 'Cosine Similarity', desc: 'Detects lateral movement via history compatibility.' };
scoringGroup.add(structuralHead);

// Lines converging to Gate
const headPathsGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-15, layerYs.scoring, 0), new THREE.Vector3(0, layerYs.gate, 0),
    new THREE.Vector3(15, layerYs.scoring, 0), new THREE.Vector3(0, layerYs.gate, 0)
]);
const headPathsLines = new THREE.LineSegments(headPathsGeom, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 }));
scene.add(headPathsLines);

// --- 5. Gate (y = 40) ---
const gateGroup = new THREE.Group();
gateGroup.position.y = layerYs.gate;
scene.add(gateGroup);

const gateMat = new THREE.MeshStandardMaterial({ 
    color: 0xffffff, 
    emissive: 0xffffff, 
    emissiveIntensity: 0.8, 
    transparent: true, 
    opacity: 0.9 
});
const gateGeom = new THREE.OctahedronGeometry(4, 0);
const gateMesh = new THREE.Mesh(gateGeom, gateMat);
gateMesh.userData = { title: 'Anomaly Gate', type: 'Threshold logic', desc: 'Final decision: updates memory if benign, drops if anomaly.' };
gateGroup.add(gateMesh);

const ringGeom = new THREE.RingGeometry(6, 6.2, 32);
const ringMat = new THREE.MeshBasicMaterial({ color: 0xff3366, side: THREE.DoubleSide });
const ringMesh = new THREE.Mesh(ringGeom, ringMat);
ringMesh.rotation.x = Math.PI / 2;
gateGroup.add(ringMesh);

// --- Inter-layer Flowing Particles ---
const flowGroup = new THREE.Group();
scene.add(flowGroup);

const flowGeom = new THREE.SphereGeometry(0.3, 8, 8);
const flowMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });

function spawnFlowParticle() {
    const mesh = new THREE.Mesh(flowGeom, flowMat);
    
    // Choose start and end points based on random layer progression
    const layers = [layerYs.events, layerYs.memory, layerYs.gnn, layerYs.scoring, layerYs.gate];
    const startIndex = Math.floor(Math.random() * (layers.length - 1));
    const endIndex = startIndex + 1;
    
    let startX = (Math.random() - 0.5) * 20;
    let startZ = (Math.random() - 0.5) * 20;
    let endX = (Math.random() - 0.5) * 15;
    let endZ = (Math.random() - 0.5) * 15;

    // Special cases for scoring heads
    if (layers[startIndex] === layerYs.scoring) {
        startX = Math.random() > 0.5 ? -15 : 15;
        startZ = 0;
    }
    if (layers[endIndex] === layerYs.scoring) {
        endX = Math.random() > 0.5 ? -15 : 15;
        endZ = 0;
    }
    // Target gate
    if (layers[endIndex] === layerYs.gate) {
        endX = 0;
        endZ = 0;
    }

    mesh.position.set(startX, layers[startIndex], startZ);
    mesh.userData = {
        target: new THREE.Vector3(endX, layers[endIndex], endZ),
        speed: Math.random() * 0.04 + 0.01
    };
    
    flowGroup.add(mesh);
    flowingParticles.push(mesh);
}

// Window resize handler
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Raycaster & Tooltips Setup ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-2, -2);
const tooltip = document.getElementById('tooltip');
const tooltipTitle = document.getElementById('tooltip-title');
const tooltipType = document.getElementById('tooltip-type');
const tooltipDesc = document.getElementById('tooltip-desc');

window.addEventListener('mousemove', (event) => {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    if (tooltip) {
        tooltip.style.left = (event.clientX + 15) + 'px';
        tooltip.style.top = (event.clientY + 15) + 'px';
    }
});

// Animation Loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    const time = clock.getElapsedTime();
    
    controls.update();
    
    // Animate Events
    eventParticles.forEach((p, i) => {
        p.userData.angle += p.userData.speed;
        p.position.x = Math.cos(p.userData.angle) * p.userData.radius;
        p.position.z = Math.sin(p.userData.angle) * p.userData.radius;
        p.position.y = Math.sin(time * 2 + i) * 2;
    });
    
    // Animate Memory Cubes
    memoryCubes.forEach((c, i) => {
        c.position.y = c.userData.baseY + Math.sin(time * 3 + c.userData.phase) * 1.5;
        c.rotation.x = time * 0.5 + c.userData.phase;
        c.rotation.y = time * 0.3 + c.userData.phase;
        c.material.opacity = 0.5 + Math.sin(time * 2 + i) * 0.4;
    });
    
    // Animate GNN Nodes
    gnnNodes.forEach((n) => {
        n.rotation.x += n.userData.rotX;
        n.rotation.y += n.userData.rotY;
        n.material.emissiveIntensity = 0.2 + Math.sin(time * 4 + n.position.x) * 0.3;
    });
    
    // Animate Scoring Heads
    featureHead.rotation.x = time;
    featureHead.rotation.y = time * 0.5;
    structuralHead.rotation.x = time * -1;
    structuralHead.rotation.z = time * 0.5;
    
    // Animate Gate
    gateMesh.rotation.y = time;
    gateMesh.rotation.z = time * 0.5;
    gateMesh.material.emissiveIntensity = 0.5 + Math.sin(time * 5) * 0.5;
    ringMesh.rotation.z = time * -2;
    
    // Raycasting for Tooltips
    raycaster.setFromCamera(mouse, camera);
    const interactables = [...memoryCubes, ...gnnNodes, featureHead, structuralHead, gateMesh].filter(obj => {
        let p = obj;
        while (p) {
            if (p.visible === false) return false;
            p = p.parent;
        }
        return true;
    });
    const intersects = raycaster.intersectObjects(interactables);
    
    if (intersects.length > 0) {
        const obj = intersects[0].object;
        if (obj.userData.title) {
            tooltipTitle.textContent = obj.userData.title;
            tooltipType.textContent = 'Type: ' + obj.userData.type;
            tooltipDesc.textContent = obj.userData.desc;
            tooltip.classList.remove('hidden');
        }
    } else {
        tooltip.classList.add('hidden');
    }

    // Animate Flowing Particles
    const isLive = document.getElementById('live-mode-toggle')?.checked;
    if (!isLive && Math.random() < 0.25) spawnFlowParticle();
    
    for (let i = flowingParticles.length - 1; i >= 0; i--) {
        const p = flowingParticles[i];
        p.position.lerp(p.userData.target, p.userData.speed);
        
        if (p.position.distanceTo(p.userData.target) < 0.5) {
            if (p.userData.path && p.userData.path.length > 0) {
                p.userData.target = p.userData.path.shift();
            } else {
                flowGroup.remove(p);
                flowingParticles.splice(i, 1);
            }
        }
    }
    
    renderer.render(scene, camera);
}

animate();

// UI Interactivity (Optional but good for polishing)
document.querySelectorAll('#layer-list li').forEach(li => {
    li.addEventListener('click', (e) => {
        document.querySelectorAll('#layer-list li').forEach(el => el.classList.remove('active'));
        e.target.classList.add('active');
        
        const layer = e.target.dataset.layer;
        
        [eventGroup, memoryGroup, gnnGroup, scoringGroup, gateGroup].forEach(g => {
            g.visible = true;
        });

        if (layer !== 'all') {
            eventGroup.visible = layer === 'input';
            memoryGroup.visible = layer === 'memory';
            gnnGroup.visible = layer === 'gnn';
            scoringGroup.visible = layer === 'scoring';
            gateGroup.visible = layer === 'gate';
        }
    });
});

// --- Data Stream & Toggles ---
const liveToggle = document.getElementById('live-mode-toggle');
const modeLabel = document.getElementById('mode-label');
const realNetToggle = document.getElementById('real-net-toggle');
const realNetLabel = document.getElementById('real-net-label');
const statNodes = document.getElementById('stat-nodes');
const statEvents = document.getElementById('stat-events');
const alertBox = document.getElementById('anomaly-alert');
const alertDesc = document.getElementById('anomaly-desc');

let ws = null;
let eventCount = 1024;

if (realNetToggle) {
    realNetToggle.addEventListener('change', (e) => {
        isExactTopology = e.target.checked;
        if (isExactTopology) {
            realNetLabel.textContent = "Exact";
            realNetLabel.className = "status-badge live";
            abstractGnnGroup.visible = false;
            realGnnGroup.visible = true;
            statNodes.textContent = realNodesMap.size;
        } else {
            realNetLabel.textContent = "Abstract";
            realNetLabel.className = "status-badge simulation";
            abstractGnnGroup.visible = true;
            realGnnGroup.visible = false;
            statNodes.textContent = "23";
        }
    });
}

let simInterval = null;

function handleStreamEvent(data) {
    eventCount++;
    if(statEvents) statEvents.textContent = eventCount.toLocaleString();
    
    // Update real system stats if provided by backend (Live Mode)
    if (liveToggle && liveToggle.checked) {
        if (data.cpu_percent !== undefined && statCpu) statCpu.textContent = data.cpu_percent.toFixed(1) + '%';
        if (data.ram_gb !== undefined && statRam) statRam.textContent = data.ram_gb.toFixed(2) + ' GB';
        if (data.inf_time_ms !== undefined && statInf) statInf.textContent = data.inf_time_ms.toFixed(1) + 'ms';
    }
    
    processEventForRealNet(data.key_src, data.key_dst);
    
    // Spawn rotating event particle to reflect real event
    const evParticleColor = data.is_anomaly ? 0xff3366 : 0x00ffcc;
    const evParticle = new THREE.Mesh(sphereGeom, new THREE.MeshBasicMaterial({ color: evParticleColor, transparent: true, opacity: 0.8 }));
    const evAngle = Math.random() * Math.PI * 2;
    const evRadius = Math.random() * 25 + 5;
    evParticle.position.set(Math.cos(evAngle) * evRadius, 0, Math.sin(evAngle) * evRadius);
    evParticle.userData = {
        speed: Math.random() * 0.03 + 0.01,
        angle: evAngle,
        radius: evRadius
    };
    eventGroup.add(evParticle);
    eventParticles.push(evParticle);
    
    if (eventParticles.length > 100) {
        const old = eventParticles.shift();
        eventGroup.remove(old);
        if (old.material) old.material.dispose();
    }

    // Determine path through the network
    let srcPos, dstPos;
    const isLiveMode = document.getElementById('live-mode-toggle')?.checked;
    const isExact = isLiveMode || isExactTopology;
    
    if (isExact) {
        const srcNode = getOrCreateRealNode(data.key_src, true);
        const dstNode = getOrCreateRealNode(data.key_dst, false);
        srcPos = srcNode.position;
        dstPos = dstNode.position;
    } else {
        const abstractSources = gnnNodes.filter(n => n.userData.type === 'Abstract Source');
        const abstractDests = gnnNodes.filter(n => n.userData.type === 'Abstract Destination');
        const srcNode = abstractSources.length > 0 ? abstractSources[Math.floor(Math.random() * abstractSources.length)] : { position: new THREE.Vector3(-10, 0, 0) };
        const dstNode = abstractDests.length > 0 ? abstractDests[Math.floor(Math.random() * abstractDests.length)] : { position: new THREE.Vector3(10, 0, 0) };
        srcPos = srcNode.position;
        dstPos = dstNode.position;
    }

    const memCube = memoryCubes[Math.floor(Math.random() * memoryCubes.length)];
    const path = [
        new THREE.Vector3(memCube.position.x, memCube.position.y + layerYs.memory, memCube.position.z),
        new THREE.Vector3(srcPos.x, srcPos.y + layerYs.gnn, srcPos.z),
        new THREE.Vector3(dstPos.x, dstPos.y + layerYs.gnn, dstPos.z),
        new THREE.Vector3(Math.random() > 0.5 ? -15 : 15, layerYs.scoring, 0),
        new THREE.Vector3(0, layerYs.gate, 0)
    ];

    // Spawn flowing particle following the path
    const mesh = new THREE.Mesh(flowGeom, new THREE.MeshBasicMaterial({ color: evParticleColor, transparent: true, opacity: 0.9 }));
    mesh.position.set(evParticle.position.x, layerYs.events, evParticle.position.z);
    mesh.userData = {
        target: path.shift(),
        path: path,
        speed: 0.15
    };
    flowGroup.add(mesh);
    flowingParticles.push(mesh);
    
    if (data.is_anomaly && alertBox) {
        alertDesc.textContent = `Alert on ${data.key_src} -> ${data.key_dst} (Score: ${data.score.toFixed(3)})`;
        alertBox.classList.remove('hidden');
        setTimeout(() => alertBox.classList.add('hidden'), 3000);
        
        gateMesh.material.color.setHex(0xff3366);
        gateMesh.material.emissive.setHex(0xff3366);
        setTimeout(() => {
            gateMesh.material.color.setHex(0xffffff);
            gateMesh.material.emissive.setHex(0xffffff);
        }, 500);
    }
}

function startSimulation() {
    const specsDiv = document.getElementById('sys-specs-static');
    if (specsDiv && (!liveToggle || !liveToggle.checked)) {
        specsDiv.innerHTML = `<strong>Host:</strong> Simulated Environment<br><strong>Arch:</strong> Browser Runtime<br><strong>Acc:</strong> None (Mocked)`;
    }

    if (simInterval) clearInterval(simInterval);
    simInterval = setInterval(() => {
        const fakeSrc = '192.168.1.' + Math.floor(Math.random() * 50);
        const fakeDst = '/api/v1/resource_' + Math.floor(Math.random() * 8);
        const isAnomaly = Math.random() > 0.95;
        const score = isAnomaly ? 0.85 + Math.random() * 0.15 : Math.random() * 0.3;
        handleStreamEvent({ is_anomaly: isAnomaly, key_src: fakeSrc, key_dst: fakeDst, score: score });
    }, 800);
}

function stopSimulation() {
    if (simInterval) clearInterval(simInterval);
}

function clearGraphState() {
    // Rimuove i pacchetti evento
    eventParticles.forEach(p => {
        eventGroup.remove(p);
        if (p.material) p.material.dispose();
    });
    eventParticles.length = 0;
    
    // Rimuove i pacchi flusso
    flowingParticles.forEach(p => {
        flowGroup.remove(p);
        if (p.material) p.material.dispose();
    });
    flowingParticles.length = 0;
    
    // Rimuove i nodi reali
    realNodesMap.forEach(mesh => {
        realGnnGroup.remove(mesh);
        const idx = gnnNodes.indexOf(mesh);
        if (idx !== -1) gnnNodes.splice(idx, 1);
    });
    realNodesMap.clear();
    
    // Resetta gli archi reali
    realEdgesSet.clear();
    realEdgesPositions.length = 0;
    realEdgesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(realEdgesPositions, 3));
    realEdgesGeometry.attributes.position.needsUpdate = true;
    
    eventCount = 0;
    if (statEvents) statEvents.textContent = '0';
    if (statNodes) statNodes.textContent = abstractGnnGroup.visible ? 23 : 0;
}

if (liveToggle) {
    liveToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            modeLabel.textContent = "Live Stream";
            modeLabel.className = "status-badge live";
            
            if (realNetToggle && !realNetToggle.checked) {
                realNetToggle.checked = true;
                realNetToggle.dispatchEvent(new Event('change'));
            }
            
            stopSimulation();
            clearGraphState();
            connectWebSocket();
        } else {
            modeLabel.textContent = "Simulation";
            modeLabel.className = "status-badge simulation";
            if (ws) {
                ws.close();
                ws = null;
            }
            clearGraphState();
            startSimulation();
        }
    });
}

function connectWebSocket() {
    const wsHost = window.location.hostname || 'localhost';
    ws = new WebSocket(`ws://${wsHost}:8888/stream`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.action === "init_sys_info") {
            const specsDiv = document.getElementById('sys-specs-static');
            if (specsDiv) {
                specsDiv.innerHTML = `<strong>Host:</strong> ${data.host}<br><strong>Arch:</strong> ${data.arch}<br><strong>Acc:</strong> ${data.device}`;
            }
            return;
        } else if (data.action === "periodic_stats") {
            if (liveToggle && liveToggle.checked) {
                if (data.cpu_percent !== undefined && statCpu) statCpu.textContent = data.cpu_percent.toFixed(1) + '%';
                if (data.ram_gb !== undefined && statRam) statRam.textContent = data.ram_gb.toFixed(2) + ' GB';
            }
            return;
        }
        
        handleStreamEvent(data);
    };
    ws.onerror = () => {
        console.error("Live Mode: Could not connect to backend.");
        if (liveToggle) liveToggle.checked = false;
        if (modeLabel) {
            modeLabel.textContent = "Simulation";
            modeLabel.className = "status-badge simulation";
        }
        startSimulation();
    }
}

startSimulation();

// --- System Performance Simulation ---
const statCpu = document.getElementById('stat-cpu');
const statRam = document.getElementById('stat-ram');
const statInf = document.getElementById('stat-inf');
setInterval(() => {
    if (liveToggle && liveToggle.checked) return; // Yield to real backend data in Live Mode
    if(statCpu) statCpu.textContent = (10 + Math.random() * 5).toFixed(1) + '%';
    if(statRam) statRam.textContent = (4.1 + Math.random() * 0.3).toFixed(2) + ' GB';
    if(statInf) statInf.textContent = (10 + Math.random() * 4).toFixed(1) + 'ms';
}, 2000);
