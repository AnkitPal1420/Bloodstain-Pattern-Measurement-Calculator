/**
 * trajectory_3d.js
 * ================
 * Interactive 3D Laboratory Trajectory Scene & 2D Convergence Inspector (Phase 4.2).
 *
 * Provides hardware-accelerated HTML5 Canvas 3D projection with:
 * - Orbit rotation (click & drag)
 * - Zoom (scroll wheel / buttons) and Pan (Shift + drag)
 * - Receiving surface rendering (Horizontal Floor, Vertical Wall, Inclined Surface)
 * - Individual trajectory visibility toggles
 * - Interactive stain selection with live coordinate & angle telemetry readout
 * - Camera view presets (Isometric, Top-Down XY, Front XZ, Side YZ) & Reset
 */

document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('trajectory3DCanvas');
    const sceneScript = document.getElementById('trajectorySceneData');
    if (!canvas || !sceneScript) return;

    let sceneData = null;
    try {
        sceneData = JSON.parse(sceneScript.textContent);
    } catch (err) {
        console.error('Failed to parse trajectory scene JSON:', err);
        return;
    }

    if (!sceneData || !sceneData.stains || sceneData.stains.length === 0) return;

    const ctx = canvas.getContext('2d');

    // Camera & Viewport State
    const defaultCamera = {
        yaw: -0.78,      // radians
        pitch: 0.46,     // radians
        zoom: 1.0,
        panX: 0,
        panY: 0
    };
    let cam = { ...defaultCamera };

    // Visibility & Selection State
    let showFloor = true;
    let showWall = true;
    let showUncertainty = true;
    let visibleStains = {};
    sceneData.stains.forEach(s => {
        visibleStains[s.stain_id] = true;
    });
    let selectedStainId = sceneData.stains[0].stain_id;

    // Compute scene bounding box and center in 3D Lab coordinates
    const allPts = [
        sceneData.convergence_2d.point_lab_mm,
        sceneData.tangent_origin_3d.point_lab_mm,
        sceneData.cpa_origin_3d.point_lab_mm
    ];
    sceneData.stains.forEach(s => {
        allPts.push(s.ray_start_lab_mm);
        allPts.push(s.ray_end_lab_mm);
    });

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = 0, maxZ = -Infinity;

    allPts.forEach(p => {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
        if (p[2] < minZ) minZ = p[2];
        if (p[2] > maxZ) maxZ = p[2];
    });

    const spanX = Math.max(maxX - minX, 150);
    const spanY = Math.max(maxY - minY, 150);
    const spanZ = Math.max(maxZ - minZ, 150);
    const maxSpan = Math.max(spanX, spanY, spanZ);

    const center3D = [
        (minX + maxX) / 2.0,
        (minY + maxY) / 2.0,
        (minZ + maxZ) / 2.0
    ];

    // Projected screen coordinates cache for click-picking stains
    let projectedStainHits = [];

    /**
     * Project a 3D Lab point [X, Y, Z] (where +Z is vertical up) onto 2D Canvas [sx, sy, depth]
     */
    function project3D(pt) {
        const w = canvas.width;
        const h = canvas.height;
        const scale = (Math.min(w, h) * 0.62 / maxSpan) * cam.zoom;

        const dx = pt[0] - center3D[0];
        const dy = pt[1] - center3D[1];
        const dz = pt[2] - center3D[2];

        // Rotate around Z-axis by yaw
        const cosY = Math.cos(cam.yaw);
        const sinY = Math.sin(cam.yaw);
        const x1 = dx * cosY - dy * sinY;
        const y1 = dx * sinY + dy * cosY;
        const z1 = dz;

        // Rotate around X-axis by pitch (keeping +Z pointing up on canvas)
        const cosP = Math.cos(cam.pitch);
        const sinP = Math.sin(cam.pitch);
        const x2 = x1;
        const y2 = y1 * sinP - z1 * cosP; // screen vertical (-Y is up)
        const depth = y1 * cosP + z1 * sinP;

        const sx = (w / 2.0) + x2 * scale + cam.panX;
        const sy = (h / 2.0) + y2 * scale + cam.panY;

        return { x: sx, y: sy, depth: depth };
    }

    function drawLine3D(p1, p2, color, width = 1.5, dash = []) {
        const a = project3D(p1);
        const b = project3D(p2);
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash(dash);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
    }

    function drawPolygon3D(pts, fillColor, strokeColor) {
        if (pts.length < 3) return;
        const proj = pts.map(p => project3D(p));
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(proj[0].x, proj[0].y);
        for (let i = 1; i < proj.length; i++) {
            ctx.lineTo(proj[i].x, proj[i].y);
        }
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
    }

    function updateInspectorPanel(stainId) {
        const st = sceneData.stains.find(s => s.stain_id === stainId);
        if (!st) return;

        const setEl = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setEl('insp-stain-id', st.stain_id);
        setEl('insp-dimensions', `${st.width_mm.toFixed(2)} mm × ${st.length_mm.toFixed(2)} mm (W/L = ${st.ratio.toFixed(4)})`);
        setEl('insp-angle', `${st.angle_degrees.toFixed(2)}° ± ${st.sigma_angle_degrees.toFixed(2)}°`);
        setEl('insp-azimuth', `γ = ${st.gamma_deg.toFixed(1)}° (Back-Ray β = ${st.reverse_azimuth_deg.toFixed(1)}°)`);
        setEl('insp-surface-pos', `(${st.surface_x_mm.toFixed(1)}, ${st.surface_y_mm.toFixed(1)}) mm`);
        setEl('insp-lab-pos', `(${st.ray_start_lab_mm[0].toFixed(1)}, ${st.ray_start_lab_mm[1].toFixed(1)}, ${st.ray_start_lab_mm[2].toFixed(1)}) mm`);
        setEl('insp-distance', `${st.planar_distance_mm.toFixed(1)} mm`);
        setEl('insp-height', `${st.tangent_height_z_mm.toFixed(1)} ± ${st.sigma_z_mm.toFixed(1)} mm`);
        setEl('insp-residual', `${st.perp_residual_2d_mm.toFixed(2)} mm`);

        // Highlight active row in stain selector list
        document.querySelectorAll('.stain-select-btn').forEach(btn => {
            if (btn.getAttribute('data-stain-id') === stainId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    function renderScene() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Background gradient
        const grad = ctx.createRadialGradient(w / 2, h / 2, 30, w / 2, h / 2, Math.max(w, h) * 0.7);
        grad.addColorStop(0, '#131d33');
        grad.addColorStop(1, '#090d16');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        const pad = maxSpan * 0.15;
        const x0 = minX - pad, x1 = maxX + pad;
        const y0 = minY - pad, y1 = maxY + pad;
        const z0 = 0, z1 = Math.max(maxZ + pad, 200);

        // 1. Draw Horizontal Floor Plane & Grid (Z = 0)
        if (showFloor) {
            drawPolygon3D(
                [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]],
                'rgba(30, 41, 59, 0.38)',
                'rgba(56, 189, 248, 0.35)'
            );
            const steps = 6;
            for (let i = 1; i < steps; i++) {
                const gx = x0 + (x1 - x0) * (i / steps);
                const gy = y0 + (y1 - y0) * (i / steps);
                drawLine3D([gx, y0, 0], [gx, y1, 0], 'rgba(51, 68, 105, 0.35)', 1);
                drawLine3D([x0, gy, 0], [x1, gy, 0], 'rgba(51, 68, 105, 0.35)', 1);
            }
        }

        // 2. Draw Vertical Reference Wall Plane (Y = y0)
        if (showWall) {
            drawPolygon3D(
                [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                'rgba(15, 23, 42, 0.32)',
                'rgba(148, 163, 184, 0.25)'
            );
        }

        // 3. Draw 3D Coordinate Axes (X = Red, Y = Emerald, Z = Cyan)
        const axisOrigin = [x0, y0, 0];
        const axisLen = maxSpan * 0.32;
        drawLine3D(axisOrigin, [axisOrigin[0] + axisLen, axisOrigin[1], 0], '#f43f5e', 2.2);
        drawLine3D(axisOrigin, [axisOrigin[0], axisOrigin[1] + axisLen, 0], '#10b981', 2.2);
        drawLine3D(axisOrigin, [axisOrigin[0], axisOrigin[1], axisLen], '#38bdf8', 2.2);

        const lblX = project3D([axisOrigin[0] + axisLen + 15, axisOrigin[1], 0]);
        const lblY = project3D([axisOrigin[0], axisOrigin[1] + axisLen + 15, 0]);
        const lblZ = project3D([axisOrigin[0], axisOrigin[1], axisLen + 15]);
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.fillStyle = '#f43f5e'; ctx.fillText('+X (mm)', lblX.x, lblX.y);
        ctx.fillStyle = '#10b981'; ctx.fillText('+Y (mm)', lblY.x, lblY.y);
        ctx.fillStyle = '#38bdf8'; ctx.fillText('+Z (mm)', lblZ.x, lblZ.y);

        // 4. Draw Surface Normal Projection Line from 2D Convergence to Tangent 3D Origin
        const convLab = sceneData.convergence_2d.point_lab_mm;
        const tanLab = sceneData.tangent_origin_3d.point_lab_mm;
        const cpaLab = sceneData.cpa_origin_3d.point_lab_mm;

        drawLine3D(convLab, tanLab, 'rgba(245, 158, 11, 0.7)', 1.6, [5, 5]);

        // 5. Draw Individual Stain Trajectories & Markers
        projectedStainHits = [];
        sceneData.stains.forEach(st => {
            if (!visibleStains[st.stain_id]) return;

            const isSelected = (st.stain_id === selectedStainId);
            const rayWidth = isSelected ? 3.2 : 1.8;
            const rayColor = st.color;

            drawLine3D(st.ray_start_lab_mm, st.ray_end_lab_mm, rayColor, rayWidth);

            const pStart = project3D(st.ray_start_lab_mm);
            projectedStainHits.push({ stain_id: st.stain_id, x: pStart.x, y: pStart.y });

            ctx.save();
            ctx.beginPath();
            ctx.arc(pStart.x, pStart.y, isSelected ? 7.5 : 5.5, 0, Math.PI * 2);
            ctx.fillStyle = rayColor;
            ctx.fill();
            ctx.lineWidth = isSelected ? 2.5 : 1.2;
            ctx.strokeStyle = isSelected ? '#ffffff' : '#0b0f19';
            ctx.stroke();

            ctx.font = isSelected ? 'bold 12px Inter, sans-serif' : '11px Inter, sans-serif';
            ctx.fillStyle = isSelected ? '#ffffff' : '#cbd5e1';
            ctx.fillText(`${st.stain_id} (${st.angle_degrees.toFixed(1)}°)`, pStart.x + 10, pStart.y - 6);
            ctx.restore();
        });

        // 6. Draw 2D Convergence Point, Tangent 3D Origin, and 3D CPA Origin
        const pConv = project3D(convLab);
        ctx.save();
        ctx.beginPath();
        ctx.arc(pConv.x, pConv.y, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.fillStyle = '#fcd34d';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('2D Convergence', pConv.x + 8, pConv.y + 4);
        ctx.restore();

        const pTan = project3D(tanLab);
        const pCpa = project3D(cpaLab);

        if (showUncertainty) {
            const uncRadiusPx = Math.max(
                14,
                (sceneData.convergence_2d.convergence_radius_mm / maxSpan) * Math.min(w, h) * 0.62 * cam.zoom
            );
            ctx.save();
            ctx.beginPath();
            ctx.arc(pCpa.x, pCpa.y, uncRadiusPx, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(225, 29, 72, 0.16)';
            ctx.fill();
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = '#e11d48';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
        }

        // Tangent Origin Marker (Crimson Triangle/Circle)
        ctx.save();
        ctx.beginPath();
        ctx.arc(pTan.x, pTan.y, 6.5, 0, Math.PI * 2);
        ctx.fillStyle = '#e11d48';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.8;
        ctx.stroke();
        ctx.fillStyle = '#fda4af';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillText(`Tangent Origin (Z=${sceneData.tangent_origin_3d.z_mean_surface_mm.toFixed(1)} mm)`, pTan.x + 10, pTan.y - 6);
        ctx.restore();

        // 3D CPA Origin Marker (Emerald Diamond)
        ctx.save();
        ctx.beginPath();
        ctx.arc(pCpa.x, pCpa.y, 6.0, 0, Math.PI * 2);
        ctx.fillStyle = '#10b981';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.8;
        ctx.stroke();
        ctx.fillStyle = '#6ee7b7';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillText(`3D CPA (${cpaLab[0].toFixed(0)}, ${cpaLab[1].toFixed(0)}, ${cpaLab[2].toFixed(0)}) mm`, pCpa.x + 10, pCpa.y + 12);
        ctx.restore();
    }

    // Mouse Orbit / Pan / Click-Selection Handlers
    let isDragging = false;
    let dragMoved = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    canvas.addEventListener('mousedown', function (e) {
        isDragging = true;
        dragMoved = false;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });

    window.addEventListener('mousemove', function (e) {
        if (!isDragging) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragMoved = true;

        if (e.shiftKey) {
            cam.panX += dx;
            cam.panY += dy;
        } else {
            cam.yaw += dx * 0.008;
            cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.008));
        }
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        renderScene();
    });

    window.addEventListener('mouseup', function (e) {
        if (isDragging && !dragMoved) {
            // Check if user clicked on a stain marker
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const cx = (e.clientX - rect.left) * scaleX;
            const cy = (e.clientY - rect.top) * scaleY;

            for (const hit of projectedStainHits) {
                const dist = Math.hypot(cx - hit.x, cy - hit.y);
                if (dist <= 16) {
                    selectedStainId = hit.stain_id;
                    updateInspectorPanel(selectedStainId);
                    renderScene();
                    break;
                }
            }
        }
        isDragging = false;
    });

    canvas.addEventListener('wheel', function (e) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.1 : 0.9;
        cam.zoom = Math.max(0.35, Math.min(4.0, cam.zoom * factor));
        renderScene();
    }, { passive: false });

    // Camera Control Buttons
    const bindBtn = (id, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => { fn(); renderScene(); });
    };

    bindBtn('btnCamReset', () => { cam = { ...defaultCamera }; });
    bindBtn('btnCamTop', () => { cam.yaw = 0; cam.pitch = 1.55; cam.panX = 0; cam.panY = 0; });
    bindBtn('btnCamFront', () => { cam.yaw = 0; cam.pitch = 0.05; cam.panX = 0; cam.panY = 0; });
    bindBtn('btnCamSide', () => { cam.yaw = -Math.PI / 2; cam.pitch = 0.05; cam.panX = 0; cam.panY = 0; });
    bindBtn('btnZoomIn', () => { cam.zoom = Math.min(4.0, cam.zoom * 1.15); });
    bindBtn('btnZoomOut', () => { cam.zoom = Math.max(0.35, cam.zoom * 0.85); });

    // Visibility Checkboxes
    const chkFloor = document.getElementById('toggleFloor');
    if (chkFloor) chkFloor.addEventListener('change', function () { showFloor = this.checked; renderScene(); });

    const chkWall = document.getElementById('toggleWall');
    if (chkWall) chkWall.addEventListener('change', function () { showWall = this.checked; renderScene(); });

    const chkUnc = document.getElementById('toggleUncertainty');
    if (chkUnc) chkUnc.addEventListener('change', function () { showUncertainty = this.checked; renderScene(); });

    document.querySelectorAll('.stain-vis-toggle').forEach(chk => {
        chk.addEventListener('change', function () {
            const sid = this.getAttribute('data-stain-id');
            visibleStains[sid] = this.checked;
            renderScene();
        });
    });

    document.querySelectorAll('.stain-select-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            selectedStainId = this.getAttribute('data-stain-id');
            updateInspectorPanel(selectedStainId);
            renderScene();
        });
    });

    updateInspectorPanel(selectedStainId);
    renderScene();
});

