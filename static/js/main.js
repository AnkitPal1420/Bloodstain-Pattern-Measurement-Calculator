/**
 * main.js
 * =======
 * Frontend interactions for Forensic Laboratory System.
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Mobile Sidebar Toggle
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('appSidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function (e) {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                    sidebar.classList.remove('open');
                }
            }
        });
    }

    // 2. Preset Buttons in Measurement Calculator
    const presetButtons = document.querySelectorAll('.btn-preset');
    const widthInput = document.getElementById('width');
    const lengthInput = document.getElementById('length');

    presetButtons.forEach(button => {
        button.addEventListener('click', function () {
            const w = this.getAttribute('data-width');
            const l = this.getAttribute('data-length');
            if (widthInput && lengthInput && w && l) {
                widthInput.value = w;
                lengthInput.value = l;
            }
        });
    });

    // 3. Client-side sanity check on width <= length
    const measurementForm = document.getElementById('measurement-form');
    if (measurementForm && widthInput && lengthInput) {
        measurementForm.addEventListener('submit', function (e) {
            const w = parseFloat(widthInput.value);
            const l = parseFloat(lengthInput.value);
            if (!isNaN(w) && !isNaN(l) && w > l) {
                // Let the server-side validation also execute for robustness,
                // but highlight the inputs
                widthInput.style.borderColor = 'var(--crimson)';
                lengthInput.style.borderColor = 'var(--crimson)';
            }
        });
    }

    // 4. Phase 3 Computer Vision Stage View Switcher Tabs
    const tabButtons = document.querySelectorAll('.cv-tab-btn');
    const stagePanes = document.querySelectorAll('.cv-stage-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            const targetId = this.getAttribute('data-target');
            tabButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            stagePanes.forEach(pane => {
                if (pane.id === targetId) {
                    pane.style.display = 'block';
                } else {
                    pane.style.display = 'none';
                }
            });
        });
    });

    // 5. Phase 3 Interactive Two-Point Spatial Calibration Click Handler
    const interactiveImg = document.getElementById('cv-interactive-img');
    const calX1 = document.getElementById('cal_x1');
    const calY1 = document.getElementById('cal_y1');
    const calX2 = document.getElementById('cal_x2');
    const calY2 = document.getElementById('cal_y2');
    const knownMm = document.getElementById('known_length_mm');
    const btnPresetScale = document.getElementById('btn-preset-scale');
    const btnClearScale = document.getElementById('btn-clear-scale');

    let nextPointIndex = 1;

    if (interactiveImg && calX1 && calY1 && calX2 && calY2) {
        interactiveImg.addEventListener('click', function (e) {
            const rect = interactiveImg.getBoundingClientRect();
            const natW = parseFloat(interactiveImg.getAttribute('data-natural-width')) || interactiveImg.naturalWidth || rect.width;
            const natH = parseFloat(interactiveImg.getAttribute('data-natural-height')) || interactiveImg.naturalHeight || rect.height;

            const scaleX = natW / rect.width;
            const scaleY = natH / rect.height;

            const clickX = Math.round((e.clientX - rect.left) * scaleX * 10) / 10;
            const clickY = Math.round((e.clientY - rect.top) * scaleY * 10) / 10;

            if (nextPointIndex === 1) {
                calX1.value = clickX;
                calY1.value = clickY;
                nextPointIndex = 2;
            } else {
                calX2.value = clickX;
                calY2.value = clickY;
                nextPointIndex = 1;
            }
        });
    }

    if (btnPresetScale && calX1 && calY1 && calX2 && calY2 && knownMm) {
        btnPresetScale.addEventListener('click', function () {
            calX1.value = 60;
            calY1.value = 340;
            calX2.value = 100;
            calY2.value = 340;
            knownMm.value = 10.0;
        });
    }

    if (btnClearScale && calX1 && calY1 && calX2 && calY2) {
        btnClearScale.addEventListener('click', function () {
            calX1.value = '';
            calY1.value = '';
            calX2.value = '';
            calY2.value = '';
            nextPointIndex = 1;
        });
    }
});
