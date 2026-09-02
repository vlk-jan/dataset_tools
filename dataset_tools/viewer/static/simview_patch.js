/*
 * Wrapper-side corrections to the stock SimView front-end.
 *
 * Nothing in the installed `simview` package is modified: every change here is
 * made from the outside, after SimView's own modules have run, through its
 * public instance (`window.__debugSimView`), its lil-gui controllers and its
 * DOM. The viewer runs against the PyPI wheel, so anything that needs a change
 * *inside* SimView is out of scope here by construction.
 *
 * Two kinds of block live in this file and they are labelled as such:
 *
 *   PERMANENT  a wrapper concern -- this dataset has bodies and layers that
 *              SimView cannot know the meaning of, so only we can present
 *              them correctly. These stay.
 *
 *   STOPGAP    a SimView bug we are papering over until the fix lands
 *              upstream. Each one is self-contained, so it can be deleted in
 *              one piece -- and until someone gets round to deleting it, it is
 *              gated on the installed version and stops firing by itself the
 *              moment the package is upgraded. That gate is the important
 *              part: a workaround still running over a version that has been
 *              fixed does not merely waste a few instructions, it overrides
 *              the real fix with a cruder one and hides whether it landed.
 */

/** The last SimView release that needs the workarounds below. Bodies leaving
 *  the visualization-mode trap, instanced-mesh culling and a bounds-derived
 *  camera far plane are all fixed after this. */
export const LAST_SIMVIEW_NEEDING_STOPGAPS = "4.2.0";

/**
 * Is `version` no newer than `bound`? Both dotted numeric strings.
 *
 * An unparseable or unknown version counts as *newer*, so a workaround is
 * never applied to a SimView we cannot identify: failing to patch a broken
 * viewer is a visible annoyance, patching a fixed one is a silent regression.
 */
export function versionAtMost(version, bound) {
    const parse = (v) => String(v).split(".").map((p) => parseInt(p, 10));
    const a = parse(version);
    const b = parse(bound);
    if (a.some((n) => !Number.isFinite(n))) return false;
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const x = a[i] || 0;
        const y = b[i] || 0;
        if (x !== y) return x < y;
    }
    return true;
}

/** Bodies that exist for bookkeeping, not as things with a pose worth reading.
 *  `lidar_cloud` is the window's clouds already accumulated in the world frame:
 *  it is a body only because that is how SimView carries point clouds, and it
 *  is deliberately stateless (see build.py). */
export const HOUSEKEEPING_BODIES = ["lidar_cloud"];

/** The wheel-odometry pose marker (see build.py's `_ghost_track`). */
export const GHOST_BODY = "wheel_odom_ghost";

/** The aggregated lidar cloud body. */
export const CLOUD_BODY = "lidar_cloud";

/** Resolve the running SimView instance once it has a loaded store. */
export function waitForSimView(timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
        const deadline = performance.now() + timeoutMs;
        const attempt = () => {
            const app = window.__debugSimView;
            if (app && app.bodies && app.uiControls && app.store && app.store.length > 0) {
                resolve(app);
                return;
            }
            if (performance.now() > deadline) {
                reject(new Error("SimView did not come up"));
                return;
            }
            requestAnimationFrame(attempt);
        };
        attempt();
    });
}

/**
 * Apply every patch and return the handles the overlay drives them with.
 *
 * `enforce()` is called from the overlay's existing animation-frame loop; it
 * is a handful of comparisons and writes nothing when the scene already
 * matches what we asked for.
 */
export function installSimviewPatches(app, simviewVersion) {
    // Everything below the gate is a workaround for a bug that is fixed in
    // SimView after LAST_SIMVIEW_NEEDING_STOPGAPS. On a newer package these do
    // not run at all, and SimView's own -- better informed -- versions of them
    // take over: point-cloud bodies out of the mode dropdown with their own
    // "Show Point Clouds" checkbox, uncullable instanced meshes, and a camera
    // far plane derived from the terrain's real bounds.
    const stopgaps = versionAtMost(simviewVersion, LAST_SIMVIEW_NEEDING_STOPGAPS);
    if (!stopgaps) {
        console.info(
            `SimView ${simviewVersion} > ${LAST_SIMVIEW_NEEDING_STOPGAPS}: ` +
            "skipping the viewer's workarounds, SimView handles these itself."
        );
    }
    if (stopgaps) {
        stopgapOriginCulling(app);
        stopgapDropPointsMode(app);
    }
    hideHousekeepingBodies(app);
    styleGhost(app);
    defaultPointCloudsOff(app);

    const desired = { cloud: false, ghost: false };

    // ---------------------------------------------------------------- cloud
    //
    // STOPGAP, gated with the rest. SimView <= 4.2 draws every body in one
    // global style, so the only way to see a point-cloud body is to put the
    // whole scene in "points" mode -- which turns the robot into nothing at
    // all, since a box has no point representation. The cloud is taken out of
    // that mechanism entirely instead: the scene stays in "mesh" mode (solid
    // robot) and the cloud's own Points objects are shown or hidden on their
    // own, driven by the overlay's checkbox.
    //
    // `Body.refreshBatchVisibility` recomputes `points[i].visible` from the
    // global mode whenever the visible batch set changes, and
    // `updateVisualizationMode` does the same on every mode change, so the
    // wish has to be re-asserted rather than set once. `group.visible` is
    // never touched by SimView, which is why the outer gate can be set once.
    //
    // Past the gate this whole control belongs to SimView: it takes point
    // clouds out of the mode dropdown itself and gives them a "Show Point
    // Clouds" checkbox. Two controls for one thing that can disagree with each
    // other would be worse than the trap this replaced, so ours steps aside
    // (the overlay hides its row -- see setupViewToggles).
    const cloud = app.bodies.get(CLOUD_BODY);
    const cloudPoints = () =>
        cloud ? (cloud.representations.points || []).filter(Boolean) : [];
    const ownsCloudToggle = stopgaps && !!cloud;

    function setCloudVisible(on) {
        if (!ownsCloudToggle) return;
        desired.cloud = !!on;
        cloud.group.visible = desired.cloud;
        enforce();
    }

    // ---------------------------------------------------------------- ghost
    //
    // PERMANENT. The ghost is a diagnostic, not part of the robot: it is where
    // wheel odometry alone believes the robot is, and it can end up a hundred
    // metres away. Shown unasked-for it reads as an unexplained box drifting
    // around the scene, so it is off until someone asks for it by name.
    const ghost = app.bodies.get(GHOST_BODY);

    function setGhostVisible(on) {
        desired.ghost = !!on && !!ghost;
        if (ghost) ghost.group.visible = desired.ghost;
    }

    function enforce() {
        if (ownsCloudToggle) {
            for (const points of cloudPoints()) {
                if (points.visible !== desired.cloud) points.visible = desired.cloud;
            }
        }
        if (stopgaps) stopgapScalarXScale(app);
    }

    if (ownsCloudToggle) setCloudVisible(false);
    setGhostVisible(false);

    return {
        setCloudVisible,
        setGhostVisible,
        enforce,
        stopgaps,
        hasCloud: ownsCloudToggle,
        hasGhost: !!ghost,
        /** SimView's own "Show Surface" checkbox, so the overlay can drop the
         *  terrain surface while the cloud is up (they describe the same
         *  ground, and the surface is drawn over the points). */
        setSurfaceVisible(on) {
            const ctrl = app.uiControls.findController("showSurface");
            if (ctrl && ctrl.getValue() !== !!on) ctrl.setValue(!!on);
        },
        terrainColorMode: () => app.uiState.terrainColorMode,
    };
}

/**
 * PERMANENT. Start with the aggregated point cloud hidden.
 *
 * SimView 4.2.1 defaults `pointCloudsVisible: true` (config.js), so the cloud
 * would cover the terrain on every window open. The cloud is a heavy, opt-in
 * overlay -- like the ghost -- so we default it off and let the viewer turn it
 * on via SimView's own "Show Point Clouds" checkbox. Driving that controller
 * (rather than poking `uiState` directly) fires its onChange, so the checkbox
 * shows unchecked and every body's points are hidden in one call. No-op on a
 * scene with no point-cloud body, where the controller does not exist.
 */
function defaultPointCloudsOff(app) {
    const ctrl = app.uiControls && app.uiControls.findController("showPointClouds");
    if (ctrl && ctrl.getValue() !== false) ctrl.setValue(false);
}

/**
 * PERMANENT. Keep bookkeeping bodies out of the "Body states" list and the
 * "Track Body" camera picker.
 *
 * The cloud carries no per-frame pose at all (build.py gives it no trajectory
 * -- it is already accumulated in the window's own frame), so its row opens on
 * an empty state readout. SimView >= 4.3 drops stateless point-cloud bodies
 * from this list itself, at which point this becomes a no-op rather than a
 * workaround, which is why it is not gated: it costs one lookup and it is
 * still right if a future scene grows another housekeeping body.
 */
function hideHousekeepingBodies(app) {
    const list = app.bodyStateWindow && app.bodyStateWindow.bodyListItems;
    for (const name of HOUSEKEEPING_BODIES) {
        const item = list && list.get(name);
        // `display` rather than the `hidden` attribute: BodyStateWindow's own
        // stylesheet sets `display` on `.body-list-item`, which would win.
        if (item) item.style.display = "none";

        const track = app.uiControls.findController("trackBody");
        const select = track && track.domElement.querySelector("select");
        if (!select) continue;
        for (const option of [...select.options]) {
            if (option.value === name) option.remove();
        }
    }
}

/**
 * PERMANENT. Give the ghost a look that says "this is not the robot".
 *
 * SimView 4.2 builds every dynamic body's mesh material from one shared
 * `DEFAULT_MESH_CONFIG` (`Body.createInstancedRepresentation` passes
 * `BODY_CONFIG.mesh`, which does not exist), so a per-body colour cannot be
 * declared from the Python side at all -- the material has to be recoloured
 * here, on the object SimView built.
 */
function styleGhost(app) {
    const ghost = app.bodies.get(GHOST_BODY);
    const mesh = ghost && ghost.representations.mesh;
    if (!mesh || !mesh.material) return;
    mesh.material.color.set(0xffb347); // amber, against the robot's grey
    mesh.material.transparent = true;
    mesh.material.opacity = 0.55;
    mesh.material.depthWrite = false;
    mesh.material.needsUpdate = true;
}

/* ======================================================================== */
/* STOPGAP for simview <= 4.2: origin-anchored frustum culling.             */
/* Delete this block on upgrade.                                            */
/*                                                                          */
/* Every dynamic body is a THREE.InstancedMesh whose pose is written into    */
/* `instanceMatrix` each frame (Body.updateInstanceMatrix). three.js r174    */
/* computes an InstancedMesh's bounding sphere from the instance matrices    */
/* *once*, lazily, on the first frustum test (Frustum.intersectsObject sees  */
/* `boundingSphere === null`, computes it, and caches it forever);           */
/* `instanceMatrix.needsUpdate = true` does not invalidate it, and SimView   */
/* never sets `frustumCulled = false` -- which it does do for trails         */
/* (Body.js:227). So the sphere keeps describing where the body *started*.   */
/*                                                                          */
/* The cull therefore fires when the body's STARTING pose leaves the view    */
/* frustum, not when the body gets far from anything: it is a function of    */
/* where the camera is pointed, which is why it reads as erratic. Following  */
/* the robot down a long traverse is simply the easiest way to provoke it,   */
/* since the start is behind you by then.                                    */
/*                                                                          */
/* Turning culling off for a handful of bodies costs nothing -- they are a   */
/* few draw calls -- and is what the upstream fix does.                      */
/*                                                                          */
/* Separately, and *not* the cause above: the camera far plane and the orbit */
/* controls' max distance are hard-coded to 500 in SimView's config.js,      */
/* while a single window here can span more than that (segment 1 of scene    */
/* 0000 is ~300 m end to end and its terrain mesh covers all of it plus a    */
/* margin), so pulling back far enough to see the whole traverse clips it.   */
/* Widened here only while the gate holds: newer SimView derives both from   */
/* the terrain's own bounds, which is strictly better than this guess.       */
/* ======================================================================== */
function stopgapOriginCulling(app) {
    app.bodies.forEach((body) => {
        for (const type of ["mesh", "wireframe"]) {
            const rep = body.representations[type];
            if (rep && rep.isInstancedMesh) rep.frustumCulled = false;
        }
    });

    const camera = app.scene && app.scene.camera;
    if (camera && camera.far < 5000) {
        camera.far = 5000;
        camera.updateProjectionMatrix();
    }
    const controls = app.scene && app.scene.controls;
    if (controls && controls.maxDistance < 2000) controls.maxDistance = 2000;
}

/* ======================================================================== */
/* STOPGAP for simview <= 4.2: the "points" body-visualization mode.        */
/* Delete this block on upgrade.                                            */
/*                                                                          */
/* `UIControls.determineAvailableVisualizationModes` offers the *union* of   */
/* the modes any body has, and `updateVisualizationMode` then applies the    */
/* chosen one to *every* body with no fallback (Body.js:697). A scene that   */
/* mixes a point cloud with solid bodies therefore gets a "points" entry     */
/* that makes the robot disappear -- it is the cloud's mode, offered as if   */
/* it were the scene's. Upstream is adding a preference-with-fallback.       */
/*                                                                          */
/* Until then the entry is removed from the menu (and from the array the "B" */
/* key cycles through, so the keyboard cannot reach it either). The cloud is */
/* shown by the overlay's own checkbox instead, which does not go through    */
/* the global mode at all -- see `setCloudVisible` above.                    */
/* ======================================================================== */
function stopgapDropPointsMode(app) {
    const ctrl = app.uiControls.findController("bodyVisualizationMode");
    if (!ctrl) return;

    const modes = app.uiControls.visualizationModes;
    const at = Array.isArray(modes) ? modes.indexOf("points") : -1;
    if (at >= 0) modes.splice(at, 1); // mutated in place: the "B" key reads it

    const select = ctrl.domElement.querySelector("select");
    if (select) {
        for (const option of [...select.options]) {
            if (option.value === "points") option.remove();
        }
    }
    if (app.uiState.bodyVisualizationMode === "points") ctrl.setValue("mesh");
}

/* ======================================================================== */
/* STOPGAP for simview <= 4.2.0: ScalarPlotter plots never draw (3 causes). */
/* Version-gated with the other two (see installSimviewPatches). Fixed in   */
/* simview 4.2.1, so this does not run there. Delete this block on upgrade   */
/* once the viewer's floor is past 4.2.0.                                    */
/*                                                                          */
/* Three independent causes (maintainers, 2026-09-02), all fixed below:      */
/*  1. The uPlot charts are built with empty data, so init autoscale leaves  */
/*     `scales.x.min/max` null (in uPlot the configured min/max are outputs, */
/*     `range` is the pin). Later updates go through                         */
/*     `setEndIndex -> setData(data, false)`, which never revisits x, so     */
/*     `valToPos` is NaN and nothing paints. Fixed by pinning x to the       */
/*     window's fixed time span.                                             */
/*  2. The render loop calls `redraw(false, true)` -- rebuildPaths=false --  */
/*     so paths freeze at the first frame's single point and later frames    */
/*     repaint a stale stub (pixels across x, no y extent). Fixed by forcing */
/*     `redraw(true, true)` on the active chart each tick.                   */
/*  3. The y axis ships a single tick increment (range/5, ~150px) that never */
/*     fits the ~76px plot, so `_found` stays [0,0] and no ticks draw. Fixed */
/*     by offering coarser multiples so one always fits.                     */
/* ======================================================================== */
const _dtScalarFixed = new WeakSet();

function stopgapScalarXScale(app) {
    const plotter = app.scalarPlotter;
    if (!plotter || !plotter.charts || !plotter.times || plotter.times.length < 2) return;
    const first = plotter.times[0];
    const last = plotter.times[plotter.times.length - 1];

    // Cause 1 (x scale) + cause 3 (y ticks), applied once per chart. See the
    // maintainers' 2026-09-02 analysis: ScalarPlotter's blank plot on <= 4.2 has
    // THREE causes, and only all three together make it draw.
    for (const chart of plotter.charts.values()) {
        if (_dtScalarFixed.has(chart)) continue;
        const xs = chart.data && chart.data[0];
        if (!xs || xs.length < 2) continue;
        // 1: the x scale is never pinned, so `valToPos` is NaN and nothing paints.
        if (chart.scales.x.min == null) chart.scales.x.range = () => [first, last];
        // 3: the y axis ships a single permitted tick increment (range/5) that
        // needs ~150px; the panel plot is ~76px tall, so no increment fits and
        // `_found` stays [0,0] -- no ticks, no labels. Offer coarser multiples
        // so one always fits the short axis.
        const yaxis = chart.axes && chart.axes[1];
        if (yaxis && Array.isArray(yaxis.incrs) && yaxis.incrs.length === 1) {
            const base = yaxis.incrs[0];
            if (base > 0) {
                yaxis.incrs = [1, 2, 2.5, 5, 10, 20, 50, 100].map((m) => base * m);
            }
        }
        chart.setData(chart.data, true);
        _dtScalarFixed.add(chart);
    }

    // Cause 2: the render loop calls `redraw(false, true)` -- rebuildPaths=false
    // -- so once a chart is drawn its path is frozen and later frames only
    // repaint the stale ~10px stub, which reads as "still blank". Rebuild the
    // paths of the one chart on screen each tick (cheap: a single active chart).
    const active = plotter.charts.get(plotter.activeScalar);
    if (active && _dtScalarFixed.has(active)) active.redraw(true, true);
}
