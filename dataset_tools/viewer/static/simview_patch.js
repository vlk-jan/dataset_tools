/*
 * Wrapper-side corrections to the stock SimView front-end.
 *
 * Nothing in the installed `simview` package is modified: every change is made
 * from outside, after SimView's own modules have run, through its public
 * instance (`window.__debugSimView`), its lil-gui controllers and its DOM.
 *
 *   PERMANENT  a wrapper concern -- bodies and layers only this dataset knows
 *              the meaning of. These stay.
 *   STOPGAP    a SimView bug papered over until the fix lands upstream. Each is
 *              gated on the installed version and stops firing by itself on
 *              upgrade: a workaround left running over a fixed release would
 *              override the real fix and hide whether it landed.
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

/** The chassis: the body a view of this dataset is meant to be looking at. */
export const ROBOT_BODY = "robot";

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
 * PERMANENT. Aim the orbit camera at the robot, keeping the viewing angle.
 *
 * SimView's camera is fixed at the world origin (`CAMERA_CONFIG.position`) and
 * never frames the scene it loaded. Most windows start the robot there -- a
 * segment is re-anchored to its own first pose -- but a split segment's later
 * parts carry on from where the previous one ended, and `?frame=N` seeks into
 * the middle of a traverse; either leaves the robot out of shot.
 *
 * Moving camera and orbit target by the same delta re-centres without changing
 * angle or distance, the move SimView makes in `changeFocusOnBatchByRowCol`.
 * Called once at load -- following every frame is what `trackBody` is for.
 * Returns false until the robot has a resolved position, so the caller retries.
 */
export function centerOnRobot(app) {
    const body = app.bodies && app.bodies.get(ROBOT_BODY);
    const scene = app.scene;
    if (!body || !scene || !scene.camera || !scene.controls || !app.batchManager) return false;

    // Body positions are per-batch and batch-local, exactly as SimView's own
    // tracking reads them: the world position is the batch's offset plus this
    // body's position within it.
    const batch = app.batchManager.currentlyActiveBatch;
    const position = body.positions && body.positions[batch];
    if (!position) return false;
    const offset = app.batchManager.getBatchOffset(batch);
    const x = position.x + offset.x;
    const y = position.y + offset.y;
    const z = position.z + offset.z;

    const { camera, controls } = scene;
    camera.position.set(
        camera.position.x + (x - controls.target.x),
        camera.position.y + (y - controls.target.y),
        camera.position.z + (z - controls.target.z),
    );
    controls.target.set(x, y, z);
    controls.update();
    return true;
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
    defaultPanelState(app);

    const desired = { cloud: false, ghost: false };

    // ---------------------------------------------------------------- cloud
    //
    // STOPGAP, gated with the rest. SimView <= 4.2 draws every body in one
    // global style, so seeing a point cloud means putting the whole scene in
    // "points" mode, which leaves the robot (a box) with nothing to draw. The
    // cloud is taken out of that mechanism instead and driven by the overlay's
    // checkbox. `refreshBatchVisibility` and `updateVisualizationMode` both
    // recompute `points[i].visible` from the global mode, so the wish is
    // re-asserted rather than set once; `group.visible` SimView never touches.
    // Past the gate SimView has its own "Show Point Clouds" checkbox and ours
    // steps aside -- two controls for one thing would be worse than the trap.
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
 * PERMANENT. Open the control panel on what this viewer is actually for.
 *
 * SimView opens Body and Terrain Options and closes Camera Options, which fits
 * a physics scene where the terrain's properties are the subject. Here the
 * terrain is scenery -- reconstructed ground the robot drove over -- and its
 * colour layers are explained by the overlay's own legend, while framing the
 * run (track body, FOV, split screen) is the thing people reach for. The Body
 * States window is likewise a readout to open on demand, not a panel that
 * should cover the scene on every window open.
 *
 * All three are set through SimView's public objects, so a viewer can undo any
 * of them with one click and nothing is removed from the UI.
 */
function defaultPanelState(app) {
    collapseBodyStates(app);
    applyFolderDefaults(app);
}

/** Start the Body States window collapsed; its header still expands it. */
function collapseBodyStates(app) {
    const bsw = app.bodyStateWindow;
    if (!bsw || !bsw.window) return;
    // `toggleCollapse` rather than the class directly: it also flips the
    // header's caret, which a bare classList write would leave pointing down.
    if (!bsw.window.classList.contains("collapsed")) bsw.toggleCollapse();
}

/** Which of SimView's lil-gui folders start open, by their title. Folders not
 *  named here (Body Options, Scene Info) keep SimView's own default. */
const FOLDER_DEFAULTS = {
    "Terrain Options": false,
    "Camera Options": true,
};

function applyFolderDefaults(app) {
    const gui = app.uiControls && app.uiControls.gui;
    const folders = (gui && gui.folders) || [];
    const seen = new Set();
    for (const folder of folders) {
        // `$title` is lil-gui's title element; the folder's own `_title` is
        // private, and matching on text is what survives a lil-gui bump.
        const title = folder.$title ? folder.$title.textContent.trim() : "";
        if (!(title in FOLDER_DEFAULTS)) continue;
        seen.add(title);
        folder.open(FOLDER_DEFAULTS[title]);
    }
    for (const title of Object.keys(FOLDER_DEFAULTS)) {
        if (!seen.has(title)) {
            // A rename upstream, not a bug here: say so rather than silently
            // leaving the panel in SimView's default state.
            console.info(`no lil-gui folder titled "${title}"; leaving it as SimView left it.`);
        }
    }
}

/**
 * PERMANENT. Keep bookkeeping bodies out of the "Body states" list and the
 * "Track Body" camera picker.
 *
 * The cloud has no per-frame pose (it is already accumulated in the window's
 * frame), so its row opens on an empty readout. Ungated: SimView >= 4.3 drops
 * stateless bodies itself, making this a no-op, and it stays right for any
 * future housekeeping body.
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
 * SimView 4.2 builds every body's material from one shared DEFAULT_MESH_CONFIG,
 * so a per-body colour cannot be declared from Python -- it has to be set here,
 * on the object SimView built.
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
/* three.js r174 computes an InstancedMesh's bounding sphere from the       */
/* instance matrices once, on the first frustum test, and caches it.        */
/* `instanceMatrix.needsUpdate` does not invalidate it, and SimView never   */
/* sets `frustumCulled = false` for bodies (it does for trails,             */
/* Body.js:227), so a body is culled once its STARTING pose leaves the      */
/* frustum -- the robot blinking out down a long traverse. Turning culling  */
/* off for a few bodies is what the upstream fix does.                      */
/*                                                                          */
/* Separately: config.js hard-codes the far plane and orbit max distance to */
/* 500, but one window can span more (scene 0000 segment 1 is ~300 m plus   */
/* terrain margin), so pulling back to see it all clips it. Newer SimView   */
/* derives both from the terrain's own bounds.                              */
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
/* `determineAvailableVisualizationModes` offers the union of the modes any */
/* body has, and `updateVisualizationMode` applies the chosen one to every  */
/* body with no fallback (Body.js:697), so a scene mixing a cloud with      */
/* solid bodies gets a "points" entry that makes the robot disappear.       */
/* Upstream is adding a preference-with-fallback.                           */
/*                                                                          */
/* Until then the entry is removed from the menu and from the array the "B" */
/* key cycles through. The cloud is shown by the overlay's own checkbox     */
/* instead -- see `setCloudVisible` above.                                  */
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
/* Fixed in 4.2.1, so this does not run there; gated with the other two.    */
/* Delete this block on upgrade once the viewer's floor is past 4.2.0.      */
/*                                                                          */
/* Three independent causes (maintainers, 2026-09-02), all fixed below:     */
/*  1. charts are built with empty data, so init autoscale leaves           */
/*     `scales.x.min/max` null and `valToPos` is NaN -- nothing paints.     */
/*     Fixed by pinning x to the window's fixed time span.                  */
/*  2. the render loop calls `redraw(false, true)`, so paths freeze at the  */
/*     first frame's single point. Fixed by forcing rebuildPaths each tick. */
/*  3. the y axis ships one tick increment (range/5, ~150px) that never     */
/*     fits the ~76px plot. Fixed by offering coarser multiples.            */
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
