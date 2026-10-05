/*
 * Wrapper-side corrections to the stock SimView front-end.
 *
 * Nothing in the installed `simview` package is modified: every change is made
 * from outside, after SimView's own modules have run, through its public
 * instance (`window.__debugSimView`), its lil-gui controllers and its DOM.
 *
 * Every change here is a wrapper concern -- bodies and layers only this dataset
 * knows the meaning of. Workarounds for SimView bugs lived here too until
 * simview 5 fixed them upstream.
 */

/** The wheel-odometry pose marker (see build.py's `_ghost_track`). */
export const GHOST_BODY = "wheel_odom_ghost";

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
 * Aim the orbit camera at the robot, keeping the viewing angle.
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

/** Apply every patch and return the handles the overlay drives them with. */
export function installSimviewPatches(app) {
    // Each on its own: SimView changes under these patches (an editable
    // install moves with its repo), and one that throws must not take the
    // others with it -- a renamed method once left every default unapplied.
    for (const patch of [styleGhost, defaultPointCloudsOff, collapseBodyStates, applyFolderDefaults, fitAnalysisBar]) {
        try {
            patch(app);
        } catch (error) {
            console.warn(`SimView patch ${patch.name} not applied:`, error);
        }
    }

    // ---------------------------------------------------------------- ghost
    //
    // The ghost is a diagnostic, not part of the robot: it is where
    // wheel odometry alone believes the robot is, and it can end up a hundred
    // metres away. Shown unasked-for it reads as an unexplained box drifting
    // around the scene, so it is off until someone asks for it by name.
    const ghost = app.bodies.get(GHOST_BODY);

    function setGhostVisible(on) {
        if (ghost) ghost.group.visible = !!on;
    }

    setGhostVisible(false);

    return {
        setGhostVisible,
        hasGhost: !!ghost,
        terrainColorMode: () => app.uiState.terrainColorMode,
    };
}

/**
 * Start with the aggregated point cloud hidden.
 *
 * SimView defaults `pointCloudsVisible: true` (config.js), so the cloud
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

/*
 * The two panel defaults below open the controls on what this viewer is
 * actually for.
 *
 * SimView opens Body and Terrain Options and closes Camera Options, which fits
 * a physics scene where the terrain's properties are the subject. Here the
 * terrain is scenery -- reconstructed ground the robot drove over -- and its
 * colour layers are explained by the overlay's own legend, while framing the
 * run (track body, FOV, split screen) is the thing people reach for. The Body
 * States window is likewise a readout to open on demand, not a panel that
 * should cover the scene on every window open.
 *
 * Both are set through SimView's public objects, so a viewer can undo any of
 * them with one click and nothing is removed from the UI.
 */

/** Start the Body States window collapsed; its header still expands it. */
function collapseBodyStates(app) {
    const bsw = app.bodyStateWindow;
    if (!bsw || !bsw.window) return;
    // A <details> since simview's collapsible-panel refactor: closing it is all there is.
    bsw.window.open = false;
}

/** Which of SimView's lil-gui folders start open, by their title -- all four
 *  named, so a change in SimView's own defaults does not change this view. */
const FOLDER_DEFAULTS = {
    "Body Options": true,
    "Terrain Options": false,
    "Camera Options": true,
    "Scene Info": false,
};

function applyFolderDefaults(app) {
    const gui = app.uiControls && app.uiControls.gui;
    const folders = (gui && gui.folders) || [];
    for (const folder of folders) {
        // `$title` is lil-gui's title element; the folder's own `_title` is
        // private, and matching on text is what survives a lil-gui bump.
        const title = folder.$title ? folder.$title.textContent.trim() : "";
        if (title in FOLDER_DEFAULTS) folder.open(FOLDER_DEFAULTS[title]);
    }
}

/**
 * Pin SimView's Analysis bar between the Body states window and the controls,
 * the same gap from each.
 *
 * Measured, not written as constants in overlay.css: both panels' sizes have
 * changed between simview versions (the Body states window lost 33 px in the
 * collapsible-panel refactor), and fixed numbers left the gaps lopsided twice.
 * Re-fitted when either panel resizes, the window resizes, or the data panel
 * opens or closes (the controls move with it, via a class on <body>).
 */
const ANALYSIS_GAP = 10;

function fitAnalysisBar(app) {
    const bar = document.querySelector(".analysis-container");
    const left = app.bodyStateWindow && app.bodyStateWindow.window;
    const right = document.querySelector(".lil-gui.root");
    if (!bar || !left || !right) return;
    const fit = () => {
        const l = left.getBoundingClientRect();
        const r = right.getBoundingClientRect();
        bar.style.setProperty("left", `${l.right + ANALYSIS_GAP}px`, "important");
        bar.style.setProperty("right", `${window.innerWidth - r.left + ANALYSIS_GAP}px`, "important");
    };
    const resized = new ResizeObserver(fit);
    resized.observe(left);
    resized.observe(right);
    new MutationObserver(fit).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("resize", fit);
    fit();
}

/**
 * Give the ghost a look that says "this is not the robot".
 *
 * SimView builds every body's material from one shared mesh config and its
 * `create_body` takes no colour, so this is set here, on the object it built.
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
