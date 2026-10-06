/*
 * Wrapper-side corrections to the stock SimView front-end.
 *
 * Nothing in the installed `simview` package is modified: what is here works
 * from outside, through SimView's public instance (`window.simview`), its
 * events and its DOM.
 *
 * What the scene itself can say -- the ghost's look, which folders open, the
 * point cloud starting hidden -- is declared at build time (build.py's
 * `viewer_defaults` and `create_body`), not patched in here.
 */

/** The wheel-odometry pose marker (see build.py's `_ghost_track`). */
export const GHOST_BODY = "wheel_odom_ghost";

/** The chassis: the body a view of this dataset is meant to be looking at. */
export const ROBOT_BODY = "robot";

/*
 * SimView announces itself with `simview:ready` once its data is loaded and
 * its UI built. Listened for from the moment this module loads, which is
 * before overlay.js inserts SimView's entry script, so the event cannot be
 * missed; every caller shares the one promise.
 */
const ready = new Promise((resolve) => {
    window.addEventListener("simview:ready", (event) => resolve(event.detail.simview), { once: true });
});

/** The running SimView instance, once it is ready. */
export function waitForSimView(timeoutMs = 120000) {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("SimView did not come up")), timeoutMs);
    });
    return Promise.race([ready, timeout]).finally(() => clearTimeout(timer));
}

/** Apply every patch and return the handles the overlay drives them with. */
export function installSimviewPatches(app) {
    // A layout nicety: one that throws must not cost the overlay its handles.
    try {
        fitAnalysisBar(app);
    } catch (error) {
        console.warn("SimView patch fitAnalysisBar not applied:", error);
    }

    // The ghost is built hidden (build.py); the panel's switch shows it.
    const ghost = app.bodies.get(GHOST_BODY);

    function setGhostVisible(on) {
        if (ghost) ghost.group.visible = !!on;
    }

    return {
        setGhostVisible,
        hasGhost: !!ghost,
        terrainColorMode: () => app.uiState.terrainColorMode,
    };
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
