/*
 * Wrapper-side corrections to SimView, through `window.simview`, its events and
 * DOM; overlay.js applies them. What the scene can declare itself is set in
 * build.py instead.
 */

/** The transforms.npy pose marker (see build.py's `_ghost_track`). */
export const GHOST_BODY = "wheel_odom_ghost";

/** The chassis: the body a view of this dataset is meant to be looking at. */
export const ROBOT_BODY = "robot";

/* Listening from module load, before overlay.js inserts SimView, so
 * `simview:ready` cannot be missed. */
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

/**
 * Pin SimView's Analysis bar between the Body states window and the controls.
 * Measured rather than constant: panel sizes change between simview versions.
 */
const ANALYSIS_GAP = 10;

export function fitAnalysisBar(app) {
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
