/*
 * Time-synced per-frame data panel for the SimView viewer.
 *
 * SimView renders geometry; a dataset frame also carries a camera image, a BEV
 * height map, its validity mask and a raw cloud, none of which are geometry.
 * This controller shows those next to the 3D view and keeps them on the frame
 * the playhead is currently at.
 *
 * It does that without modifying SimView at all. Two stock affordances make it
 * possible:
 *
 *   1. `window.__simviewStaticBase` (set by the server-rendered page) points
 *      SimView's fetches at our own bundle endpoints.
 *   2. `window.__debugSimView` is the running SimView instance, whose
 *      `animationController.currentTime` is the playhead. We poll it; we never
 *      write to it.
 *
 * SimView's entry module is withheld by the server until the window's bundle
 * has been built (see `_defer_main_script`), so this file loads it once the
 * data it will immediately fetch is actually there.
 */

const cfg = window.__dtViewer || {};

const els = {
  build: document.getElementById("dt-build"),
  buildMsg: document.querySelector("#dt-build .dt-build-msg"),
  panel: document.getElementById("dt-panel"),
  collapse: document.getElementById("dt-collapse"),
  expand: document.getElementById("dt-expand"),
  frame: document.getElementById("dt-frame"),
  time: document.getElementById("dt-time"),
  stamp: document.getElementById("dt-stamp"),
  pose: document.getElementById("dt-pose"),
  buildInfo: document.getElementById("dt-buildinfo"),
  tiles: Array.from(document.querySelectorAll(".dt-tile")),
};

let meta = null;
let available = {};
let lastFrame = -1;
let lastUpdate = 0;

/** Minimum ms between panel updates: playback runs far faster than the images
 *  can be fetched, and every frame is a separate HTTP request. */
const UPDATE_INTERVAL_MS = 110;

function fail(message) {
  els.build.classList.add("failed");
  els.build.classList.remove("done");
  els.buildMsg.textContent = message;
}

/** Fetch the window's metadata, which also triggers the (cached) build. */
async function loadMeta() {
  const response = await fetch(cfg.metaUrl);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (e) {
      /* a non-JSON error body is fine; the status line is enough */
    }
    throw new Error(detail);
  }
  return response.json();
}

/** Load SimView's own entry point, now that its data exists. */
function startSimView() {
  const src = window.__dtSimviewEntry;
  if (!src) {
    fail("SimView's entry script was not found in the page.");
    return;
  }
  const script = document.createElement("script");
  script.type = "module";
  script.src = src;
  script.onerror = () => fail("SimView's viewer script failed to load.");
  document.body.appendChild(script);
}

/** Nearest frame index in the window for a playback time (seconds). */
function frameIndexForTime(t) {
  const times = meta.times;
  if (!times || times.length === 0) return 0;
  let lo = 0;
  let hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid;
  }
  return t - times[lo] <= times[hi] - t ? lo : hi;
}

function setTile(tile, frame) {
  const kind = tile.dataset.kind;
  const wrap = tile.querySelector(".dt-imgwrap");
  const img = tile.querySelector("img");
  const list = available[kind];

  // `available` comes from the build and lists, per kind, exactly which frames
  // have a file. A kind missing from it entirely (this scene has no height
  // maps at all) has no list; a kind that has one may still be missing this
  // particular frame, because those directories are built in flight. Either
  // way there is nothing to request -- asking anyway would 404 per frame.
  if (!list || !list.has(frame)) {
    wrap.classList.add("absent");
    img.classList.add("hidden");
    img.removeAttribute("src");
    return;
  }
  const url = `${cfg.frameBase}/${frame}/${tile.dataset.endpoint}`;
  img.classList.remove("hidden");
  wrap.classList.remove("absent");
  img.src = url;
}

function update(force) {
  const app = window.__debugSimView;
  if (!app || !app.animationController || !meta) return;
  const now = performance.now();
  if (!force && now - lastUpdate < UPDATE_INTERVAL_MS) return;
  lastUpdate = now;

  const t = app.animationController.currentTime || 0;
  const index = frameIndexForTime(t);
  if (index === lastFrame && !force) return;
  lastFrame = index;

  const frame = meta.frames[index];
  els.frame.textContent = frame;
  els.time.textContent = `${(meta.times[index] || 0).toFixed(2)} s`;
  els.stamp.textContent = meta.abs_times ? meta.abs_times[index].toFixed(3) : "—";
  const valid = meta.pose_valid ? meta.pose_valid[index] : true;
  els.pose.textContent = meta.pose_source === "none"
    ? "none"
    : (valid ? meta.pose_source : `${meta.pose_source} (extrapolated)`);
  els.pose.style.color = valid ? "" : "#d8a640";

  for (const tile of els.tiles) setTile(tile, frame);
}

/* Switch between the solid-robot view and the point-cloud view.
 *
 * SimView keeps a single global "body visualization mode": every body is drawn
 * either as a mesh or as points, never a mix, and the mode is auto-selected
 * from the first body in the model (a box here, so "mesh"). A point-cloud body
 * is therefore invisible until the mode is "points". Both switches below go
 * through SimView's own GUI controllers, so its UI stays in sync and nothing
 * in SimView needs changing. */
function setView(view) {
  const app = window.__debugSimView;
  const controls = app && app.uiControls;
  if (!controls || !controls.findController) return;
  const mode = controls.findController("bodyVisualizationMode");
  const surface = controls.findController("showSurface");
  if (mode) mode.setValue(view === "points" ? "points" : "mesh");
  // The terrain surface and the cloud describe the same ground, so the surface
  // hides the points; drop it for the cloud view and bring it back after.
  if (surface) surface.setValue(view !== "points");
  for (const button of document.querySelectorAll("#dt-view .dt-seg button")) {
    button.classList.toggle("active", button.dataset.view === view);
  }
}

function tick() {
  update(false);
  requestAnimationFrame(tick);
}

function setupPanel() {
  els.collapse.addEventListener("click", () => {
    els.panel.classList.add("collapsed");
    els.expand.classList.add("show");
    // Lets SimView's own control panel reclaim the full width (see overlay.css).
    document.body.classList.add("dt-collapsed");
  });
  els.expand.addEventListener("click", () => {
    els.panel.classList.remove("collapsed");
    els.expand.classList.remove("show");
    document.body.classList.remove("dt-collapsed");
    update(true);
  });
  // Broken/absent per-frame artefacts must read as "absent", not as a broken
  // image icon: a scene legitimately has gaps in any per-frame directory.
  for (const tile of els.tiles) {
    const img = tile.querySelector("img");
    img.addEventListener("error", () => {
      tile.querySelector(".dt-imgwrap").classList.add("absent");
      img.classList.add("hidden");
    });
    img.addEventListener("load", () => {
      tile.querySelector(".dt-imgwrap").classList.remove("absent");
    });
  }
}

async function main() {
  setupPanel();
  try {
    meta = await loadMeta();
  } catch (error) {
    fail(`Could not build this window: ${error.message}`);
    return;
  }

  available = {};
  for (const [kind, frames] of Object.entries(meta.available || {})) {
    available[kind] = new Set(frames);
  }
  // A tile whose directory this scene lacks entirely is dimmed rather than
  // removed, so the panel still says what the scene does not have.
  for (const tile of els.tiles) {
    const kind = tile.dataset.kind;
    if (!(kind in available)) {
      tile.classList.add("unavailable");
      tile.querySelector(".dt-imgwrap").classList.add("absent");
      tile.querySelector(".dt-absent").textContent = "not present in this scene";
      tile.querySelector("img").classList.add("hidden");
    }
  }

  const build = meta.build || {};
  const parts = [];
  if (build.terrain_source) {
    parts.push(`terrain from ${build.terrain_source.replace("_", " ")}` +
      (build.terrain_resolution_m ? ` @ ${build.terrain_resolution_m} m` : ""));
  }
  if (build.height_maps && build.height_maps.used) parts.push(`${build.height_maps.used} height maps`);
  if (build.clouds && build.clouds.used) parts.push(`${build.clouds.used} clouds / ${build.clouds.points} pts`);
  if (build.seconds) parts.push(`built in ${build.seconds}s`);
  els.buildInfo.textContent = parts.join(" · ");

  // Only offer the cloud view when this window actually aggregated a cloud.
  if (build.clouds && build.clouds.used > 0) {
    const view = document.getElementById("dt-view");
    view.hidden = false;
    for (const button of view.querySelectorAll(".dt-seg button")) {
      button.addEventListener("click", () => setView(button.dataset.view));
    }
  }

  els.build.classList.add("done");
  startSimView();
  update(true);
  requestAnimationFrame(tick);
}

main();
