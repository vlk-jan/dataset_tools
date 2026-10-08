/*
 * Time-synced per-frame data panel (camera, height map, mask, cloud) beside
 * the SimView 3D view. Uses only SimView's embedding hooks: `__simviewStaticBase`,
 * `window.simview`, `simview:ready` / `simview:frame`. The server withholds
 * SimView's entry module until the bundle is built (`_defer_main_script`), so
 * this file loads it. Patches into SimView's objects live in simview_patch.js.
 */

import { ROBOT_BODY, installSimviewPatches, waitForSimView } from "./simview_patch.js";
import { frameField, makeButton, noteRequest, topicSelect } from "./notes_ui.js";

const cfg = window.__dtViewer || {};

const els = {
  build: document.getElementById("dt-build"),
  buildMsg: document.querySelector("#dt-build .dt-build-msg"),
  panel: document.getElementById("dt-panel"),
  edge: document.getElementById("dt-edge"),
  prevWindow: document.getElementById("dt-prev-window"),
  nextWindow: document.getElementById("dt-next-window"),
  view: document.getElementById("dt-view"),
  ghost: document.getElementById("dt-ghost"),
  legend: document.getElementById("dt-legend"),
  legendList: document.getElementById("dt-legend-list"),
  legendNow: document.getElementById("dt-legend-now"),
  frame: document.getElementById("dt-frame"),
  stamp: document.getElementById("dt-stamp"),
  attitude: document.getElementById("dt-attitude"),
  attNow: document.getElementById("dt-att-now"),
  attRoll: document.getElementById("dt-att-roll"),
  attPitch: document.getElementById("dt-att-pitch"),
  attErr: document.getElementById("dt-att-err"),
  attVerdict: document.getElementById("dt-att-verdict"),
  buildInfo: document.getElementById("dt-buildinfo"),
  noteList: document.getElementById("dt-notelist"),
  noteCount: document.getElementById("dt-notes-count"),
  noteForm: document.getElementById("dt-noteform"),
  noteFrom: document.getElementById("dt-note-from"),
  noteTo: document.getElementById("dt-note-to"),
  noteTopic: document.getElementById("dt-note-topic"),
  noteText: document.getElementById("dt-note-text"),
  noteMsg: document.getElementById("dt-note-msg"),
  tiles: Array.from(document.querySelectorAll(".dt-tile")),
};

let meta = null;
let available = {};
let lastFrame = -1;
let lastUpdate = 0;
/** Pending trailing update, so the last frame of a burst is still shown. */
let trailingUpdate = null;
/** The handles returned by installSimviewPatches, once SimView is up. */
let patches = null;
/** meta.attitude once the window has one: absent without gravity_align.npy. */
let attitude = null;

/** Legend rows by colour-mode name, for highlighting the active one. */
const legendRows = new Map();
let lastColorMode = null;

/** Minimum ms between panel updates: playback outruns the image fetches. */
const UPDATE_INTERVAL_MS = 110;

function fail(message) {
  els.build.classList.add("failed");
  els.build.classList.remove("done");
  els.buildMsg.textContent = message;
}

/** Fetch the window's metadata, which also triggers the (cached) build. */
async function loadMeta() {
  return (await noteRequest(cfg.metaUrl, "GET")).json();
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

// At most one batch of tile requests in flight; when it finishes, jump to the
// latest wanted frame. Keeps a slow link from lagging behind playback.
let tileWantFrame = null;
let tileShownFrame = null;
let tileInFlight = false;

function applyTiles(frame) {
  const pending = [];
  for (const tile of els.tiles) {
    const kind = tile.dataset.kind;
    const wrap = tile.querySelector(".dt-imgwrap");
    const img = tile.querySelector("img");
    const list = available[kind];
    // No file for this frame (or kind): don't fetch, it would 404.
    if (!list || !list.has(frame)) {
      wrap.classList.add("absent");
      img.removeAttribute("src");
      continue;
    }
    wrap.classList.remove("absent");
    const url = `${cfg.frameBase}/${frame}/${tile.dataset.endpoint}`;
    // What zoom.js opens for this tile: a larger render and/or the cell grid.
    const name = tile.querySelector("h3").firstChild.textContent.trim();
    img.alt = `${name} for frame ${frame}`;
    if (tile.dataset.zoomSize) img.dataset.zoom = `${url}?size=${tile.dataset.zoomSize}`;
    if (tile.dataset.probe !== undefined) img.dataset.grid = `${cfg.frameBase}/${frame}/grid.json`;
    // "Cloud on image" in the zoom dialog, only where the frame has a cloud.
    if (tile.dataset.cloud !== undefined) {
      if (available.clouds && available.clouds.has(frame)) img.dataset.cloud = `${cfg.frameBase}/${frame}/overlay.jpg?w=1920`;
      else delete img.dataset.cloud;
    }
    img.src = url;
    pending.push(img.decode());
  }
  // Settled, loaded or not: a failed image reads as absent (setupPanel).
  Promise.allSettled(pending).then(() => {
    tileShownFrame = frame;
    tileInFlight = false;
    pumpTiles(); // a newer frame may have been wanted while this batch loaded
  });
}

function pumpTiles() {
  if (tileInFlight || tileWantFrame === null || tileWantFrame === tileShownFrame) return;
  tileInFlight = true;
  applyTiles(tileWantFrame);
}

function requestTiles(frame) {
  // Collapsed panel fetches nothing; expanding calls update(true).
  if (els.panel && els.panel.classList.contains("collapsed")) return;
  tileWantFrame = frame;
  pumpTiles();
}

/* Mirror the playhead into ?frame=N, at most once a second: browsers throttle
 * faster history rewrites. */
const URL_WRITE_MS = 1000;
let lastUrlWrite = 0;
let urlTimer = null;

function writeFrameToUrl(frame) {
  clearTimeout(urlTimer);
  const write = () => {
    lastUrlWrite = performance.now();
    const url = new URL(window.location.href);
    url.searchParams.set("frame", String(frame));
    history.replaceState(history.state, "", url);
  };
  const wait = URL_WRITE_MS - (performance.now() - lastUrlWrite);
  if (wait <= 0) write();
  else urlTimer = setTimeout(write, wait);
}

function update(force) {
  const app = window.simview;
  if (!app || !app.animationController || !meta) return;
  const now = performance.now();
  const wait = UPDATE_INTERVAL_MS - (now - lastUpdate);
  if (!force && wait > 0) {
    if (trailingUpdate === null) {
      trailingUpdate = setTimeout(() => {
        trailingUpdate = null;
        update(false);
      }, wait);
    }
    return;
  }
  lastUpdate = now;

  // SimView's states are this window's frames, one for one (build.py).
  const index = app.animationController.getCurrentStateIndex();
  if (index < 0 || index >= meta.frames.length) return;
  if (index === lastFrame && !force) return;
  lastFrame = index;

  const frame = meta.frames[index];
  els.frame.textContent = frame;
  if (cfg.framePage) els.frame.href = `${cfg.framePage}/${frame}`;
  writeFrameToUrl(frame);
  els.stamp.textContent = meta.abs_times ? meta.abs_times[index].toFixed(3) : "-";
  showAttitude(index);

  requestTiles(frame);
}

/** Class for a tilt gap, on the bands the build shipped with the window. */
function attitudeClass(deg) {
  const bands = attitude.bands || { ok: 10, warn: 20 };
  if (deg < bands.ok) return "dt-att-ok";
  return deg < bands.warn ? "dt-att-warn" : "dt-att-bad";
}

function setAttitudeText(el, text, cls) {
  el.textContent = text;
  el.classList.remove("dt-att-ok", "dt-att-warn", "dt-att-bad");
  if (cls) el.classList.add(cls);
}

/* Window-wide GT-vs-IMU verdict: a single frame's gap can be large for an
 * honest reason (a jolt the interpolated GT smooths over). */
function setupAttitude() {
  attitude = meta.attitude || null;
  if (!attitude || !attitude.error || attitude.error.length === 0) return;
  const median = attitude.median_error;
  const cls = attitudeClass(median);
  const verdict = {
    "dt-att-ok": "The ground truth's own attitude tracks the IMU, so roll and pitch here can be read as real.",
    "dt-att-warn": "The ground truth's attitude is wandering from the IMU. Read roll and pitch with suspicion.",
    "dt-att-bad": "The ground truth's attitude has diverged from the IMU. This window's orientation is not trustworthy, whatever the motion looks like.",
  }[cls];
  els.attVerdict.textContent =
    `${verdict} Median gap ${median.toFixed(1)}\u00b0 over the window, 95th percentile ` +
    `${attitude.p95_error.toFixed(1)}\u00b0. The IMU is independent of the odometry.`;
  els.attitude.hidden = false;
  if (cls !== "dt-att-ok") els.attitude.open = true;
}

/** Per-frame half of the comparison, called from `update`. */
function showAttitude(index) {
  if (!attitude || index < 0 || index >= attitude.error.length) return;
  const gap = attitude.error[index];
  setAttitudeText(els.attNow, `${gap.toFixed(1)}\u00b0`, attitudeClass(gap));
  setAttitudeText(els.attErr, `${gap.toFixed(1)}\u00b0`, attitudeClass(gap));
  els.attRoll.textContent =
    `${attitude.gt_roll[index].toFixed(1)}\u00b0 GT / ${attitude.imu_roll[index].toFixed(1)}\u00b0 IMU`;
  els.attPitch.textContent =
    `${attitude.gt_pitch[index].toFixed(1)}\u00b0 GT / ${attitude.imu_pitch[index].toFixed(1)}\u00b0 IMU`;
}

/* Wire the ghost toggle to SimView, once it exists. */
function setupViewToggles() {
  if (!patches) return;
  if (patches.hasGhost) els.ghost.addEventListener("change", () => patches.setGhostVisible(els.ghost.checked));
  els.view.hidden = !patches.hasGhost;
}

/* Terrain colour-mode legend: SimView lists the modes by bare name, the
 * descriptions come from build.py via meta.json. */
function setupLegend() {
  const modes = (meta.build && meta.build.terrain_color_modes) || [];
  if (modes.length === 0) return;
  const frag = document.createDocumentFragment();
  for (const mode of modes) {
    const dt = document.createElement("dt");
    dt.textContent = mode.name;
    const dd = document.createElement("dd");
    dd.textContent = mode.description || "no description for this layer";
    const group = document.createElement("div");
    group.appendChild(dt);
    group.appendChild(dd);
    frag.appendChild(group);
    legendRows.set(mode.name, group);
  }
  els.legendList.appendChild(frag);
  els.legend.hidden = false;
}

/** Highlight whichever colour mode SimView is currently drawing. */
function syncLegendHighlight() {
  if (!patches || legendRows.size === 0) return;
  const mode = patches.terrainColorMode();
  if (mode === lastColorMode) return;
  lastColorMode = mode;
  els.legendNow.textContent = mode || "";
  for (const [name, row] of legendRows) row.classList.toggle("active", name === mode);
}

/* -------------------------------------------------------------------- notes
 *
 * Fetched from their own endpoint, not the window bundle: the bundle is cached
 * per window, so a note in it would be stale or force a rebuild.
 */

/** Seek the playhead to a frame of this window, if it holds it. */
function seekToFrame(frame) {
  const app = window.simview;
  if (!app || !app.animationController || !meta || !meta.frames) return;
  const local = meta.frames.indexOf(frame);
  if (local < 0 || !meta.times || local >= meta.times.length) return;
  // goToTime, not `currentTime =`, which is ignored until playback advances.
  app.animationController.goToTime(meta.times[local]);
  update(true);
}

/** Inclusive [lo, hi] from two fields; blanks fall back to defaults, ends ordered. */
function rangeOf(fromEl, toEl, fallbackLo, fallbackHi) {
  const from = parseInt(fromEl.value, 10);
  const to = parseInt(toEl.value, 10);
  let lo = Number.isFinite(from) ? from : fallbackLo;
  let hi = Number.isFinite(to) ? to : fallbackHi;
  if (lo > hi) [lo, hi] = [hi, lo];
  return [Math.max(0, lo), Math.max(0, hi)];
}

/** The compose form's range, falling back to the window's own edges. */
function readNoteRange() {
  return rangeOf(els.noteFrom, els.noteTo, cfg.frameStart, cfg.frameEnd);
}

function rangeText(lo, hi) {
  return lo === hi ? `frame ${lo}` : `frames ${lo}-${hi}`;
}

/** The dataset frame the playhead is on, or null before the first update. */
function frameShowingNow() {
  if (!meta || !meta.frames) return null;
  const frame = meta.frames[lastFrame];
  return frame === undefined ? null : frame;
}

/** Wire a `now` button to one end of a range; both ends are written back ordered. */
function wireNowButton(button, field, fromEl, toEl, fallbackLo, fallbackHi) {
  button.addEventListener("click", () => {
    const frame = frameShowingNow();
    if (frame === null) return;
    field.value = frame;
    const [lo, hi] = rangeOf(fromEl, toEl, fallbackLo, fallbackHi);
    fromEl.value = lo;
    toEl.value = hi;
  });
}

/** Only the notes touching this window: the rest are about other driving. */
function notesHere(all) {
  const lo = cfg.frameStart;
  const hi = cfg.frameEnd;
  return all.filter((n) => n.frame_start <= hi && lo <= n.frame_end);
}

function renderNotes(all) {
  const here = notesHere(all);
  els.noteCount.textContent = here.length
    ? `${here.length} on this window`
    : (all.length ? `none here, ${all.length} in the scene` : "");
  els.noteList.textContent = "";
  for (const note of here) els.noteList.appendChild(renderNote(note));
}

/** One note: its range (click to seek), topic and text, tools on the right. */
function renderNote(note) {
  const item = document.createElement("div");
  item.className = "dt-note";
  const main = document.createElement("div");
  main.className = "dt-notemain";
  const head = document.createElement("div");
  head.className = "dt-notehead";
  const range = document.createElement("button");
  range.type = "button";
  range.className = "dt-noterangebtn";
  range.textContent = note.label;
  range.title = "Seek to the first frame of this note";
  range.addEventListener("click", () => seekToFrame(note.frame_start));
  const topic = document.createElement("span");
  topic.className = "dt-notetopic";
  topic.textContent = note.topic;
  head.append(range, topic);
  const body = document.createElement("div");
  body.className = "dt-notebody";
  body.textContent = note.text;
  main.append(head, body);
  item.append(main);
  if ((cfg.notes || {}).writable) item.append(noteTools(item, note));
  return item;
}

/** The edit / delete icon pair. */
function noteTools(item, note) {
  const tools = document.createElement("span");
  tools.className = "dt-notetools";
  const edit = noteToolButton("✏️", "dt-notebtn-edit", "Edit this note's range, topic or text");
  edit.addEventListener("click", () => {
    const editor = noteEditor(note);
    item.replaceWith(editor);
    // After it is in the document: focus() on a detached element does nothing.
    editor.querySelector("textarea").focus();
  });
  const remove = noteToolButton("🗑️", "dt-notebtn-del", "Delete this note");
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Delete the note on ${note.label}?`)) return;
    try {
      await noteRequest(`${cfg.notesUrl}/${note.id}`, "DELETE");
      await loadNotes();
    } catch (error) {
      els.noteMsg.textContent = `could not delete: ${error.message}`;
    }
  });
  tools.append(edit, remove);
  return tools;
}

function noteToolButton(glyph, className, label) {
  const button = makeButton(glyph, `dt-notebtn ${className}`);
  button.title = label;
  button.setAttribute("aria-label", label);
  return button;
}

/** In-place editor shaped like the compose form; blank ends keep the note's own. */
function noteEditor(note) {
  const wrap = document.createElement("div");
  wrap.className = "dt-note dt-noteediting";

  const row = document.createElement("div");
  row.className = "dt-noterange";
  const from = rangeField(note.frame_start, "first frame of the range");
  const to = rangeField(note.frame_end, "last frame of the range");
  const fromField = document.createElement("label");
  fromField.className = "dt-rangefield";
  const fromNow = nowButton();
  fromField.append("from ", from, fromNow);
  const toField = document.createElement("label");
  toField.className = "dt-rangefield";
  const toNow = nowButton();
  toField.append("to ", to, toNow);
  wireNowButton(fromNow, from, from, to, note.frame_start, note.frame_end);
  wireNowButton(toNow, to, from, to, note.frame_start, note.frame_end);

  const topic = topicSelect((cfg.notes || {}).topics || [], note.topic);
  row.append(fromField, toField, topic);

  const text = document.createElement("textarea");
  text.rows = 2;
  text.value = note.text;
  text.setAttribute("aria-label", "note text");

  const actions = document.createElement("div");
  actions.className = "dt-noteactions";
  const msg = document.createElement("span");
  msg.className = "dt-hint";
  const save = makeButton("Save", "dt-rangebtn dt-save");
  const cancel = makeButton("Cancel", "dt-rangebtn");
  actions.append(msg, save, cancel);

  cancel.addEventListener("click", () => wrap.replaceWith(renderNote(note)));
  save.addEventListener("click", async () => {
    const body = text.value.trim();
    if (!body) {
      msg.textContent = "a note needs some text";
      return;
    }
    msg.textContent = "saving…";
    const [lo, hi] = rangeOf(from, to, note.frame_start, note.frame_end);
    try {
      await noteRequest(`${cfg.notesUrl}/${note.id}`, "PATCH", {
        frames: [lo, hi],
        topic: topic.value,
        text: body,
      });
      // Reload the list (the note may have left this window); `msg` is about
      // to be replaced, so confirm on the section's line.
      els.noteMsg.textContent = `saved on ${rangeText(lo, hi)}`;
      await loadNotes();
    } catch (error) {
      msg.textContent = `could not save: ${error.message}`;
    }
  });

  wrap.append(row, text, actions);
  return wrap;
}

function rangeField(value, label) {
  // Upper bound borrowed from the compose form, which the server renders.
  return frameField(value, label, els.noteFrom && els.noteFrom.max);
}

function nowButton() {
  const button = makeButton("now", "dt-nowbtn");
  button.title = "Set this end to the frame showing now";
  return button;
}

async function loadNotes() {
  const response = await fetch(cfg.notesUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status}`);
  const data = await response.json();
  renderNotes(data.notes || []);
}

/** Load the notes list and wire the compose form, if the root allows. */
async function setupNotes() {
  const status = cfg.notes || {};
  if (!status.enabled) return;

  try {
    await loadNotes();
  } catch (error) {
    els.noteCount.textContent = "unavailable";
    console.warn("could not load notes:", error);
  }

  if (!status.writable) return;

  for (const button of els.noteForm.querySelectorAll(".dt-nowbtn[data-act]")) {
    const field = button.dataset.act === "from" ? els.noteFrom : els.noteTo;
    wireNowButton(button, field, els.noteFrom, els.noteTo, cfg.frameStart, cfg.frameEnd);
  }

  els.noteForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = els.noteText.value.trim();
    if (!text) {
      els.noteMsg.textContent = "A note needs some text.";
      return;
    }
    els.noteMsg.textContent = "saving…";
    const frames = readNoteRange();
    try {
      await noteRequest(cfg.notesUrl, "POST", { frames, topic: els.noteTopic.value, text });
      els.noteText.value = "";
      els.noteMsg.textContent = `saved on ${rangeText(...frames)}`;
      await loadNotes();
    } catch (error) {
      els.noteMsg.textContent = `could not save: ${error.message}`;
    }
  });
}

/**
 * Seek to ?frame=N (via goToTime) and centre the camera on the robot.
 * `focusBody` is false until the robot has a position, so it is retried.
 * Failure just leaves the stock view.
 */
async function openInitialView() {
  if (!meta) return;
  const target = cfg.initialFrame;
  const local =
    target === null || target === undefined || !meta.frames
      ? -1
      : meta.frames.indexOf(target);
  const seekTo =
    local >= 0 && meta.times && local < meta.times.length ? meta.times[local] : null;

  let app;
  try {
    app = await waitForSimView(60000);
  } catch {
    return;
  }
  if (seekTo !== null) app.animationController.goToTime(seekTo);

  const deadline = performance.now() + 60000;
  let centered = false;
  const attempt = () => {
    if (centered || app.focusBody(ROBOT_BODY)) {
      centered = true;
      // After centring: the carried camera is an offset from the robot.
      if (restoreCarriedView(app)) {
        update(true);
        return;
      }
    }
    if (performance.now() < deadline) requestAnimationFrame(attempt);
  };
  attempt();
}

/** Is this node a form field or editable content? */
function editable(node) {
  return Boolean(node?.matches?.("input, textarea, select") || node?.isContentEditable);
}

/** The subset of `editable` one types text into. */
function typing(node) {
  return Boolean(
    node?.matches?.("textarea, input:not([type=checkbox], [type=radio], [type=range], [type=button], [type=submit])") ||
      node?.isContentEditable,
  );
}

// SimView's orbit camera binds arrow keys on `window` (bubble phase), so stop
// key events from panel fields from panning the view.
function claimKeyboardForFields() {
  for (const type of ["keydown", "keyup", "keypress"]) {
    els.panel.addEventListener(type, (event) => {
      if (editable(event.target)) event.stopPropagation();
    });
  }
}

/**
 * Space is play/pause whatever has focus (except text fields). Capture phase,
 * because a focused button would re-activate and lil-gui swallows keydown;
 * keyup is prevented too, since buttons fire on it.
 */
function setupSpaceIsPlay() {
  for (const type of ["keydown", "keyup"]) {
    document.addEventListener(
      type,
      (event) => {
        if (event.key !== " " || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
        if (typing(event.target)) return;
        event.preventDefault();
        if (type !== "keydown") return;
        event.stopPropagation();
        document.querySelector("button.sv-play")?.click();
        event.target.blur?.();
      },
      true,
    );
  }
}

/** `[` / `]` page to the previous / next window (SimView's unused episode keys). */
function setupWindowNav() {
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (editable(event.target) || editable(document.activeElement)) return;
    const link = event.key === "[" ? els.prevWindow : event.key === "]" ? els.nextWindow : null;
    // Absent at either end of the scene.
    if (!link) return;
    event.preventDefault();
    link.click();
  });
  for (const link of [els.prevWindow, els.nextWindow]) {
    if (link) link.addEventListener("click", saveCarriedView);
  }
}

/* View setup carried across a `[` / `]` page load, read once (sessionStorage).
 * The camera travels as an offset from the orbit target, since each window has
 * its own coordinates. Play state is not carried. */
const CARRY_KEY = "dt.carriedView";

function saveCarriedView() {
  const app = window.simview;
  if (!app || !app.scene) return;
  try {
    // SimView's view state minus time and absolute camera (kept: FOV).
    const { time, camera, ...view } = app.getViewState();
    const state = {
      view: { ...view, camera: { fov: camera.fov } },
      offset: camera.position.clone().sub(camera.target).toArray(),
      ghost: els.ghost ? els.ghost.checked : undefined,
    };
    sessionStorage.setItem(CARRY_KEY, JSON.stringify(state));
  } catch (error) {
    console.warn("view setup not carried over:", error);
  }
}

/** Apply a carried setup, once. False while SimView's panel is not built yet. */
function restoreCarriedView(app) {
  let state;
  try {
    state = JSON.parse(sessionStorage.getItem(CARRY_KEY) || "null");
  } catch {
    state = null;
  }
  if (!state) return true;
  // The panel's own ghost switch is wired once the patches are in.
  if (!app.uiControls || !patches) return false;
  try {
    sessionStorage.removeItem(CARRY_KEY);
  } catch {
    /* read-once is a nicety: a stuck key only re-applies the same setup */
  }
  try {
    // Through SimView's controls, so its panel reflects what is applied.
    if (state.view) app.setViewState(state.view);
    if (Array.isArray(state.offset) && state.offset.length === 3) {
      const { camera, controls } = app.scene;
      camera.position.copy(controls.target).add(camera.position.clone().fromArray(state.offset));
      controls.update();
    }
    const el = els.ghost;
    if (el && !el.closest("[hidden]") && typeof state.ghost === "boolean" && el.checked !== state.ghost) {
      el.checked = state.ghost;
      el.dispatchEvent(new Event("change"));
    }
  } catch (error) {
    console.warn("carried view setup only partly applied:", error);
  }
  return true;
}

function setupPanel() {
  claimKeyboardForFields();
  setupSpaceIsPlay();
  setupWindowNav();
  els.edge.addEventListener("click", () => {
    const collapsed = els.panel.classList.toggle("collapsed");
    els.edge.setAttribute("aria-expanded", String(!collapsed));
    els.edge.title = collapsed ? "Show this panel" : "Hide this panel";
    // Lets SimView's control panel reclaim the width (overlay.css).
    document.body.classList.toggle("dt-collapsed", collapsed);
    if (!collapsed) update(true);
  });
  // A missing per-frame file reads as "absent", not a broken image icon.
  for (const tile of els.tiles) {
    const img = tile.querySelector("img");
    const wrap = tile.querySelector(".dt-imgwrap");
    img.addEventListener("error", () => wrap.classList.add("absent"));
    img.addEventListener("load", () => wrap.classList.remove("absent"));
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
  // A kind the scene lacks is dimmed, so the panel still says it's missing.
  for (const tile of els.tiles) {
    const kind = tile.dataset.kind;
    if (!(kind in available)) {
      // Optional tiles (rarely-present products) are hidden instead.
      if (tile.dataset.optional !== undefined) {
        tile.hidden = true;
        continue;
      }
      tile.classList.add("unavailable");
      tile.querySelector(".dt-imgwrap").classList.add("absent");
      tile.querySelector(".dt-absent").textContent = "not present in this scene";
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
  if (build.missing_files && build.missing_files.length) parts.push(`absent: ${build.missing_files.join(", ")}`);
  els.buildInfo.textContent = parts.join(" · ");

  setupLegend();
  setupAttitude();
  // Never fatal.
  setupNotes().catch((error) => console.warn("notes are unavailable:", error));

  els.build.classList.add("done");
  // Before SimView exists: its first frame event must find the panel listening.
  window.addEventListener("simview:frame", () => update(false));
  // Colour mode changes via SimView's dropdown or the hash; read it a frame
  // later, after SimView has applied it.
  const syncLegendSoon = () => requestAnimationFrame(syncLegendHighlight);
  document.addEventListener("change", syncLegendSoon);
  window.addEventListener("hashchange", syncLegendSoon);
  startSimView();
  openInitialView();

  // Patches need SimView's UI built; a failure must not take the panel down.
  try {
    patches = installSimviewPatches(await waitForSimView());
    setupViewToggles();
    syncLegendHighlight();
  } catch (error) {
    console.warn("SimView patches were not applied:", error);
  }
}

main();
