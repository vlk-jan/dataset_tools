/*
 * Time-synced per-frame data panel for the SimView viewer.
 *
 * SimView renders geometry; a dataset frame also carries a camera image, a BEV
 * height map, its validity mask and a raw cloud, none of which are. This shows
 * those next to the 3D view, on the frame the playhead is at.
 *
 * It modifies SimView not at all, on two stock affordances:
 *   1. `window.__simviewStaticBase` points its fetches at our bundle endpoints.
 *   2. `window.__debugSimView` is the running instance. We read its playhead
 *      every frame and never write it -- the one seek we ask for goes through
 *      `goToTime` (see `openInitialView`).
 *
 * The server withholds SimView's entry module until the bundle is built (see
 * `_defer_main_script`), so this file loads it once its data is there.
 * Corrections that reach into SimView's objects live in `simview_patch.js`.
 */

import { centerOnRobot, installSimviewPatches, waitForSimView } from "./simview_patch.js";

const cfg = window.__dtViewer || {};

const els = {
  build: document.getElementById("dt-build"),
  buildMsg: document.querySelector("#dt-build .dt-build-msg"),
  panel: document.getElementById("dt-panel"),
  edge: document.getElementById("dt-edge"),
  prevWindow: document.getElementById("dt-prev-window"),
  nextWindow: document.getElementById("dt-next-window"),
  view: document.getElementById("dt-view"),
  cloudRow: document.getElementById("dt-cloud-row"),
  cloud: document.getElementById("dt-cloud"),
  ghostRow: document.getElementById("dt-ghost-row"),
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
  notes: document.getElementById("dt-notes"),
  noteList: document.getElementById("dt-notelist"),
  noteCount: document.getElementById("dt-notes-count"),
  noteForm: document.getElementById("dt-noteform"),
  noteFrom: document.getElementById("dt-note-from"),
  noteTo: document.getElementById("dt-note-to"),
  noteTopic: document.getElementById("dt-note-topic"),
  noteText: document.getElementById("dt-note-text"),
  noteMsg: document.getElementById("dt-note-msg"),
  notesWhy: document.getElementById("dt-notes-why"),
  tiles: Array.from(document.querySelectorAll(".dt-tile")),
};

let meta = null;
let available = {};
let lastFrame = -1;
let lastUpdate = 0;
/** The handles returned by installSimviewPatches, once SimView is up. */
let patches = null;
/** meta.attitude once the window has one: absent without gravity_align.npy. */
let attitude = null;

/** Legend rows by colour-mode name, for highlighting the active one. */
const legendRows = new Map();
let lastColorMode = null;

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

// Bandwidth-adaptive per-frame tiles. Four images per frame, one request each,
// served off contended storage: firing them every frame floods a slow link and
// the panel lags playback until it drains ("it jumps to the current frame when
// I pause"). At most ONE batch is in flight, and on finishing it jumps to the
// latest wanted frame -- skipping intermediates, so the panel stays near-live
// on a fast link and always lands on the frame you pause on. Frame numbers and
// timestamps still update every tick, so the readout is never behind.
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
    // `available` lists, per kind, exactly which frames have a file: a kind
    // missing entirely (no height maps in this scene) has no list; one that has
    // a list may still lack this frame (directories built in flight). Either
    // way there is nothing to fetch -- a request would 404 per frame.
    if (!list || !list.has(frame)) {
      wrap.classList.add("absent");
      img.classList.add("hidden");
      img.removeAttribute("src");
      continue;
    }
    img.classList.remove("hidden");
    wrap.classList.remove("absent");
    pending.push({ img, url: `${cfg.frameBase}/${frame}/${tile.dataset.endpoint}` });
  }
  if (pending.length === 0) {
    tileShownFrame = frame;
    tileInFlight = false;
    pumpTiles();
    return;
  }
  let remaining = pending.length;
  const done = () => {
    remaining -= 1;
    if (remaining > 0) return;
    tileShownFrame = frame;
    tileInFlight = false;
    pumpTiles(); // a newer frame may have been wanted while this batch loaded
  };
  for (const { img, url } of pending) {
    img.onload = () => { img.onload = img.onerror = null; done(); };
    img.onerror = () => { img.onload = img.onerror = null; done(); };
    img.src = url;
  }
}

function pumpTiles() {
  if (tileInFlight || tileWantFrame === null || tileWantFrame === tileShownFrame) return;
  tileInFlight = true;
  applyTiles(tileWantFrame);
}

function requestTiles(frame) {
  tileWantFrame = frame;
  pumpTiles();
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
  els.stamp.textContent = meta.abs_times ? meta.abs_times[index].toFixed(3) : "—";
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

/* The window's standing verdict, from the whole window rather than a frame.
 *
 * A single frame's gap can be large for an honest reason -- a jolt the IMU
 * catches and the interpolated ground truth smooths over -- so what says
 * whether the ground truth is usable is the level it holds across the window.
 */
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

/* Wire the 3D-view toggles to SimView, once it exists.
 *
 * These were once one either/or switch that pushed SimView's global mode to
 * "points", which made the robot vanish whenever the cloud was asked for (a box
 * has no point representation). Cloud and ghost are now independent of that
 * mode and of each other; see simview_patch.js.
 */
function setupViewToggles() {
  if (!patches) return;
  const view = els.view;
  let any = false;

  // `hasCloud` is false on a SimView new enough to own this control itself --
  // it shows point clouds through its own "Show Point Clouds" checkbox, and a
  // second one here could only contradict it.
  if (patches.hasCloud) {
    any = true;
    els.cloudRow.hidden = false;
    els.cloud.addEventListener("change", () => {
      patches.setCloudVisible(els.cloud.checked);
      // The terrain surface and the aggregated cloud describe the same ground,
      // and the surface is drawn over the points.
      patches.setSurfaceVisible(!els.cloud.checked);
    });
  }
  if (patches.hasGhost) {
    any = true;
    els.ghostRow.hidden = false;
    els.ghost.addEventListener("change", () => patches.setGhostVisible(els.ghost.checked));
  }
  view.hidden = !any;
}

/* Render the terrain colour-mode legend from the build metadata.
 *
 * SimView's "Color Mode" dropdown lists the scene's properties by bare name and
 * explains none of them. The names are ours (build.py), so the descriptions
 * travel with them in meta.json and are shown here rather than guessed at.
 */
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
 * Notes are written here because here is where a bad stretch is actually
 * noticed -- scrubbing a segment and seeing the ground truth walk off the
 * terrain. The two range buttons pin either end to the frame showing now, so
 * marking "frames 1204-1388 have wrong odometry" is two clicks and a sentence.
 *
 * They are fetched from their own endpoint, never from the window bundle: a
 * bundle URL carries a content stamp and its meta.json is cached per window on
 * disk, so a note folded in there would be invisible until the next code
 * change, or would rebuild the window every time someone typed one.
 */

/** Seek the playhead to a frame of this window, if it holds it. */
function seekToFrame(frame) {
  const app = window.__debugSimView;
  if (!app || !app.animationController || !meta || !meta.frames) return;
  const local = meta.frames.indexOf(frame);
  if (local < 0 || !meta.times || local >= meta.times.length) return;
  // goToTime, not `currentTime =`: nothing reads that field until playback
  // advances it, so an assignment leaves the view where it was (openInitialView).
  app.animationController.goToTime(meta.times[local]);
  update(true);
}

/** One note mutation. Throws carrying the server's own reason, which is the
 *  useful half of a refusal ("this root is not writable", not "403"). */
async function noteRequest(url, method, body) {
  const init = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const parsed = await response.json();
      if (parsed && parsed.detail) detail = parsed.detail;
    } catch (e) { /* the status alone is enough */ }
    throw new Error(detail);
  }
  return response;
}

/**
 * The range a pair of fields names, both ends inclusive.
 *
 * Read out of the fields rather than kept in a variable beside them, so a
 * number typed by hand and one pinned with `now` are the same thing. A blank
 * or unparseable field falls back to the given default -- a half-filled form
 * still means something -- and the ends are ordered, so typing them the other
 * way round is not an error.
 */
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
  return lo === hi ? `frame ${lo}` : `frames ${lo}–${hi}`;
}

function noteRangeText() {
  return rangeText(...readNoteRange());
}

function setNoteRange(lo, hi) {
  els.noteFrom.value = lo;
  els.noteTo.value = hi;
}

/** The dataset frame the playhead is on, or null before the first update. */
function frameShowingNow() {
  if (!meta || !meta.frames) return null;
  const frame = meta.frames[lastFrame];
  return frame === undefined ? null : frame;
}

/**
 * Wire a `now` button to one end of a range.
 *
 * Pinning one end past the other means the two were meant the other way round,
 * so both are written back ordered rather than the click being refused.
 */
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

/**
 * One note as it reads: its range (which seeks), its topic, its text.
 *
 * The note is a row -- what it says, then the buttons that change it in a
 * column down its right edge. Spelled `edit` and `delete` on the head they
 * took a third of a panel that is only as wide as the sidebar, and the range
 * and topic were squeezed into wrapping; two icons stacked beside the note
 * cost the text nothing.
 */
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
  // A read-only root shows its notes and no way to change them, the same way
  // it shows the compose form's reason instead of the form.
  if ((cfg.notes || {}).writable) item.append(noteTools(item, note));
  return item;
}

/**
 * The edit / delete pair, stacked down the note's right edge.
 *
 * Icons, so each is one glyph wide, and coloured to say which is which without
 * being read: amber for the one that opens a form, red for the one that throws
 * the note away. Neither is only its colour -- the pen and the bin carry the
 * meaning, and both carry a title and an aria-label besides.
 */
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
  const button = document.createElement("button");
  button.type = "button";
  button.className = `dt-notebtn ${className}`;
  button.textContent = glyph;
  button.title = label;
  // The glyph is decoration to a screen reader, which reads the label instead.
  button.setAttribute("aria-label", label);
  return button;
}

/**
 * A note opened for editing, in place, over every field it has.
 *
 * The same shape as the compose form -- two range fields with a `now` button
 * each, a topic and the text -- because a note is corrected where and how it
 * was written: scrubbing to the frame the bad stretch really starts on and
 * pinning that end to it is the whole point of writing notes from here.
 *
 * Its own range falls back to the note's stored ends rather than the window's,
 * so clearing a field and saving leaves that end where it was.
 */
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

  const topic = document.createElement("select");
  topic.setAttribute("aria-label", "topic");
  const topics = ((cfg.notes || {}).topics || []).slice();
  // A topic the server no longer offers stays on the note: the store renders
  // an unknown one verbatim, so editing the text must not quietly retag it.
  if (note.topic && !topics.includes(note.topic)) topics.unshift(note.topic);
  for (const name of topics) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    option.selected = name === note.topic;
    topic.appendChild(option);
  }
  row.append(fromField, toField, topic);

  const text = document.createElement("textarea");
  text.rows = 2;
  text.value = note.text;
  text.setAttribute("aria-label", "note text");

  const actions = document.createElement("div");
  actions.className = "dt-noteactions";
  const msg = document.createElement("span");
  msg.className = "dt-hint";
  const save = document.createElement("button");
  save.type = "button";
  save.className = "dt-rangebtn dt-save";
  save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "dt-rangebtn";
  cancel.textContent = "Cancel";
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
      // Reload rather than patch this one row: a range moved off this window
      // takes the note off the list with it, and the list is the only thing
      // that knows whether it still belongs here. Which is also why the
      // confirmation goes to the section's own line and not to `msg` -- the
      // editor holding `msg` is about to be replaced, possibly by nothing.
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
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  // The scene's last frame, borrowed from the compose form rather than
  // restated: the server renders it there from the probe, and one bound that
  // is wrong in one place only is worse than none.
  if (els.noteFrom && els.noteFrom.max) input.max = els.noteFrom.max;
  input.inputMode = "numeric";
  input.value = String(value);
  input.setAttribute("aria-label", label);
  return input;
}

function nowButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "dt-nowbtn";
  button.textContent = "now";
  button.title = "Set this end to the frame showing now";
  return button;
}

async function loadNotes() {
  const response = await fetch(cfg.notesUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status}`);
  const data = await response.json();
  renderNotes(data.notes || []);
}

/**
 * Wire up the notes section.
 *
 * Never fatal: a root whose notes are disabled or read-only still gets the
 * panel and the 3D view, with the section saying which -- the same rule the
 * page follows for an absent artefact, where "this scene has none" is shown
 * rather than implied by an empty space.
 */
async function setupNotes() {
  const status = cfg.notes || {};
  if (!status.enabled) {
    els.notes.hidden = true;
    return;
  }
  // Opens as the whole window: the common case is "this whole segment is
  // wrong", and either end is then narrowed by hand or by `now`.
  setNoteRange(cfg.frameStart, cfg.frameEnd);

  try {
    await loadNotes();
  } catch (error) {
    els.noteCount.textContent = "unavailable";
    console.warn("could not load notes:", error);
  }

  if (!status.writable) {
    els.notesWhy.textContent = `Read-only here: ${status.reason}`;
    return;
  }
  els.notesWhy.textContent =
    "Saved to " + status.dir + ". Both ends of the range are included.";

  for (const topic of status.topics || []) {
    const option = document.createElement("option");
    option.value = option.textContent = topic;
    els.noteTopic.appendChild(option);
  }
  els.noteForm.hidden = false;

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
    try {
      await noteRequest(cfg.notesUrl, "POST", {
        frames: readNoteRange(),
        topic: els.noteTopic.value,
        text,
      });
      const saved = noteRangeText();
      els.noteText.value = "";
      els.noteMsg.textContent = `saved on ${saved}`;
      await loadNotes();
    } catch (error) {
      els.noteMsg.textContent = `could not save: ${error.message}`;
    }
  });
}

/**
 * Open the window where it should open: on the right frame, robot in shot.
 *
 * Both are the wrapper's to do -- SimView has no start-time URL parameter and
 * never frames the scene it loaded.
 *
 *   1. Seek via `goToTime` when reached from a frame's page (`?frame=N`).
 *      Assigning `currentTime` instead looks equivalent and is not: nothing
 *      reads that field until playback advances it, so the view stayed on the
 *      first frame until play was pressed.
 *   2. Centre the camera on the robot (`centerOnRobot`) -- a window does not
 *      always start it at the origin, the only place the stock camera looks.
 *
 * Failure here (no such frame, SimView slow) leaves the window at its start
 * under the stock camera: normal behaviour, never an error.
 */
function openInitialView() {
  if (!meta) return;
  const target = cfg.initialFrame;
  const local =
    target === null || target === undefined || !meta.frames
      ? -1
      : meta.frames.indexOf(target);
  const seekTo =
    local >= 0 && meta.times && local < meta.times.length ? meta.times[local] : null;

  const deadline = performance.now() + 60000;
  let seeked = false;
  let centered = false;
  const attempt = () => {
    const app = window.__debugSimView;
    if (app && app.animationController && app.store && app.store.length > 0) {
      if (seekTo !== null && !seeked) {
        app.animationController.goToTime(seekTo);
        seeked = true;
      }
      // The camera can only be aimed once the robot has a resolved position,
      // which the seek above (or SimView's own initial draw) is what produces.
      // Keep waiting if the bodies are not built yet.
      if (centered || centerOnRobot(app)) {
        centered = true;
        // After centring: the carried camera is an offset from the robot.
        if (restoreCarriedView(app)) {
          update(true);
          return;
        }
      }
    }
    if (performance.now() < deadline) requestAnimationFrame(attempt);
  };
  requestAnimationFrame(attempt);
}

function tick() {
  // Every branch below is wrapped so that a transient throw -- SimView briefly
  // mid-rebuild when patches.enforce() pokes it, an unexpected store shape --
  // cannot kill the rAF chain. Before this guard, a single exception anywhere
  // in the loop left `requestAnimationFrame(tick)` unreached and the panel
  // frozen on whatever frame it last showed until a full reload (the "side
  // panel went static" report, 2026-09-02). The loop must outlive one bad frame.
  try {
    update(false);
  } catch (error) {
    if (!tick._warned) { console.warn("panel update skipped a frame:", error); tick._warned = true; }
  }
  if (patches) {
    try {
      // SimView recomputes body visibility from its global mode whenever the
      // visible batch set or that mode changes, so our per-body wishes have to
      // be re-asserted rather than set once. Writes nothing when already right.
      patches.enforce();
      syncLegendHighlight();
    } catch (error) {
      if (!tick._patchWarned) { console.warn("SimView patch enforce skipped a frame:", error); tick._patchWarned = true; }
    }
  }
  requestAnimationFrame(tick);
}

/**
 * A focused field in the panel owns the keyboard.
 *
 * SimView binds its shortcuts to `document` and `window` with no check on what
 * has focus: `PlaybackControls` (`keydownListener`) has none at all, so typing
 * a space into a note toggled playback and a left arrow stepped the playhead
 * instead of moving the caret; `Controls` checks `activeElement.tagName !==
 * "INPUT"`, which a textarea is not. Both listen in the bubble phase, so
 * stopping the event on our own panel is enough to keep it from reaching them,
 * and the field still gets it -- the target is reached first.
 *
 * This is not a version-gated stopgap like the ones in `simview_patch.js`: it
 * is not reaching into SimView at all, and "the thing you are typing into gets
 * the keystroke" stays right whatever upstream does.
 */
function claimKeyboardForFields() {
  const editable = (node) =>
    node &&
    (node.tagName === "INPUT" ||
      node.tagName === "TEXTAREA" ||
      node.tagName === "SELECT" ||
      node.isContentEditable);
  for (const type of ["keydown", "keyup", "keypress"]) {
    els.panel.addEventListener(type, (event) => {
      if (editable(event.target)) event.stopPropagation();
    });
  }
}

/**
 * `[` / `]` page to the previous / next window of this scene.
 *
 * The two keys are SimView's own episode shortcuts, which do nothing here --
 * a dataset scene declares no episodes -- so they are free, and they already
 * mean "the segment either side of this one" to anyone who knows SimView.
 * A key pressed while typing is left alone (the panel's fields stop their own
 * events, but this listens on the document, which a note textarea does not
 * cover), and so is one carrying a modifier, which is a browser shortcut.
 */
function setupWindowNav() {
  const editable = (node) =>
    node &&
    (node.tagName === "INPUT" ||
      node.tagName === "TEXTAREA" ||
      node.tagName === "SELECT" ||
      node.isContentEditable);
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (editable(event.target) || editable(document.activeElement)) return;
    const link = event.key === "[" ? els.prevWindow : event.key === "]" ? els.nextWindow : null;
    // Absent at either end of the scene: the template renders a disabled
    // <span> with no id rather than a link that goes nowhere.
    if (!link) return;
    event.preventDefault();
    link.click();
  });
  for (const link of [els.prevWindow, els.nextWindow]) {
    if (link) link.addEventListener("click", saveCarriedView);
  }
}

/* The view setup carried from one window to the next by `[` / `]`.
 *
 * Paging is a full page load, so without this every jump reset the camera,
 * the tracked body, the playback speed and every toggle to SimView's defaults.
 * Play state is deliberately not carried: a new window always opens paused.
 * Saved only on a jump and read once, so an ordinary load still opens stock.
 * sessionStorage keeps it to this tab. The camera travels as its offset from
 * the orbit target: each window is its own world, so absolute coordinates
 * mean nothing in the next one, but "this far behind the robot" does.
 */
const CARRY_KEY = "dt.carriedView";

function guiControllers(app) {
  const ui = app.uiControls;
  if (!ui || !ui.gui) return [];
  // The "Scene Info" folder is per-window metadata shown as read-only fields.
  const skip = new Set(ui.metadataFolder ? ui.metadataFolder.controllersRecursive() : []);
  return ui.gui.controllersRecursive().filter((c) => !skip.has(c) && typeof c.getValue() !== "function");
}

function saveCarriedView() {
  const app = window.__debugSimView;
  if (!app || !app.scene) return;
  try {
    const { camera, controls } = app.scene;
    const anim = app.animationController;
    const state = {
      gui: Object.fromEntries(guiControllers(app).map((c) => [c.property, c.getValue()])),
      speed: anim ? anim.playbackSpeed : undefined,
      offset: camera.position.clone().sub(controls.target).toArray(),
      cloud: els.cloud ? els.cloud.checked : undefined,
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
  // The panel's own cloud/ghost switches are wired once the patches are in.
  if (!app.uiControls || !app.uiControls.gui || !patches) return false;
  try {
    sessionStorage.removeItem(CARRY_KEY);
  } catch {
    /* read-once is a nicety: a stuck key only re-applies the same setup */
  }
  try {
    // setValue runs each control's own onChange, so SimView's state follows.
    // A control this window lacks, or an option it does not offer, is skipped.
    for (const c of guiControllers(app)) {
      if (!(c.property in (state.gui || {}))) continue;
      const value = state.gui[c.property];
      if (typeof value !== typeof c.getValue()) continue;
      const select = c.domElement.querySelector("select");
      if (select && ![...select.options].some((o) => o.textContent === String(value) || o.value === String(value))) continue;
      if (value !== c.getValue()) c.setValue(value);
    }
    const anim = app.animationController;
    if (anim && Number.isFinite(state.speed)) {
      anim.setSpeed(state.speed);
      const select = anim.playbackControls && anim.playbackControls.speedSelect;
      if (select) select.value = String(state.speed);
    }
    if (Array.isArray(state.offset) && state.offset.length === 3) {
      const { camera, controls } = app.scene;
      camera.position.copy(controls.target).add(camera.position.clone().fromArray(state.offset));
      controls.update();
    }
    for (const [el, want] of [[els.cloud, state.cloud], [els.ghost, state.ghost]]) {
      if (el && !el.closest("[hidden]") && typeof want === "boolean" && el.checked !== want) {
        el.checked = want;
        el.dispatchEvent(new Event("change"));
      }
    }
  } catch (error) {
    console.warn("carried view setup only partly applied:", error);
  }
  return true;
}

function setupPanel() {
  claimKeyboardForFields();
  setupWindowNav();
  // A single tab on the panel's own edge, so it slides out with the panel and
  // is always reachable at the middle of the window edge -- the old pair of
  // buttons put the "show again" one under SimView's control panel.
  els.edge.addEventListener("click", () => {
    const collapsed = els.panel.classList.toggle("collapsed");
    els.edge.setAttribute("aria-expanded", String(!collapsed));
    els.edge.title = collapsed ? "Show this panel" : "Hide this panel";
    // Lets SimView's own control panel reclaim the full width (see overlay.css).
    document.body.classList.toggle("dt-collapsed", collapsed);
    if (!collapsed) update(true);
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
      // A tile marked optional is for a product most roots never carry
      // (DRIFT's driven-similarity maps); saying "not present" would be the
      // panel's own noise, so it is removed instead of dimmed.
      if (tile.dataset.optional !== undefined) {
        tile.classList.add("hidden");
        continue;
      }
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

  setupLegend();
  setupAttitude();
  // Never fatal: the panel and the 3D view are the point of this page, and
  // notes are a thing you can add on top of them.
  setupNotes().catch((error) => console.warn("notes are unavailable:", error));

  els.build.classList.add("done");
  startSimView();
  openInitialView();
  update(true);
  requestAnimationFrame(tick);

  // Corrections to SimView's own UI can only be applied once it has built it.
  // A failure here must never take the panel down with it: the per-frame data
  // is the point of this page, the patches are a better 3D view on top of it.
  try {
    patches = installSimviewPatches(await waitForSimView(), cfg.simviewVersion);
    setupViewToggles();
    syncLegendHighlight();
  } catch (error) {
    console.warn("SimView patches were not applied:", error);
  }
}

main();
