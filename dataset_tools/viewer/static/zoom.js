// The frame page's click-to-zoom. Any `.zoomable` picture opens big in the
// dialog: `data-zoom` names a larger render, else the shown image is asked for
// at full width (so the cloud-on-image toggle zooms whatever it is showing).
// A picture with `data-grid` is one of the 128x128 height-map grids, and its
// zoom is a probe: hover reads the cell under the pointer, a click pins it
// (a click on the pinned cell lets it go), arrow keys walk the pin. The camera
// image instead magnifies further in the dialog: wheel, drag, double click.
(() => {
  const dlg = document.getElementById("zoom");
  if (!dlg) return;
  const img = dlg.querySelector(".zoomstage img");
  const cell = dlg.querySelector(".zoomcell");
  const read = dlg.querySelector(".zoomread");
  const hint = dlg.querySelector(".zoomhint");
  const title = dlg.querySelector(".zoomtitle");
  const gridHint = hint.textContent;
  // One fetch per page: all three grid tiles share one grid.json.
  const grids = new Map();
  let grid = null;
  let pinned = null;

  // Two decimals, padded to a fixed width (the readout is monospace and keeps
  // its spaces): a line that changes length as the pointer moves reflows the
  // dialog, and the picture shakes.
  const fixed = (v, width) => v.toFixed(2).padStart(width);
  const side = (v, pos, neg) => `${fixed(Math.abs(v), 5)} m ${v >= 0 ? pos : neg}`;
  const num = (v) => (v == null ? "no value" : fixed(v, 5));

  // Display cell (r, c) -> file cell (i, j): the tiles are drawn flipped in
  // both axes so forward is up and left is left (render._bev_to_display).
  const show = (r, c) => {
    const { rows, cols, res, x0, layers } = grid;
    const i = rows - 1 - r;
    const j = cols - 1 - c;
    const k = i * cols + j;
    const x = x0 + res * (i + 0.5);
    const y = x0 + res * (j + 0.5);
    Object.assign(cell.style, {
      left: `${(100 * c) / cols}%`, top: `${(100 * r) / rows}%`,
      width: `${100 / cols}%`, height: `${100 / rows}%`,
    });
    cell.hidden = false;
    const lines = [
      `cell [${String(i).padStart(3)}, ${String(j).padStart(3)}] · ${side(x, "ahead", "behind")}, ` +
        `${side(y, "left", "right")} · ${fixed(Math.hypot(x, y), 5)} m away`,
    ];
    const what = [];
    if (layers.height) what.push(`height ${num(layers.height[k])} m`);
    if (layers.validity && layers.validity[k] != null) {
      what.push(grid.validity_labels[layers.validity[k]] ?? `validity ${layers.validity[k]}`);
    }
    if (what.length) lines.push(what.join(" · "));
    if (layers.max_sim) {
      const sim = layers.max_sim[k];
      const parts = [layers.driven[k] ? "driven" : "not driven", `similarity ${num(sim)}`];
      if (sim != null && sim > grid.threshold) parts.push(`trusted (> ${grid.threshold.toFixed(2)})`);
      if (layers.max_sim_geom) parts.push(`geometry-only ${num(layers.max_sim_geom[k])}`);
      lines.push(parts.join(" · "));
    }
    read.textContent = lines.join("\n");
  };

  const cellAt = (e) => {
    const box = img.getBoundingClientRect();
    const r = Math.floor(((e.clientY - box.top) / box.height) * grid.rows);
    const c = Math.floor(((e.clientX - box.left) / box.width) * grid.cols);
    return r >= 0 && c >= 0 && r < grid.rows && c < grid.cols ? [r, c] : null;
  };

  const idle = () => {
    cell.hidden = true;
    read.textContent = grid.height_range
      ? `colours span ${grid.height_range[0].toFixed(2)} … ${grid.height_range[1].toFixed(2)} m (2nd-98th percentile of this map)`
      : "hover over the map to read a cell";
  };

  img.addEventListener("mousemove", (e) => {
    if (!grid || pinned) return;
    const at = cellAt(e);
    at ? show(...at) : idle();
  });
  img.addEventListener("mouseleave", () => { if (grid && !pinned) idle(); });
  img.addEventListener("click", (e) => {
    if (!grid) return;
    const at = cellAt(e);
    const same = at && pinned && at[0] === pinned[0] && at[1] === pinned[1];
    pinned = same ? null : at;
    cell.classList.toggle("pinned", !!pinned);
    if (at) show(...at);
  });
  dlg.addEventListener("keydown", (e) => {
    const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!grid || !step) return;
    e.preventDefault();
    const [r, c] = pinned ?? [grid.rows >> 1, grid.cols >> 1];
    pinned = [Math.min(grid.rows - 1, Math.max(0, r + step[0])), Math.min(grid.cols - 1, Math.max(0, c + step[1]))];
    cell.classList.add("pinned");
    show(...pinned);
  });
  // The camera image (and only it) magnifies: the wheel zooms toward the
  // pointer, a drag pans, a double click or "0" goes back to the whole frame.
  // The grids stay 1:1 -- a click there is the probe's, and a cell is already big.
  const stage = dlg.querySelector(".zoomstage");
  const MAX_ZOOM = 8;
  let photo = false;
  let view = { s: 1, x: 0, y: 0 };
  let drag = null;
  const place = (s, x, y) => {
    const w = img.clientWidth;
    const h = img.clientHeight;
    s = Math.min(MAX_ZOOM, Math.max(1, s));
    // Keep the frame covering the stage: no empty margin to pan into.
    view = { s, x: Math.min(0, Math.max(w - w * s, x)), y: Math.min(0, Math.max(h - h * s, y)) };
    img.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.s})`;
    stage.classList.toggle("zoomed", s > 1);
  };
  const zoomAt = (factor, px, py) => {
    const s = Math.min(MAX_ZOOM, Math.max(1, view.s * factor));
    place(s, px - ((px - view.x) * s) / view.s, py - ((py - view.y) * s) / view.s);
  };
  stage.addEventListener("wheel", (e) => {
    if (!photo) return;
    e.preventDefault();
    const box = stage.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - box.left, e.clientY - box.top);
  }, { passive: false });
  stage.addEventListener("pointerdown", (e) => {
    if (!photo || view.s === 1) return;
    e.preventDefault(); // no native image drag
    drag = { px: e.clientX, py: e.clientY, x: view.x, y: view.y };
    stage.setPointerCapture(e.pointerId);
    stage.classList.add("dragging");
  });
  stage.addEventListener("pointermove", (e) => {
    if (drag) place(view.s, drag.x + e.clientX - drag.px, drag.y + e.clientY - drag.py);
  });
  const release = () => { drag = null; stage.classList.remove("dragging"); };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener("dblclick", () => { if (photo) place(1, 0, 0); });
  dlg.addEventListener("keydown", (e) => {
    if (!photo) return;
    const mid = [img.clientWidth / 2, img.clientHeight / 2];
    if (e.key === "+" || e.key === "=") zoomAt(1.25, ...mid);
    else if (e.key === "-") zoomAt(0.8, ...mid);
    else if (e.key === "0") place(1, 0, 0);
    else return;
    e.preventDefault();
  });

  // The cloud toggle, mirrored in the dialog. On the frame page it presses the
  // page's own button (so the choice is shared and remembered) and swaps the
  // zoomed picture to match, leaving the magnification where it was. Where
  // there is no page button -- the 3D view's camera tile -- the dialog keeps
  // the choice itself, for as long as the page is open, and the picture names
  // its projected twin in data-cloud. Off on every page load, like the page's.
  const zoomCloud = document.getElementById("zoomcloud");
  const pageCloud = document.getElementById("cloudtoggle");
  const bigframe = document.getElementById("bigframe");
  // The photo at full width: the frame page's carries a w= to raise, the 3D
  // view's tile asks for the endpoint's 640 px default and gets one added.
  const fullSize = (url) =>
    /[?&]w=\d+/.test(url) ? url.replace(/([?&]w=)\d+/, "$11920") : `${url}${url.includes("?") ? "&" : "?"}w=1920`;
  let ownCloud = false;
  const cloudOn = () => (pageCloud ? pageCloud.getAttribute("aria-pressed") === "true" : ownCloud);
  let current = null;
  // The photo, plain or with the cloud on it and its depth legend over the
  // stage's corner (depthlegend.js) -- outside the picture, so it stays put
  // while the picture magnifies and pans. A newer request wins over a slower one.
  const legend = dlg.querySelector(".depthlegend");
  const plainOf = (src) => fullSize(src.dataset.plain ?? src.src);
  const cloudOf = (src) => (pageCloud ? fullSize(pageCloud.dataset.cloud) : src.dataset.cloud);
  let photoTurn = 0;
  // Resolves once the picture set has loaded (or failed), so following
  // playback can wait for one before asking for the next.
  const loaded = () => new Promise((resolve) => {
    img.addEventListener("load", resolve, { once: true });
    img.addEventListener("error", resolve, { once: true });
  });
  const setPicture = (url) => {
    // The same source again fires no load event, and would hold up following.
    if (new URL(url, location.href).href === img.src) return Promise.resolve();
    const done = loaded();
    dtDepth.setSrc(img, url);
    return done;
  };
  const showPhoto = async (src) => {
    const turn = ++photoTurn;
    if (cloudOn() && cloudOf(src)) {
      try {
        const overlay = await dtDepth.load(cloudOf(src));
        if (turn !== photoTurn) return URL.revokeObjectURL(overlay.src);
        dtDepth.show(legend, overlay.range);
        return setPicture(overlay.src);
      } catch {}
    }
    if (turn !== photoTurn) return;
    dtDepth.show(legend, null);
    return setPicture(plainOf(src));
  };
  zoomCloud?.addEventListener("click", () => {
    if (pageCloud) {
      pageCloud.click();
      img.alt = bigframe.alt;
    } else {
      ownCloud = !ownCloud;
    }
    zoomCloud.setAttribute("aria-pressed", String(cloudOn()));
    shownKey = keyOf(current);
    showPhoto(current);
  });

  // Keys pressed in the dialog are its own: over the 3D view, SimView's
  // shortcuts (arrows, A, C, space...) listen on the window behind it.
  dlg.addEventListener("keydown", (e) => { if (e.key !== "Escape") e.stopPropagation(); });

  // A click on the backdrop (the dialog box itself, outside its content) closes it.
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });

  // What the dialog is showing for its picture, as the URLs it came from: on
  // the 3D view a tile moves on with playback, and the dialog follows it.
  const keyOf = (src) =>
    src.dataset.zoom ? `${src.dataset.zoom}|${src.dataset.grid ?? ""}` : `${plainOf(src)}|${cloudOn() ? cloudOf(src) ?? "" : ""}`;
  let shownKey = null;
  let gridTurn = 0;
  const loadGrid = async (src) => {
    const url = src.dataset.grid;
    if (!url) return;
    const turn = ++gridTurn;
    if (!grids.has(url)) grids.set(url, fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null));
    const g = await grids.get(url);
    if (!dlg.open || turn !== gridTurn) return; // closed or moved on meanwhile
    if (!g) { grid = null; cell.hidden = true; read.textContent = "no cell values for this frame"; return; }
    grid = g;
    // A pinned cell stays pinned across frames, reading the new frame's values.
    pinned ? show(...pinned) : idle();
  };

  // Follow the picture the dialog was opened from as it changes (3D playback),
  // keeping the magnification and the pinned cell. One frame at a time: a
  // frame that arrives while one is loading is skipped for the newest.
  let following = false;
  let followAgain = false;
  const follow = async () => {
    if (!dlg.open || !current) return;
    if (following) { followAgain = true; return; }
    following = true;
    try {
      const src = current;
      const key = keyOf(src);
      if (key !== shownKey) {
        shownKey = key;
        title.textContent = src.alt.replace(/, lidar cloud.*$/, "");
        img.alt = src.alt;
        if (src.dataset.zoom) {
          photoTurn += 1;
          if (src.dataset.grid) grids.clear(); // one frame's grid at a time, not the whole run
          loadGrid(src);
          await setPicture(src.dataset.zoom);
        } else {
          if (zoomCloud) zoomCloud.hidden = !pageCloud && !src.dataset.cloud;
          await showPhoto(src);
        }
      }
    } finally {
      following = false;
    }
    if (followAgain) { followAgain = false; follow(); }
  };

  const open = async (src) => {
    grid = null;
    pinned = null;
    cell.hidden = true;
    cell.classList.remove("pinned");
    photo = !src.dataset.zoom;
    place(1, 0, 0);
    stage.classList.toggle("photo", photo);
    read.hidden = !src.dataset.grid;
    hint.hidden = !src.dataset.grid && !photo;
    hint.textContent = photo ? "scroll or +/− to zoom · drag to pan · double-click or 0 to reset" : gridHint;
    read.textContent = src.dataset.grid ? "loading the cell values…" : "";
    // The heading names the frame only; whether the cloud is on is the button's to say.
    title.textContent = src.alt.replace(/, lidar cloud.*$/, "");
    img.alt = src.alt;
    current = src;
    shownKey = keyOf(src);
    if (photo) showPhoto(src);
    else {
      photoTurn += 1; // a photo still loading must not land on this picture
      dtDepth.setSrc(img, src.dataset.zoom);
      dtDepth.show(legend, null);
    }
    if (zoomCloud) {
      zoomCloud.hidden = !photo || (!pageCloud && !src.dataset.cloud);
      zoomCloud.setAttribute("aria-pressed", String(cloudOn()));
    }
    img.classList.toggle("bev", !!src.dataset.zoom);
    dlg.showModal();
    gridTurn += 1;
    loadGrid(src);
  };

  for (const el of document.querySelectorAll(".zoomable")) {
    el.addEventListener("click", () => open(el));
    // A new picture in the element the dialog came from: show it there too.
    // On the frame page nothing it is keyed on changes, so this does nothing.
    el.addEventListener("load", () => { if (el === current) follow(); });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(el); }
    });
  }
})();
