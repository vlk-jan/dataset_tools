/* GNSS track on an OpenStreetMap backdrop, for each `<div data-map data-src>`.
 * Leaflet is vendored for offline use; without tiles the track is drawn on a
 * plain backdrop and the map says so.
 */

const OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

/* Enough failures to mean "there is no tile server", not "one tile timed out". */
const MAX_TILE_ERRORS = 4;
/* Zoom used when a recording never moved: bounds of zero extent cannot be fitted. */
const STANDING_STILL_ZOOM = 18;
/* The deepest OpenStreetMap raster zoom; a short outing is fitted to it. */
const MAX_TILE_ZOOM = 19;
/* Metres of bounding box under which a track counts as standing still. */
const STANDING_STILL_M = 5;

function note(el, text) {
  const target = el.parentElement?.querySelector(".mapnote");
  if (!target) return;
  target.textContent = text;
  target.hidden = false;
}

function marker(latlng, cls, title) {
  return L.marker(latlng, {
    title,
    keyboard: false,
    icon: L.divIcon({ className: `mapdot ${cls}`, iconSize: [13, 13], iconAnchor: [6.5, 6.5] }),
  });
}

/** Whether a bounding box is large enough to fit a view to. */
function hasExtent(bounds) {
  if (!bounds) return false;
  const [[s, w], [n, e]] = bounds;
  const mid = ((s + n) / 2) * (Math.PI / 180);
  return Math.hypot((n - s) * 111320, (e - w) * 111320 * Math.cos(mid)) > STANDING_STILL_M;
}

async function draw(el) {
  let data;
  try {
    const response = await fetch(el.dataset.src, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
  } catch (err) {
    note(el, `The track could not be loaded: ${err.message}.`);
    return;
  }

  const track = data.points || [];
  const context = data.context || [];
  if (!track.length && !context.length && !data.marker) {
    note(el, "No GNSS fixes inside this stretch.");
    return;
  }

  const map = L.map(el, { scrollWheelZoom: false, zoomSnap: 0.5 });
  // Leaflet only notices window resizes on its own.
  new ResizeObserver(() => map.invalidateSize()).observe(el);
  /* Wheel zoom only after a click, so the map doesn't trap page scrolling. */
  map.once("focus click", () => map.scrollWheelZoom.enable());

  let tileErrors = 0;
  const tiles = L.tileLayer(OSM_URL, { maxZoom: MAX_TILE_ZOOM, attribution: OSM_ATTRIBUTION });
  tiles.on("tileerror", () => {
    if (++tileErrors === MAX_TILE_ERRORS) {
      note(el, "No map tiles could be fetched - the track is drawn on a plain backdrop.");
      el.classList.add("notiles");
    }
  });
  tiles.addTo(map);

  /* The rest of the scene, faint, for context. */
  if (context.length) {
    L.polyline(context, { className: "trackctx", interactive: false }).addTo(map);
  }
  if (track.length) {
    L.polyline(track, { className: "track", interactive: false }).addTo(map);
    marker(track[0], "start", "start of this stretch").addTo(map);
    marker(track[track.length - 1], "end", "end of this stretch").addTo(map);
  } else if (context.length) {
    note(el, "No GNSS fixes inside this stretch; the rest of the scene is shown.");
  }
  if (data.marker) {
    marker(data.marker, "here", "this frame").addTo(map);
  }

  const focus = track.length ? track : context;
  if (hasExtent(data.bounds)) {
    /* Fit a stretch inside a scene less tightly, so the backdrop stays useful. */
    const maxZoom = context.length ? 17 : MAX_TILE_ZOOM;
    map.fitBounds(L.latLngBounds(data.bounds), { padding: [24, 24], maxZoom });
  } else if (focus.length) {
    map.setView(data.marker || focus[0], STANDING_STILL_ZOOM);
  }
  /* Never past the tile server's deepest zoom. */
  if (map.getZoom() > MAX_TILE_ZOOM) map.setZoom(MAX_TILE_ZOOM);
}

document.querySelectorAll("[data-map]").forEach(draw);
