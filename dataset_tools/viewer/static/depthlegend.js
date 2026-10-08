// Cloud-on-image picture plus its depth legend, from the X-Depth-Range header
// (fetched, since an <img> can't read headers).
window.dtDepth = {
  async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} for ${url}`);
    const range = (r.headers.get("X-Depth-Range") || "").split(" ").map(Number);
    return {
      src: URL.createObjectURL(await r.blob()),
      range: range.length === 2 && range.every(Number.isFinite) ? range : null,
    };
  },
  // Swap an image's source, letting go of the object URL it had, if any.
  setSrc(img, src) {
    if (img.src.startsWith("blob:") && img.src !== src) URL.revokeObjectURL(img.src);
    img.src = src;
  },
  // Fill a .depthlegend, or hide it when there is no range to state.
  show(legend, range) {
    if (!legend) return;
    legend.hidden = !range;
    if (!range) return;
    // The far end is the 98th percentile, hence "+".
    legend.querySelector("[data-near]").textContent = `${range[0].toFixed(1)} m`;
    legend.querySelector("[data-far]").textContent = `${range[1].toFixed(1)} m+`;
  },
};
