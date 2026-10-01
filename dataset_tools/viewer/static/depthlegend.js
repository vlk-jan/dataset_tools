// The cloud-on-image picture and its depth legend. The server sends the depth
// the colours run over in an X-Depth-Range header instead of painting a legend
// into the picture, and the page draws it over the picture's corner: painted
// in, it zoomed and panned away with the picture. Fetched rather than set as a
// plain src, since an <img> cannot read a response header.
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
    // The far end is the 98th percentile, not the farthest return: everything
    // beyond it is the same dark red, and "+" says so.
    legend.querySelector("[data-near]").textContent = `${range[0].toFixed(1)} m`;
    legend.querySelector("[data-far]").textContent = `${range[1].toFixed(1)} m+`;
  },
};
