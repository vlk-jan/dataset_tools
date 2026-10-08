/* Shared crosshair, readout and click-to-frame for a [data-signals] card's
 * server-rendered charts, which all share one time axis. */

function fmtTime(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.floor(s - 60 * m)} s`;
}

function fmtValue(v) {
  if (v === null || v === undefined) return "-";
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
}

function wire(card) {
  const charts = [...card.querySelectorAll(".sig")].map((el) => ({
    el,
    data: JSON.parse(el.dataset.series),
    cursors: el.querySelectorAll("line.cursor"),
    out: el.querySelector("[data-sigval]"),
  }));
  if (!charts.length) return;
  const bins = charts[0].data.series[0].v.length;
  const frames = card.dataset.frames ? JSON.parse(card.dataset.frames) : null;
  const duration = Number(card.dataset.duration);
  // Time (s) of the charts' left edge, nonzero on a frame page.
  const offset = Number(card.dataset.offset ?? 0);
  const time = card.querySelector("[data-sigtime]");
  const idle = time.textContent;
  let column = null;

  const show = (col) => {
    column = col;
    for (const c of charts) {
      for (const line of c.cursors) {
        line.setAttribute("x1", col === null ? -10 : col + 0.5);
        line.setAttribute("x2", col === null ? -10 : col + 0.5);
      }
      c.out.textContent =
        col === null
          ? ""
          : c.data.series.map((s) => `${c.data.series.length > 1 ? s.label + " " : ""}${fmtValue(s.v[col])}`).join(" · ") +
            ` ${c.data.unit}`;
    }
    if (col === null) time.textContent = idle;
    else {
      const at = fmtTime(offset + ((col + 0.5) / bins) * duration);
      time.textContent = frames ? `${at} · frame ${frames[col]}` : at;
    }
  };

  const columnAt = (event, plot) => {
    const box = plot.getBoundingClientRect();
    const f = (event.clientX - box.left) / box.width;
    return Math.max(0, Math.min(bins - 1, Math.floor(f * bins)));
  };

  for (const plot of card.querySelectorAll(".plot")) {
    plot.addEventListener("pointermove", (e) => show(columnAt(e, plot)));
    plot.addEventListener("pointerleave", () => show(null));
    if (frames) {
      plot.style.cursor = "pointer";
      plot.addEventListener("click", (e) => {
        const col = columnAt(e, plot);
        window.location.assign(`/scene/${card.dataset.scene}/frame/${frames[col]}`);
      });
    }
  }
  // Keyboard: the card is focusable, and the arrow keys walk the columns.
  card.tabIndex = 0;
  card.addEventListener("keydown", (e) => {
    // A focused chart header's keys are its own: Enter/Space fold it.
    if (e.target !== card) return;
    const step = e.shiftKey ? 20 : 1;
    if (e.key === "ArrowRight") show(Math.min(bins - 1, (column ?? -1) + step));
    else if (e.key === "ArrowLeft") show(Math.max(0, (column ?? bins) - step));
    else if (e.key === "Enter" && column !== null && frames)
      window.location.assign(`/scene/${card.dataset.scene}/frame/${frames[column]}`);
    else if (e.key === "Escape") show(null);
    else return;
    e.preventDefault();
  });
}

for (const card of document.querySelectorAll("[data-signals]")) wire(card);
