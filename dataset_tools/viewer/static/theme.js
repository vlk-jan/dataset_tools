// Light/dark theme. A blocking script in <head>, so it applies before first
// paint. localStorage "theme" is "light"/"dark"; absent follows the OS.
// data-theme is always set: a few rules (map tiles, the Open 3D fill) key on it.
(() => {
  const html = document.documentElement;
  const os = window.matchMedia("(prefers-color-scheme: dark)");
  const saved = () => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  };
  const apply = (t) => {
    html.dataset.theme = t === "dark" || t === "light" ? t : os.matches ? "dark" : "light";
  };
  apply(saved());
  os.addEventListener("change", () => apply(saved()));

  // The toggle is base.html's; theme.css shows the icon for the other theme.
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".theme-toggle")) return;
    const t = html.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("theme", t); } catch {}
    apply(t);
  });

  document.addEventListener("DOMContentLoaded", () => {
    const header = document.querySelector("body > header");
    if (!header) return;
    new ResizeObserver(() =>
      html.style.setProperty("--headh", `${header.offsetHeight}px`)).observe(header);
  });
})();
