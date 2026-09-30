// Light/dark theme. Loaded as a plain blocking script in <head>, so the saved
// or OS theme is on <html> before the first paint -- a deferred module would
// flash the light page at every dark-mode reader on every navigation. Same
// contract as the author's web page: localStorage "theme" is "light"/"dark",
// absent means follow the OS.
(() => {
  const svg = (body) =>
    `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
  const SUN = svg(
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>'
  );
  const MOON = svg('<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>');
  const html = document.documentElement;
  const os = window.matchMedia("(prefers-color-scheme: dark)");
  const saved = () => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  };
  const apply = (t) => {
    if (t === "dark") html.setAttribute("data-theme", "dark");
    else html.removeAttribute("data-theme");
    const btn = document.querySelector(".theme-toggle");
    if (btn) {
      // Drawn, not typed: ☀/☾ are missing from many system fonts.
      btn.innerHTML = t === "dark" ? SUN : MOON;
      btn.title = t === "dark" ? "Switch to light theme" : "Switch to dark theme";
    }
  };
  const current = () => {
    const t = saved();
    return t === "dark" || t === "light" ? t : os.matches ? "dark" : "light";
  };
  apply(current());
  os.addEventListener("change", () => apply(current()));

  document.addEventListener("DOMContentLoaded", () => {
    const header = document.querySelector("body > header");
    if (!header) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    btn.setAttribute("aria-label", "Toggle dark/light theme");
    btn.addEventListener("click", () => {
      const t = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
      try { localStorage.setItem("theme", t); } catch {}
      apply(t);
    });
    header.append(btn);
    apply(current());
  });
})();
