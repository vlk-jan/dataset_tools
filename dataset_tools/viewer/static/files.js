/* File inventory: panels are server-rendered expanded (no-JS fallback); this
 * collapses them and makes the chips toggle one open panel at a time.
 */

const chips = document.getElementById("filechips");
const panels = Array.from(document.querySelectorAll("[data-fileinfo]"));
if (chips && panels.length) {
  const buttons = Array.from(chips.querySelectorAll("[data-file]"));
  const panelFor = (key) => document.getElementById(`fileinfo-${key}`);

  const close = () => {
    panels.forEach((p) => {
      p.hidden = true;
    });
    buttons.forEach((b) => b.setAttribute("aria-expanded", "false"));
  };
  close();

  chips.addEventListener("click", (event) => {
    const button = event.target.closest("[data-file]");
    if (!button) return;
    const panel = panelFor(button.dataset.file);
    if (!panel) return;
    const wasOpen = button.getAttribute("aria-expanded") === "true";
    close();
    if (!wasOpen) {
      button.setAttribute("aria-expanded", "true");
      panel.hidden = false;
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
}
