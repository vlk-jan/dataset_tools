/* The file inventory on a scene page: an entry opens in place.
 *
 * Progressive on purpose. The server renders every panel expanded, so a page
 * with no JavaScript (or one that failed to load this module) is a readable
 * list of what each artefact is; this collapses them and turns the strip into
 * a set of toggles. One panel is open at a time -- the card sits beside the
 * map in a two-column pair, and stacking panels would push the map's height
 * around every click.
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
