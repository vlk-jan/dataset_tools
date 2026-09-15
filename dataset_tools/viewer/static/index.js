/**
 * The index's train/validation filter.
 *
 * Every card is on the page before this file loads, each marked with the side
 * of the split it is on (`data-split`), so this only shows and hides: nothing
 * is fetched, and if the file fails to load every scene simply stays listed.
 * A card the split file says nothing about has no `data-split` and is hidden
 * by any filter but "All" -- it is on neither side, so it belongs to neither.
 *
 * The choice is kept in the address (`?split=val`), so a filtered index can be
 * linked and survives a reload.
 */

const panel = document.getElementById("splits");
if (panel) setup(panel);

function setup(panel) {
  const buttons = [...panel.querySelectorAll("button[data-split]")];
  const known = new Set(buttons.map((b) => b.dataset.split));
  const sections = [...document.querySelectorAll("main > section")].map((section) => {
    const head = section.querySelector(".shead");
    let shown = null;
    if (head) {
      shown = document.createElement("span");
      shown.className = "sshown";
      shown.hidden = true;
      head.append(shown);
    }
    return { section, cards: [...section.querySelectorAll("article.scene")], shown };
  });

  function apply(which, remember) {
    if (!known.has(which)) which = "";
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.split === which));
    for (const { section, cards, shown } of sections) {
      let visible = 0;
      for (const card of cards) {
        card.hidden = Boolean(which) && card.dataset.split !== which;
        if (!card.hidden) visible += 1;
      }
      // A group with nothing left in it would be a heading over empty space.
      section.hidden = cards.length > 0 && visible === 0;
      if (shown) {
        shown.textContent = `${visible} of ${cards.length} shown`;
        shown.hidden = !which;
      }
    }
    if (remember) {
      const url = new URL(window.location.href);
      if (which) url.searchParams.set("split", which);
      else url.searchParams.delete("split");
      window.history.replaceState(null, "", url);
    }
  }

  panel.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-split]");
    if (button) apply(button.dataset.split, true);
  });
  apply(new URLSearchParams(window.location.search).get("split") || "", false);
}
