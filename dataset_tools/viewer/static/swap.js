/* In-place switching for tabs and pagers: fetch the linked page and swap in
 * only the enclosing `data-swap` region (by id), keeping scroll position and
 * updating history. Falls back to normal navigation on failure.
 */

const SWAP_LINKS = "a.tab, .pager a";

/** In-flight request per region; a newer click aborts the older one. */
const pending = new Map();
/** What the regions show now, so a #fragment-only popstate doesn't refetch. */
let shown = window.location.pathname + window.location.search;

async function fetchPage(url, signal) {
  const response = await fetch(url, { headers: { Accept: "text/html" }, signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return new DOMParser().parseFromString(await response.text(), "text/html");
}

async function swap(id, url, push) {
  pending.get(id)?.abort();
  const controller = new AbortController();
  pending.set(id, controller);
  document.getElementById(id)?.setAttribute("aria-busy", "true");
  try {
    const doc = await fetchPage(url, controller.signal);
    const fresh = doc.getElementById(id);
    // Looked up again: another swap may have replaced it meanwhile.
    const region = document.getElementById(id);
    if (!fresh || !region) throw new Error(`no #${id} in the response`);
    region.replaceWith(fresh);
    if (push) history.pushState({ swap: id }, "", url);
    const at = new URL(url, window.location.href);
    shown = at.pathname + at.search;
  } catch (error) {
    if (error.name === "AbortError") return; // a newer click on this region took over
    console.warn("in-place swap failed, loading the page instead:", error);
    window.location.assign(url);
  } finally {
    if (pending.get(id) === controller) pending.delete(id);
  }
}

/** Re-render the regions with these ids from the current address. Throws on fetch failure. */
export async function refresh(ids) {
  const doc = await fetchPage(window.location.href);
  for (const id of ids) {
    const fresh = doc.getElementById(id);
    const region = document.getElementById(id);
    if (fresh && region) region.replaceWith(fresh);
  }
}

document.addEventListener("click", (event) => {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target.closest(SWAP_LINKS);
  const region = link && link.closest("[data-swap]");
  if (!region) return;
  event.preventDefault();
  swap(region.id, link.href, true);
});

// Back/forward: swap the regions to the new address unless only the #fragment moved.
window.addEventListener("popstate", () => {
  const now = window.location.pathname + window.location.search;
  if (now === shown) return;
  for (const region of document.querySelectorAll("[data-swap]")) swap(region.id, window.location.href, false);
});
