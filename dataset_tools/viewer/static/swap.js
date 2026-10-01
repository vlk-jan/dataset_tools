/* In-place switching for the scene page's tabs and the frame strips' pagers.
 *
 * The tabs and the pager are ordinary links to the same server-rendered page
 * with a different query, and without this file that is exactly what they
 * are: the page reloads and scrolls back to the top. Here a click on one
 * fetches that page and swaps in only the region the link lives in -- the
 * element marked `data-swap`, found again by its id in the response -- so the
 * cards above it and the scroll position stay put. The address is updated so
 * the result can still be linked, and the back button walks it. Anything else
 * (a frame tile, which opens a different page) is left to the browser, and so
 * is a click with a modifier, which means "in a new tab". If the fetch fails
 * the link is followed the ordinary way.
 */

const SWAP_LINKS = "a.tab, .pager a";

/** The request in flight per region: a newer click on the same region aborts
 *  the older one, so the address and the content can never end up from two
 *  different clicks. */
const pending = new Map();
/** What the swapped regions show now, so back after an in-page anchor
 *  (#notes) does not refetch a page that has not changed. */
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
    // Looked up again: the region from before the fetch may have been
    // replaced by another swap in the meantime.
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

/**
 * Re-render the regions with these ids from the current address, in place.
 * For a change made on this page (a note saved) that shows in more than one
 * place; the server renders every place, so they cannot disagree. Throws if
 * the page cannot be fetched -- the caller then reloads.
 */
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

// Back and forward: the address already says which tab or page; fetch it and
// swap the region, exactly as the click did -- unless only the #fragment moved.
window.addEventListener("popstate", () => {
  const now = window.location.pathname + window.location.search;
  if (now === shown) return;
  for (const region of document.querySelectorAll("[data-swap]")) swap(region.id, window.location.href, false);
});
