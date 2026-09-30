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

async function swap(region, url, push) {
  region.setAttribute("aria-busy", "true");
  try {
    const response = await fetch(url, { headers: { Accept: "text/html" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = new DOMParser().parseFromString(await response.text(), "text/html");
    const fresh = doc.getElementById(region.id);
    if (!fresh) throw new Error(`no #${region.id} in the response`);
    region.replaceWith(fresh);
    if (push) history.pushState({ swap: region.id }, "", url);
  } catch (error) {
    console.warn("in-place swap failed, loading the page instead:", error);
    window.location.assign(url);
  }
}

document.addEventListener("click", (event) => {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target.closest(SWAP_LINKS);
  const region = link && link.closest("[data-swap]");
  if (!region) return;
  event.preventDefault();
  swap(region, link.href, true);
});

// Back and forward: the address already says which tab or page; fetch it and
// swap the region, exactly as the click did.
window.addEventListener("popstate", () => {
  for (const region of document.querySelectorAll("[data-swap]")) swap(region, window.location.href, false);
});
