/** Shared by the notes editors in notes.js and overlay.js. */

/** JSON API request; throws with the server's `detail` when there is one. */
export async function noteRequest(url, method, body) {
  const init = { method, cache: "no-store", headers: { "Content-Type": "application/json" } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const parsed = await response.json();
      if (parsed && parsed.detail) detail = parsed.detail;
    } catch (e) { /* the status alone is enough */ }
    throw new Error(detail);
  }
  return response;
}

/** Select `value`; one the picker doesn't offer is kept as an option, not rewritten. */
export function keepOption(select, value) {
  if (![...select.options].some((o) => o.value === value)) select.add(new Option(value, value), 0);
  select.value = value;
}
