/**
 * The pieces the two notes editors share: the scene page's card (notes.js)
 * and the 3D view's panel (overlay.js).
 */

/** One note mutation. Throws carrying the server's own reason, which is the
 *  useful half of a refusal ("this root is not writable", not "403"). */
export async function noteRequest(url, method, body) {
  const init = { method, headers: { "Content-Type": "application/json" } };
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

/**
 * The topic picker, opened on the note's own topic.
 *
 * A topic the server no longer offers is kept as an option rather than
 * silently rewritten to the first one in the list -- the store renders an
 * unknown topic verbatim (notes.TOPICS), and an edit to a note's text should
 * not quietly retag it.
 */
export function topicSelect(topics, current) {
  const select = document.createElement("select");
  const names = topics.slice();
  if (current && !names.includes(current)) names.unshift(current);
  for (const name of names) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    option.selected = name === current;
    select.append(option);
  }
  select.setAttribute("aria-label", "topic");
  return select;
}

/** One frame-number field of an editor's range; `max` is the last frame, if known. */
export function frameField(value, label, max) {
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  if (max != null && max !== "") input.max = String(max);
  input.inputMode = "numeric";
  input.value = String(value);
  input.setAttribute("aria-label", label);
  return input;
}

export function makeButton(text, className) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = className;
  el.textContent = text;
  return el;
}
