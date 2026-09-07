/**
 * The scene page's notes card: compose, edit, delete.
 *
 * The list itself is rendered by the server, so the notes are on the page
 * before this file loads and stay there if it fails. This only adds the three
 * mutations, and each one reloads afterwards rather than patching the DOM --
 * a note changes the windows table's markers too, and one reload is honest
 * about the whole page where hand-updating two places invites them to disagree.
 *
 * An edit reaches every field a note has -- range, topic and text -- because
 * all three are things that get typed wrong: a range pinned one segment too
 * early, a topic picked before reading the rest of the window. Correcting one
 * of those by deleting the note and writing it again would throw away its
 * author and its date, which are the parts nobody can retype.
 */

const card = document.getElementById("notes");
if (card) setup(card);

function setup(card) {
  const scene = card.dataset.scene;
  const base = `/api/scene/${scene}/notes`;
  const nFrames = Number(card.dataset.frames) || 0;
  const form = card.querySelector("#noteform");
  const msg = card.querySelector(".notemsg");

  /** Report what the server said, in the card, rather than in a dialog. */
  function say(text, bad) {
    if (!msg) return;
    msg.textContent = text;
    msg.hidden = !text;
    msg.classList.toggle("bad", Boolean(bad));
  }

  async function send(url, method, body) {
    const init = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) init.body = JSON.stringify(body);
    const response = await fetch(url, init);
    if (!response.ok) {
      let detail = `${response.status}`;
      try {
        const data = await response.json();
        if (data && data.detail) detail = data.detail;
      } catch (error) { /* a non-JSON error body is still a failure */ }
      throw new Error(detail);
    }
    return response.status === 204 ? null : response.json();
  }

  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const start = Number(data.get("frame_start"));
      const end = Number(data.get("frame_end"));
      const text = String(data.get("text") || "").trim();
      if (!text) return say("A note needs some text.", true);
      say("saving…");
      try {
        await send(base, "POST", {
          frames: [start, end],
          topic: data.get("topic"),
          text,
        });
        window.location.reload();
      } catch (error) {
        say(`Could not save: ${error.message}`, true);
      }
    });
  }

  /** One frame-number field of the editor's range. */
  function frameField(value, label) {
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    if (nFrames) input.max = String(nFrames - 1);
    input.inputMode = "numeric";
    input.value = String(value);
    input.setAttribute("aria-label", label);
    return input;
  }

  /**
   * The topic picker, opened on the note's own topic.
   *
   * A topic the server no longer offers is kept as an option rather than
   * silently rewritten to the first one in the list -- the store renders an
   * unknown topic verbatim (notes.TOPICS), and an edit to a note's text should
   * not quietly retag it.
   */
  function topicField(current) {
    const select = document.createElement("select");
    const topics = (card.dataset.topics || "").split(",").filter(Boolean);
    if (current && !topics.includes(current)) topics.unshift(current);
    for (const topic of topics) {
      const option = document.createElement("option");
      option.value = option.textContent = topic;
      option.selected = topic === current;
      select.append(option);
    }
    select.setAttribute("aria-label", "topic");
    return select;
  }

  function makeButton(text, className) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = className;
    el.textContent = text;
    return el;
  }

  /**
   * Swap a note's rendered body for an editor over all of its fields.
   *
   * The head (range chip, topic, author, date) stays put and is left stale
   * until the reload: it is what the note *is* now, next to what it is being
   * changed to, and the reload is a moment away.
   */
  function openEditor(item, id) {
    const paragraph = item.querySelector(".notetext");
    if (!paragraph || item.querySelector(".noteedit")) return; // already editing

    const editor = document.createElement("div");
    editor.className = "noteedit";

    const range = document.createElement("div");
    range.className = "noterow";
    const from = frameField(item.dataset.frameStart, "first frame of the range");
    const to = frameField(item.dataset.frameEnd, "last frame of the range");
    const topic = topicField(item.dataset.topic);
    const fromLabel = document.createElement("label");
    fromLabel.append("frames ", from);
    const toLabel = document.createElement("label");
    toLabel.append("to ", to);
    const topicLabel = document.createElement("label");
    topicLabel.append("topic ", topic);
    range.append(fromLabel, toLabel, topicLabel);

    const box = document.createElement("textarea");
    box.rows = 3;
    box.value = paragraph.textContent.trim();
    box.setAttribute("aria-label", "note text");

    const save = makeButton("Save", "btn primary");
    const cancel = makeButton("Cancel", "btn");
    const actions = document.createElement("div");
    actions.className = "noteeditactions";
    actions.append(save, cancel);

    editor.append(range, box, actions);
    paragraph.replaceWith(editor);
    box.focus();

    cancel.addEventListener("click", () => {
      editor.replaceWith(paragraph);
      say("");
    });

    save.addEventListener("click", async () => {
      const text = box.value.trim();
      if (!text) return say("A note needs some text.", true);
      // Both ends go up together even when only one was touched: the server
      // orders a backwards pair, so typing them the other way round is not an
      // error here either.
      const start = Number(from.value);
      const end = Number(to.value);
      if (!Number.isFinite(start) || !Number.isFinite(end)) {
        return say("A note needs two frame numbers.", true);
      }
      say("saving…");
      try {
        await send(`${base}/${id}`, "PATCH", {
          frames: [start, end],
          topic: topic.value,
          text,
        });
        window.location.reload();
      } catch (error) {
        say(`Could not save: ${error.message}`, true);
      }
    });
  }

  card.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-act]");
    if (!button) return;
    const item = button.closest("li[data-note-id]");
    if (!item) return;
    const id = item.dataset.noteId;

    if (button.dataset.act === "delete") {
      if (!window.confirm("Delete this note?")) return;
      try {
        await send(`${base}/${id}`, "DELETE");
        window.location.reload();
      } catch (error) {
        say(`Could not delete: ${error.message}`, true);
      }
      return;
    }

    if (button.dataset.act === "edit") openEditor(item, id);
  });
}
